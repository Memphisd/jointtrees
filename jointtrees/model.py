"""The model: one evaluation for every problem and every tree.  Standard library only.

A tree has the base merge (join 0) and the joins 1..d.  Join i combines two lists of level i-1 into a list of level
i, and level d is the solution.  All quantities are exponents per n in bits: a list of level i has 2^(L_i n)
elements.  `evaluate` is the whole model, in the order of the paper:

    parts   = problem.parts(inst, prm)         the solution and its label cut into parts (parts.py, problems/)
    shares  = contributions(inst, prm, parts)  [lemma 1] top-down: what every part contributes at every join
    sums    = per_join(parts, shares)          r_i, c_i, c_label_i, ell_i, final_i per join, K per part, joint types
    list_sizes                                 [list sizes]: D_i, L_i and the fixed label bits
    repetitions                                [budget], [repetitions]: the deficits Delta_i
    problem.runs                               runs of the whole tree (decoding: [isd])
    base_merge_cost, join_cost                 [dissection], [plain join], [filtered join]: the cost terms
    constraints                                the only place where constraints are collected (the problem adds its own)
    time = max(terms) + runs, memory = max(lists)                                           [tree cost]

Nothing here depends on the problem: what differs between subset sum, LWE and decoding enters through the parts,
the runs and the problem's constraints.

Paper locations are named in brackets, and their numbers are in references.py.  The parameters `prm` of a tree are
a plain dict (the model's dict), and build.variables documents its schema.
"""
from dataclasses import dataclass, field

from .build import filter_of
from .entropy import H
from .filters import filter_exponents
from .parts import label, partition, split_name
from .record import Record
from .references import ref


class NoFeasiblePoint(RuntimeError):
    """optimize found no parameters that satisfy the constraints of the model."""


@dataclass
class Constraint:
    name: str
    value: float                    # the constraint is value >= 0
    group: str                      # 'model' (stated in the paper) or 'domain' (a quantity is in its range)
    reference: str = ''             # where the paper states it
    boxed: bool = False             # the optimizer enforces it through the bounds of its variables


@dataclass
class JoinSums:
    """The per-join totals of [lemma 1], summed over the parts."""
    r: list                         # representations gained at join i
    c: list                         # consistency loss at join i: every part split by a joint type there
    c_label: list                   # loss from the target condition at join i
    ell: list                       # label bits matched exactly at join i
    final: list                     # the final weight check on `rest` (-log2 Q), non-zero at the top join only
    residual: list                  # maxent joint types: deviation from the required sums at join i (0 in the domain)
    K: dict                         # per part: its constraints on the representations of the solution
    J: list                         # per join: {part name: joint type of the two children}


@dataclass
class Evaluation:
    time: float = 0.0
    memory: float = 0.0             # largest stored list (the top list is not stored)
    runs: float = 0.0               # log2 of the runs of the whole tree, per n (decoding: the permutations)
    budget: float = 0.0             # representations of the solution minus constraints on them
    final_check: float = 0.0        # -log2 Q: a candidate of the top list passes the weight check on `rest` w.p. Q
    levels: list = field(default_factory=list)      # per level 0..d: D, L, ell, delta, type of the tree part
    joins: list = field(default_factory=list)       # per join 0..d: r, c, c_label, ell, J, filters, alpha, beta, terms
    terms: dict = field(default_factory=dict)       # every cost term with its repetitions (not the runs)
    records: list = field(default_factory=list)     # every Constraint, in the order of `constraints`
    checks: dict = field(default_factory=dict)      # deviations that are ~0 by construction
    part_sizes: dict = field(default_factory=dict)  # size of every tree part (fraction of n), one without partitioning
    by_part: dict = field(default_factory=dict)     # per tree part: types, joint types, D, r, c, alpha, beta
    kappa: float = None             # decoding: size of the window (fraction of n)
    omega_1: float = None           # decoding: weight of the error on the window (fraction of n)
    unit: float = 1.0               # bits per label unit (log2 q for decoding): time / unit is the exponent in base q
    digits: tuple = ()
    shift: tuple = None             # rule 4: (join j, {a: a + 1}), from level j up the types are those of e - (1,...,1)
    memory_0: float = None          # memory of join 0: a half of a base list, or the 7-dissection's

    @property
    def constraints(self):
        """{constraint name: slack}.  Feasible means every slack >= 0."""
        return {c.name: c.value for c in self.records}

    @property
    def violations(self):
        """{constraint name: slack} of the constraints that do not hold (slack below -1e-6, the tolerance of verify)."""
        return {c.name: c.value for c in self.records if c.value < -1e-6}

    @property
    def feasible(self):
        """True if every constraint of the model holds (up to 1e-6)."""
        return not self.violations

    @property
    def memories(self):
        """{list: size}: every stored list (the halves of a base list or the 7-dissection's memory, L_0 .. L_{d-1}).
        memory is the largest."""
        return {'join 0': self.memory_0, **{f'L_{i}': v['L'] for i, v in enumerate(self.levels[:-1])}}

    # ---- the paper's quantities, per level or join i = 0..d (lists over i)
    @property
    def L(self):
        """L^(i): list sizes.  L^(d) is the top list before the final check."""
        return [v['L'] for v in self.levels]

    @property
    def D(self):
        """D^(i): entropy of the tree part on level i (summed over the parts)."""
        return [v['D'] for v in self.levels]

    @property
    def Delta(self):
        """Δ^(i): repetitions of the joins of level i and above."""
        return [v['delta'] for v in self.levels]

    @property
    def r(self):
        """r^(i): representations gained at join i (summed over the parts)."""
        return [j['r'] for j in self.joins]

    @property
    def c(self):
        """c^(i): consistency loss at join i (summed over the parts)."""
        return [j['c'] for j in self.joins]

    @property
    def alpha(self):
        """α of join i (summed over its filtered parts)."""
        return [j['alpha'] for j in self.joins]

    @property
    def beta_x(self):
        """β_x of join i."""
        return [j['beta_x'] for j in self.joins]

    @property
    def beta_y(self):
        """β_y of join i."""
        return [j['beta_y'] for j in self.joins]

    @property
    def T(self):
        """T^(i): cost of join i ([plain join], [filtered join]), without the repetitions Δ^(i)."""
        return [max(j['terms'].values()) for j in self.joins]

    @property
    def tau(self):
        """τ = max_i (T^(i) + Δ^(i)): the cost of one run.  time = τ + runs."""
        return max(self.terms.values())

    @property
    def ell(self):
        """ℓ^(0), ..., ℓ^(d-1): label bits matched exactly at the joins below the top."""
        return [j['ell'] for j in self.joins[:-1]]

    @property
    def ell_d(self):
        """ℓ^(d): label bits matched exactly at the top join."""
        return self.joins[-1]['ell']

    @property
    def F(self):
        """The time exponent in base q (decoding over F_q: q^(F n) = 2^(time n)), for subset sum and LWE the time."""
        return self.time / self.unit

    def paper_type(self, i, t):
        """The type of level i as the paper writes it: with rule 4, from the shifted join up, the type of
        e - (1,...,1)."""
        if self.shift is None or i < self.shift[0]:
            return dict(sorted(t.items()))
        return {a: t.get(up, 0.0) for a, up in sorted(self.shift[1].items())}

    def nested(self, J):
        """A joint type {(a, b): share} as rows {a: {b: share}} over all digits."""
        if J is None:
            return None
        return {a: {b: J.get((a, b), 0.0) for b in self.digits} for a in self.digits}

    def _one_part(self, name):
        if len(self.by_part) > 1:
            raise AttributeError(f'this tree has {len(self.by_part)} parts: use evaluation.parts[k].{name}, '
                                 f'k = 1..{len(self.by_part)}')
        return next(iter(self.by_part.values()))

    @property
    def p(self):
        """p^(i): type of level i, as the paper writes it (with partitioning: evaluation.parts[k].p)."""
        return [self.paper_type(i, t) for i, t in enumerate(self._one_part('p')['types'])]

    @property
    def J(self):
        """J^(i): joint type of join i as rows J[i][a][b], None at join 0 and at split joins (with partitioning:
        evaluation.parts[k].J)."""
        return [self.nested(J) for J in self._one_part('J')['J']]

    @property
    def parts(self):
        """With partitioning: {k: per-part p, J, size, D, r, c, alpha, beta_x, beta_y} for k = 1..P."""
        out = {}
        for name, part in self.by_part.items():
            k = split_name(name)[1]
            if k is not None:
                out[k] = Record(size=part['size'], p=[self.paper_type(i, t) for i, t in enumerate(part['types'])],
                                J=[self.nested(J) for J in part['J']],
                                **{key: part[key] for key in ('D', 'r', 'c', 'alpha', 'beta_x', 'beta_y')})
        return out


# ----------------------------------------------------------------------------------- step 1: what each part contributes
def contributions(inst, prm, parts):
    """[lemma 1], top-down.  Every part has the type of the solution (or target) on level d.  At every join from the
    top down to the join that introduces it, it contributes a Share and passes its type one level down.  Returns
    shares[i][part name]."""
    d = inst.depth
    shares = [dict() for _ in range(d + 1)]
    for p in parts:
        p.types, p.sizes = {d: p.type}, {d: p.size}
        for i in range(d, p.intro - 1, -1):
            share = p.share(i, inst, prm)
            shares[i][p.name] = share
            if share.child is not None:
                p.types[i - 1], p.sizes[i - 1] = share.child
    return shares


def per_join(parts, shares):
    """The totals of the shares: per join the sums r_i, c_i, c_label_i, ell_i, final_i and the joint types, per part
    K."""
    joins = range(len(shares))
    sums = JoinSums(r=[0.0 for _ in joins], c=[0.0 for _ in joins], c_label=[0.0 for _ in joins],
                    ell=[0.0 for _ in joins], final=[0.0 for _ in joins], residual=[0.0 for _ in joins],
                    K={p.name: 0.0 for p in parts}, J=[dict() for _ in joins])
    for i, at_join in enumerate(shares):
        for name, s in at_join.items():
            sums.r[i] += s.r
            sums.c[i] += s.c
            sums.c_label[i] += s.c_label
            sums.ell[i] += s.ell
            sums.final[i] += s.final
            sums.residual[i] = max(sums.residual[i], s.residual)
            sums.K[name] += s.K
            if s.J is not None:
                sums.J[i][name] = s.J
    return sums


# -------------------------------------------------------------------------------------------- step 2: list sizes
def list_sizes(inst, parts, sums):
    """[list sizes].  D_i is the entropy of the tree part on level i, and fixed_i are the label bits fixed on a list
    of level i (for [saturation]: the syndrome part fixes none, since its weight check is the final check).  The two
    halves of a base list are plain enumerations:

        L_half = D_0 / 2,     L_0 = 2 L_half - c_label_0          (c_0 = 0: join 0 halves the tree part),
        L_i    = 2 L_{i-1} - c_i - c_label_i                     (1 <= i <= d)

    The top row L_d is the size of the top list before the final weight check on the syndrome part (sums.final[d] =
    -log2 Q, [many solutions]): the candidates per run that pass every other check, apart from the planted
    solution.  It is negative when fewer than one false candidate survives per run."""
    d, u = inst.depth, inst.unit
    D = [sum(t.sizes[i] * H(t.types[i]) for t in parts.trees) for i in range(d + 1)]     # summed over the parts
    fixed = [sum(p.sizes[i] * (u - H(p.types[i])) for p in parts.exact if i in p.types) for i in range(d + 1)]
    half = D[0] / 2
    L = [2 * half - sums.c[0] - sums.c_label[0]]
    for i in range(1, d + 1):
        L.append(2 * L[i - 1] - sums.c[i] - sums.c_label[i])
    return D, fixed, half, L


# --------------------------------------------------------------------------- step 3: representations of the solution
def repetitions(inst, parts, sums):
    """[budget], [repetitions].  The solution has 2^(R n) representations in the whole tree, R = sum_i 2^(d-i) r_i,
    and the label parts put sum_parts K_part constraints on them.  budget = R - sum K.

        accounting 'budget':          budget >= 0 (a constraint, see `constraints`): one representation survives
        accounting 'repetitions':     the joins of level i and above are repeated 2^(Delta_i n) times,
                                      Delta_i = max(0, sum_{parts introduced at joins <= i} K_part - R)
        accounting 'many-solutions':  no condition, and a deficit lowers the size of the top list

    The paper couples the budget to "no memory bound" and the repetitions to "a memory bound".  Here the accounting is
    chosen in the Tree (`accounting='repetitions'`), and a memory bound may be combined with either (the budget is
    then the stricter choice)."""
    d = inst.depth
    R = sum(2 ** (d - i) * sums.r[i] for i in range(1, d + 1))
    budget = R - sum(sums.K.values())
    if inst.accounting == 'repetitions':
        return budget, [max(0.0, sum(sums.K[p.name] for p in parts if p.intro <= i) - R) for i in range(d + 1)]
    return budget, [0.0] * (d + 1)


# -------------------------------------------------------------------------------------------- step 4: join costs
def base_merge_cost(inst, D, half, ell):
    """Join 0 with exact labels.  Two halves: enumerate them (2^(L_half n) each) and match them on the labels.
    [dissection]: with memory 2^(M_0 n), M_0 = min(M, D_0/4), the time is max(D_0/2, 2 (D_0 - M_0)/3).
    Returns (cost terms, memory used).  The third term of [dissection], writing the output L_0, is added by
    `evaluate` as the term 'output L_0' of every join."""
    if not inst.tree.dissection:
        return {'input': half, 'label matches': 2 * half - ell}, half
    M0 = D[0] / 4 if inst.memory_bound is None else min(inst.memory_bound, D[0] / 4)
    return {'dissection D_0/2': D[0] / 2, 'dissection 2(D_0-M_0)/3': 2 * (D[0] - M0) / 3}, M0


def join_cost(L_in, ell, records):
    """[plain join], [filtered join].  Without a filter: read the input lists and enumerate the pairs that match on
    the ell label bits, max(L, 2L - ell).  With filters:

        routing (left list)    L + alpha - beta_x
        routing (right list)   L + alpha - beta_y
        candidates             2L - ell + alpha - beta_x - beta_y

    alpha, beta_x, beta_y are summed over the parts filtered at this join (e.g. the tree part and `rest` at a
    far-neighbour top join): a center is then the product of one center per part, so the exponents add
    ([filter blocks] uses the same product structure)."""
    alpha, beta_x, beta_y = (sum(x[k] for x in records) for k in ('alpha', 'beta_x', 'beta_y'))
    if all(x['kind'] == 'none' for x in records):
        return {'input': L_in, 'label matches': 2 * L_in - ell}
    return {'routing (left list)': L_in + alpha - beta_x, 'routing (right list)': L_in + alpha - beta_y,
            'candidates': 2 * L_in - ell + alpha - beta_x - beta_y}


# ------------------------------------------------------------------------------------------------- constraints
def fraction_checks(prm):
    """Per (join, part): the smallest pair fraction (a domain constraint) and |sum of the fractions - 1| (a check)."""
    smallest, deviation = {}, {}
    for i, join in enumerate(prm['joins']):
        for name, tables in join.get('fractions', {}).items():
            smallest[i, name] = min(min(f.values()) for f in tables.values())
            deviation[i, name] = max(abs(sum(f.values()) - 1) for f in tables.values())
    return smallest, deviation


def constraints(inst, parts, prm, D, fixed, half, L, budget, residual=None):
    """Every constraint of the model, in a fixed order.  Each is `value >= 0`.

    group 'model': stated in the paper, with its location.
    group 'domain': a quantity is in its range: sizes, weights, pair fractions, type entries and list sizes are >= 0,
    and the weight of the syndrome part is at most its size.
    The problem adds its own model constraints (problem.constraints): what the parts leave for the remainder.

    Saturation holds for the levels 1 <= i < d, as in the paper, and for the top list with accounting
    'many-solutions'.  At level 0 it holds by construction (L_0 = D_0 - fixed_0), so it is not a constraint."""
    d, M = inst.depth, inst.memory_bound
    out = []

    def model(name, value, where):
        out.append(Constraint(name, value, 'model', ref(where)))

    def domain(name, value, boxed=False):
        out.append(Constraint(name, value, 'domain', '', boxed))
    # ---- model ----
    # saturation: a list has no more elements than there are vectors of its type with its fixed label bits
    top = d + 1 if inst.accounting == 'many-solutions' else d
    for i in range(1, top):
        where = 'many solutions' if i == d else 'saturation'
        model(f'saturation: L_{i} <= D_{i} - fixed_{i}', D[i] - fixed[i] - L[i], where)
    if inst.accounting == 'budget':
        model('representation budget >= 0', budget, 'budget')
    if M is not None:
        if inst.tree.dissection:
            model('7-dissection: D_0/7 <= M', M - D[0] / 7, 'dissection')
        else:
            model('memory: L_half <= M', M - half, 'tree cost')
        for i in range(d):
            model(f'memory: L_{i} <= M', M - L[i], 'tree cost')
    out += inst.problem.constraints(inst, parts)                        # what the parts leave for the remainder
    # ---- domain ----
    for p in parts:
        if not p.remainder:
            domain(f'size of {label(inst, p.name)} >= 0', p.size)
        if p.kind == 'syndrome':
            domain(f'weight of {label(inst, p.name)} <= its size', p.size - p.weight)
    digit = min(parts.tree.type, key=parts.tree.type.get)
    domain(f'type of the {label(inst, "tree")}: share of digit {digit} >= 0', parts.tree.type[digit])
    for (i, name), value in fraction_checks(prm)[0].items():
        domain(f'join {i} {label(inst, name)}: pair fractions >= 0', value, boxed=True)
    if inst.joint_types == 'maxent':
        for i in range(1, d + 1):
            if 'child' in prm['joins'][i]:
                for t in parts.trees:
                    if 'mean' not in prm['joins'][i]['child'][t.name]:      # the shares of 0 (and 1) are derived
                        domain(f'join {i} {label(inst, t.name)}: level type shares >= 0', min(t.types[i - 1].values()))
                # the scaling residual is 1e-13 (its tolerance) inside the domain and grows outside.  The constraint
                # is exactly 0 inside, so that the optimizer's finite-difference gradient does not pick up the noise
                excess = (residual[i] if residual else 0.0) - 1e-10
                domain(f'join {i}: joint type exists (scaling residual = 0)', -max(0.0, excess))
    for i in range(d):
        domain(f'L_{i} >= 0', L[i])
    return out


# ----------------------------------------------------------------------------------------------- the evaluation
def evaluate(inst, prm):
    """Time and memory exponent of the instance at the parameters `prm`, with every intermediate quantity."""
    ev, d = Evaluation(), inst.depth
    parts = partition(inst, prm, inst.problem.parts(inst, prm))                          # [partitioning]
    shares = contributions(inst, prm, parts)                                              # [lemma 1]
    sums = per_join(parts, shares)
    final = sums.final[d]                                                                 # -log2 Q of the syndrome part
    D, fixed, half, L = list_sizes(inst, parts, sums)                                     # [list sizes]
    ev.budget, delta = repetitions(inst, parts, sums)                                     # [budget], [repetitions]
    runs = inst.problem.runs(inst, parts)                                                 # decoding: [isd]

    memory_0 = half
    # the joins: [plain join], [filtered join]
    for i in range(d + 1):
        records = [dict(filter_exponents(p.sizes[i], sums.J[i][p.name], *filter_of(inst, p, i),
                                         prm['joins'][i].get('filter', {}).get(p.name, {})), part=p.name)
                   for p in parts if p.name in sums.J[i]]
        if i == 0 and 0 not in inst.syndromes:
            terms, memory_0 = base_merge_cost(inst, D, half, sums.ell[0])
        else:
            terms = join_cost(half if i == 0 else L[i - 1], sums.ell[i], records)
        if i < d:
            terms[f'output L_{i}'] = L[i]
        ev.terms.update({f'join {i}: {k}': v + delta[i] for k, v in terms.items()})
        ev.joins.append(dict(J=sums.J[i], r=sums.r[i], c=sums.c[i], c_label=sums.c_label[i], ell=sums.ell[i],
                             filters=records, alpha=sum(x['alpha'] for x in records),
                             beta_x=sum(x['beta_x'] for x in records), beta_y=sum(x['beta_y'] for x in records),
                             terms=terms, delta=delta[i]))
    ev.part_sizes = {t.name: t.size for t in parts.trees}
    whole = sum(ev.part_sizes.values())
    for t in parts.trees:                                                                 # per part, for the reader
        filtered = [[x for x in join['filters'] if x['part'] == t.name] for join in ev.joins]
        exponents = {key: [sum(x[key] for x in f) for f in filtered] for key in ('alpha', 'beta_x', 'beta_y')}
        ev.by_part[t.name] = dict(size=t.size / whole if whole > 0 else 0.0, types=[t.types[i] for i in range(d + 1)],
                                  J=[shares[i][t.name].J if t.name in shares[i] else None for i in range(d + 1)],
                                  D=[t.sizes[i] * H(t.types[i]) for i in range(d + 1)],
                                  r=[shares[i][t.name].r if t.name in shares[i] else 0.0 for i in range(d + 1)],
                                  c=[shares[i][t.name].c if t.name in shares[i] else 0.0 for i in range(d + 1)],
                                  **exponents)
    ev.unit, ev.digits = inst.unit, inst.digits
    ev.shift = (min(inst.shift), {a: inst.problem.add(a, 1) for a in inst.digits}) if inst.shift else None
    if inst.rate is not None:
        ev.kappa, ev.omega_1 = parts.tree.size, parts.tree.weight
    ev.levels = [dict(D=D[i], L=L[i], ell=sums.ell[i], delta=delta[i], type=parts.trees[0].types[i],
                      types={t.name: t.types[i] for t in parts.trees}) for i in range(d + 1)]
    ev.records = constraints(inst, parts, prm, D, fixed, half, L, ev.budget, sums.residual)
    ev.checks = {f'join {i} {label(inst, name)}: |sum of pair fractions - 1|': value
                 for (i, name), value in fraction_checks(prm)[1].items()}
    ev.final_check = final

    if inst.accounting == 'many-solutions':             # [many solutions]: runs until one of the N solutions appears
        runs += max(0.0, -L[d] + final + min(0.0, inst.log_solutions))
    ev.runs, ev.memory, ev.memory_0 = runs, max([memory_0] + L[:d]), memory_0
    ev.time = max(ev.terms.values()) + runs                                               # [tree cost]
    return ev
