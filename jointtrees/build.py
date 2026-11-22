"""From (problem, Tree) to the instance the model evaluates.  Standard library only.

This file applies the rules of RULES.md that belong to the Tree.  The rules of a problem belong to its class
(problems/).  No formula of the paper is here (those are in model.py, filters.py and families.py).

    check(problem, tree)                  rejects input the model has no meaning for, with a message that says why
    build(problem, tree, memory_bound)    resolves the rules: the joins, the filters, the accounting, and the
                                          problem's own rules (window, digit symmetry, large weight, solutions)
    variables(instance)                   the free parameters and their bounds, for the optimizer.  Its docstring
                                          is the schema of the model's parameter dict

The parts of the solution and of its label (parts.py) come from the problem: problem.parts(instance, parameters).
"""
from dataclasses import dataclass, field
from typing import NamedTuple

from .families import FilterContext, Variable
from .jointtypes import classes
from .lee import lee_weight
from .references import ref

UNLIMITED_LABELS = 1.0  # the search box of each ell_j when the joins below the top have no bound (LWE)


@dataclass
class Instance:
    problem: object                 # the problem (problems/)
    tree: object                    # the Tree, with the digits of the lists filled in
    memory_bound: object            # the memory bound M, or None
    digits: tuple                   # the digits of the lists
    depth: int
    filters: dict                   # {join: filter family} on the tree part at representation joins
    syndromes: dict                 # {d: filter family} on the syndrome part, if the top join has a syndrome filter
    accounting: str                 # 'budget', 'repetitions' or 'many-solutions'
    unit: float = 1.0               # bits per label unit (problem.unit): log2 q for syndrome coordinates, else 1
    q: int = None                   # size of Z_q of a q-ary problem (the digits may be a subset of it)
    log_solutions: float = 0.0      # log2 of the expected number of solutions, per n (rule 5, problem.log_solutions)
    joint_types: str = 'free'       # 'free' pair fractions, or 'maxent' child types (Tree.joint_types)
    # set by the problem's rules (problem.rules):
    alike: bool = False             # all non-zero digits alike (rule 3)
    shift: dict = field(default_factory=dict)   # {join: 1}: parent = x + y + (1,...,1) at this join (rule 4)
    weight: object = None           # decoding: W, the share of non-zero digits of the error
    root_type: dict = None          # decoding: the type of the error over the digits of the lists, {digit: share}
    class_of: dict = None           # digit -> representative of its class (symmetry), None without symmetry
    lee_cutoff: float = None        # Boltzmann types on a subset of Z_q: the largest Lee weight among the digits
    # the lists use a subset of Z_q: element of Z_q -> min(a, q - a).  A syndrome filter then has one price per class
    # {a, -a} of Z_q, since the labels use all of Z_q
    field_class_of: dict = None
    rules: list = field(default_factory=list)
    cache: dict = field(default_factory=dict)

    @property
    def rate(self):
        return self.problem.rate

    @property
    def tree_parts(self):
        """The names of the tree parts: 'tree', or 'tree_1', ..., 'tree_P' with partitioning."""
        P = self.tree.parts
        if P == 1:
            return ['tree']
        return [f'tree_{k}' for k in range(1, P + 1)]


class Remainder(NamedTuple):
    """A group of pair fractions in the model's dict (one join, one part, one parent digit) and the class of pairs
    whose fraction is the remainder 1 - (the others)."""
    path: tuple                     # ('joins', join, 'fractions', part, parent)
    join: int
    part: str
    parent: str
    first: str


class Variables(NamedTuple):
    """The free parameters of an instance (see `variables`)."""
    template: dict                  # the model's parameter dict with start values
    slots: list                     # [Variable(path in the dict, low, high, search)]
    remainders: list                # [Remainder]
    owners: dict                    # {path of a filter's dict: (filter family, its context)}


def check(problem, tree):
    """Reject input the model has no meaning for.  Every message says what is expected (the Tree and its joins check
    themselves on construction).  Returns the digits of the lists: Tree.digits, sorted, or the problem's default."""
    digits = problem.default_digits() if tree.digits is None else tuple(sorted(tree.digits))
    if len(set(digits)) != len(digits) or len(digits) < 2 or not all(isinstance(a, int) for a in digits):
        raise ValueError('digits: at least two distinct integers')
    if tree.types == 'boltzmann' and (tree.joint_types != 'maxent' or not hasattr(problem, 'boltzmann')):
        raise ValueError("types='boltzmann' needs joint_types='maxent' and LeeDecoding (Boltzmann types of the Lee "
                         'metric)')
    problem.check(tree, digits)
    return digits


def build(problem, tree, memory_bound=None):
    """The instance of (problem, tree, memory_bound), with the list of rules that were applied."""
    bound = memory_bound
    if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float)) or bound < 0):
        raise ValueError(f'memory_bound: None or a number >= 0, the memory 2^(M n) (got {bound!r})')
    digits = check(problem, tree)
    tree = tree.replace(digits=digits)
    rules = ['rule 1 (base merge): join 0 merges two halves into a base list'
             + (' by the 7-dissection' if tree.dissection else ''),
             'joins: ' + ', '.join(f'{i} {join.kind}' for i, join in enumerate(tree.joins))]
    filters = {i: tree.joins[i].filter for i in tree.representation_joins if tree.joins[i].filter is not None}
    syndromes = {i: join.syndrome for i, join in enumerate(tree.joins) if join.syndrome is not None}
    inst = Instance(problem, tree, bound, digits, tree.depth, filters, syndromes, tree.accounting, unit=problem.unit,
                    q=problem.q, log_solutions=problem.log_solutions, joint_types=tree.joint_types, rules=rules)
    if tree.joint_types == 'maxent':
        rules.append('joint types: the free numbers are the types of the levels, and each join uses the joint type of '
                     'maximum entropy with those types (the typical joint type of the standard join)')
    if tree.parts > 1:
        rules.append(f'partitioning: the positions of the tree are cut into {tree.parts} parts, fixed at random and '
                     'the same for every list.  Each part has the top-level type of the tree part and its own joint '
                     f"types and filter parameters, and the label units are shared ({ref('partitioning')})")
    rules += problem.rules(inst)
    for flt in filters.values():                        # each family rejects places where it has no meaning
        flt.check(context(inst, False))
    for flt in syndromes.values():
        flt.check(context(inst, True))
    if inst.alike and any(flt.name != 'prices' for flt in [*filters.values(), *syndromes.values()]):
        rules.append('rule 3: with all non-zero digits alike every filter family reduces to one price on the digit 0')
    rules.append(accounting_rule(tree.accounting))
    return inst


def accounting_rule(accounting):
    """The line of the applied rules that says how missing representations are paid."""
    if accounting == 'budget':
        return f"accounting: one representation of the solution must survive ({ref('budget')})"
    if accounting == 'repetitions':
        return f"accounting: subtrees are repeated when representations are missing ({ref('repetitions')})"
    return ('accounting: many solutions, the runs follow from the size of the top list, which a deficit of '
            f"representations lowers ({ref('many solutions')})")


# ---------------------------------------------------------------------------------------------- shared helpers
def expand(inst, by_class):
    """Shares given per class representative -> shares per digit (a class of two digits splits its share equally)."""
    if inst.class_of is None:
        return dict(by_class)
    members = {}
    for a, rep in inst.class_of.items():
        members.setdefault(rep, []).append(a)
    out = {}
    for rep, share in by_class.items():
        for a in members[rep]:
            out[a] = share / len(members[rep])
    return out


def context(inst, syndrome):
    """The FilterContext of the tree part (syndrome=False) or of the syndrome part (syndrome=True), cached."""
    key = ('context', syndrome)
    if key not in inst.cache:
        classes_of = (inst.field_class_of or inst.class_of) if syndrome else None   # window filter: price per digit
        digits = tuple(range(inst.q)) if syndrome and inst.field_class_of else inst.digits
        rep = {a: (classes_of[a] if classes_of else a) for a in digits}
        q = None if inst.problem.integers else inst.q
        inst.cache[key] = FilterContext(digits, rep, 0 if 0 in digits else min(digits), inst.alike, q, syndrome)
    return inst.cache[key]


def filter_of(inst, part, join):
    """The one rule of which part is filtered at which join: (filter family, its context), or (None, None).  A tree part
    at a representation join carries the join's `filter`, and the syndrome part the `syndrome` filter of the top join.
    `variables` creates the parameters of exactly these filters."""
    if part.kind == 'tree' and join in inst.filters:
        return inst.filters[join], context(inst, False)
    if part.kind == 'syndrome' and join in inst.syndromes:
        return inst.syndromes[join], context(inst, True)
    return None, None


# --------------------------------------------------------------------------------------- variables (optimizer)
def set_path(target, path, value):
    """target[path[0]][path[1]]... = value, creating the dicts on the way."""
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def variables(inst):
    """The free parameters of an instance as Variables: the model's parameter dict with start values (`template`),
    its free entries as Variable(path, low, high, search) (`slots`), for every group of pair fractions the entry
    that takes the remainder (`remainders`), and for every filter its place in the dict with (family, context)
    (`owners`), so that the optimizer asks the family for start values.

    The parameters of a tree in the model are one dict, and `template` is an instance of it:

        ell        [ell_0, ..., ell_{d-1}]   label units matched exactly at joins 0..d-1 (units, not bits)
        ell_d      ell_d of an exact or mixed top join of a decoding problem (otherwise ell_d follows from the problem)
        window     decoding: {'weight': P} (weight of the error on the window, fraction of n) with digits alike, or
                   {'type': {digit: share}} (shares of the window positions) with digits free
        parts      with partitioning: {'tree_k': {'size': eps_k}} for k = 1..P-1 (the last part takes the rest)
        joins      one dict per join 0..d:
            fractions   {part: {parent key: {pair class: fraction}}} for every tree part split at this
                        representation join (jointtypes.pair_classes)
            child       joint_types='maxent': {part: {digit: share}} the child type of the tree part(s)
            filter      {part: parameters of its filter family}: a tree part ('tree', or 'tree_k') at a filtered
                        join, and the syndrome part ('rest') at a top join with a syndrome filter.  The keys are the
                        family's (families.py): 'z' {digit: price}, 'z_y', 'sigma', and 'u' (the size of a free
                        center)

    `search` names the sides ('low', 'high') of a bound that only limit the search and are not implied by a
    constraint of the model.  The optimizer warns when its result lies on such a side.  All other bounds are implied
    by the constraints of model.constraints (sizes, weights and fractions between 0 and their maximum)."""
    d, trees = inst.depth, inst.tree_parts
    template = {'ell': [0.0] * d, 'joins': [dict() for _ in range(d + 1)]}
    slots, remainders, owners = list(inst.problem.variables(inst, template)), [], {}    # the problem's own first
    if len(trees) > 1:                                              # partitioning: the sizes of all parts but the last
        template['parts'] = {name: {'size': 1.0 / len(trees)} for name in trees[:-1]}
        slots += [Variable(('parts', name, 'size'), 0.0, 1.0) for name in trees[:-1]]
    representation_joins = inst.tree.representation_joins
    for i in range(d + 1):
        join = template['joins'][i]
        representation = i in representation_joins
        if inst.joint_types == 'maxent' and representation:
            levels_down = sum(1 for j in range(i, d + 1) if j in representation_joins)
            for name in trees:
                slots += child_type_start(inst, join.setdefault('child', {}), i, name, levels_down)
        if inst.joint_types == 'free' and representation:          # the joint types by their pair fractions
            table = classes(inst, inst.shift.get(i, 0))
            for name in trees:
                for parent, names in table['fractions'].items():
                    fractions = {x: float(n == 0) for n, x in enumerate(names)}
                    join.setdefault('fractions', {}).setdefault(name, {})[parent] = fractions
                    path = ('joins', i, 'fractions', name, parent)
                    slots += [Variable(path + (x,), 0.0, 1.0) for x in names[1:]]
                    remainders.append(Remainder(path, i, name, parent, names[0]))
        filtered = [(name, inst.filters[i], context(inst, False)) for name in trees if i in inst.filters]
        if i in inst.syndromes:
            filtered.append(('rest', inst.syndromes[i], context(inst, True)))
        for name, flt, ctx in filtered:                             # the parts filtered here: see filter_of
            base = ('joins', i, 'filter', name)
            f = join.setdefault('filter', {}).setdefault(name, {})
            for v in flt.declared(ctx):
                set_path(f, v.path, 0.5 if v.path == ('u',) else 0.0)
                slots.append(Variable(base + v.path, v.low, v.high, v.search))
            owners[base] = (flt, ctx)
    return Variables(template, slots, remainders, owners)


def child_type_start(inst, children, i, name, levels_down):
    """joint_types='maxent': the start values of the child type of tree part `name` at join i (into `children`) and
    its slots.  A Boltzmann type has one parameter, its mean Lee weight.  Otherwise the type has one share per non-zero
    digit (class), and 0 takes the remainder.  With integer addition the mean of a type is conserved (mean(parent) =
    2 mean(child)), so the share of one more digit is determined too (balance_digit, parts.TreePart.share).  Over Z_q
    no mean is conserved, and the shares start near the type of the error."""
    if inst.tree.types == 'boltzmann':
        mean = sum(share * lee_weight(a, inst.q) for a, share in inst.root_type.items())
        children[name] = {'mean': 0.8 * mean * 0.7 ** levels_down}
        return [Variable(('joins', i, 'child', name, 'mean'), 0.0, inst.q // 2)]
    if inst.problem.integers:
        start = {str(a): 0.01 for a in reps(inst) if a not in (0, balance_digit(inst))}
    else:
        start = {str(a): class_share(inst, inst.root_type, a) * 0.6 ** levels_down for a in reps(inst) if a != 0}
    children[name] = start
    return [Variable(('joins', i, 'child', name, a), 0.0, 1.0) for a in start]


def model_variables(inst):
    """`variables(inst)`, computed once per instance (the caller must not change what it returns)."""
    if 'variables' not in inst.cache:
        inst.cache['variables'] = variables(inst)
    return inst.cache['variables']


def balance_digit(inst):
    """With integer addition, the digit whose share balances the mean of a child type: the smallest positive digit
    (1 if it is a digit), or the largest negative one if there is no positive digit."""
    positive = [a for a in inst.digits if a > 0]
    return min(positive) if positive else max(a for a in inst.digits if a < 0)


def child_type_of(inst, parent, given):
    """The child type of a maxent join from the free shares `given` (per digit): 0 takes the remainder, and with
    integer addition the digit b = balance_digit(inst) balances the mean, b child(b) = mean(parent)/2 - sum_{a != 0, b}
    a child(a)."""
    child = {a: given.get(a, 0.0) for a in inst.digits}
    if inst.problem.integers:
        b = balance_digit(inst)
        mean = sum(s * share for s, share in parent.items())
        child[b] = (mean / 2 - sum(a * share for a, share in child.items() if a not in (0, b))) / b
    child[0] = 1.0 - sum(share for a, share in child.items() if a != 0)
    return child


def reps(inst):
    """The digits that carry a variable: every digit, or one representative per class with a symmetry."""
    if inst.class_of is None:
        return list(inst.digits)
    return sorted(set(inst.class_of.values()))


def class_share(inst, type_, rep):
    """The share of a whole class in a type (the sum over its digits)."""
    if inst.class_of is None:
        return type_.get(rep, 0.0)
    return sum(v for a, v in type_.items() if inst.class_of[a] == rep)


def class_size(inst, a):
    """The number of digits that share the variable of digit a (1 without a digit symmetry)."""
    if inst.class_of is None:
        return 1
    rep = inst.class_of[a]
    return sum(1 for b in inst.class_of if inst.class_of[b] == rep)
