"""Syndrome decoding over F_q in the Hamming metric: information set decoding with a tree.  Standard library only."""
import math
from dataclasses import replace

from ..build import class_share, class_size, expand, reps
from ..entropy import H, gv
from ..families import Variable
from ..model import Constraint
from ..parts import ExactPart, Parts, SyndromePart, TreePart
from ..references import ref
from .base import Problem


def is_prime(q):
    return q >= 2 and all(q % p for p in range(2, int(q ** 0.5) + 1))


def is_prime_power(q):
    p = next(p for p in range(2, q + 1) if q % p == 0)
    while q % p == 0:
        q //= p
    return q == 1


class Decoding(Problem):
    """Syndrome decoding of a random code of rate R over F_q: find e in F_q^n of weight W n with H e = s.

    weight   W: a number in (0, 1], 'gv' (the q-ary Gilbert-Varshamov weight of the rate, H_q^{-1}(1 - R)) or
             'gv/2' (half of it).  The non-zero entries of e are uniform over F_q^*.  `weight` keeps the argument as
             given, so that 'gv' follows the rate in replace(rate=...), and `omega` is ω, the weight as a number.

    The labels are the (1 - R) n syndrome coordinates, u = log2 q bits each.  The tree runs on a window: the R n
    information coordinates plus the coordinates its exact joins match (information set decoding, rule 2).  The
    syndrome coordinates that the window leaves (the syndrome part, `rest` in the model) are checked by the top join:
    by distance if it has a syndrome filter, and by the final weight check.  The error must have the assumed weight
    on the window, so the runs of the whole tree are the permutations of the coordinates, unless the tree is built
    for many solutions (Tree(accounting='many-solutions'), rule 5).

    Problems with an error of another type (typed.py: restricted errors, Lee metric) override __init__, `check`,
    `digit_rules` and, for another description of the window, `window_type` and `window_variables`.
    """
    arguments = ('q', 'rate', 'weight')
    integers = False

    def __init__(self, q, rate, weight='gv'):
        if not (isinstance(q, int) and q >= 2 and is_prime_power(q)):
            raise ValueError(f'q = {q} is not a prime power: there is no field with {q} elements')
        if not 0 < rate < 1:
            raise ValueError('rate: between 0 and 1')
        if not (weight in ('gv', 'gv/2') or (isinstance(weight, (int, float)) and 0 < weight <= 1)):
            raise ValueError(f"weight: a number in (0, 1], 'gv' or 'gv/2' (got {weight!r})")
        self.q, self.rate, self.weight, self.unit = q, rate, weight, math.log2(q)
        if weight == 'gv':
            W = gv(rate, q)
        elif weight == 'gv/2':
            W = gv(rate, q) / 2
        else:
            W = weight
        self.omega = W                                                  # ω, the weight as a number
        self.solution = {a: 1.0 - W if a == 0 else W / (q - 1) for a in range(q)}    # the type of the error e
        self.log_solutions = H(self.solution) - (1 - rate) * math.log2(q)            # rule 5

    def add(self, a, b):
        return (a + b) % self.q

    def default_digits(self):
        """The digits of the error and the sums of two of them (for Decoding: all of F_q)."""
        support = [a for a, share in self.solution.items() if share > 0]
        return tuple(sorted({0} | set(support) | {self.add(a, b) for a in support for b in support}))

    def check(self, tree, digits):
        q = self.q
        if digits != tuple(range(q)):
            raise ValueError(f'Tree.digits: Decoding over F_{q} uses all digits 0, ..., {q - 1} in the lists (got '
                             f'{len(digits)} digits)')
        free = tree.nonzero == 'free'
        if free and not is_prime(q):
            raise ValueError(f'q = {q} is not a prime: mod({q}) is not the addition of the field.  With all non-zero '
                             'digits alike only the numbers of digit pairs matter, which are the same.  '
                             "nonzero='free' needs a prime q")
        if self.omega > (q - 1) / q and q >= 3 and not free:
            raise ValueError(f"weight {self.omega} > (q-1)/q over F_{q}: large weight needs Tree(nonzero='free') for "
                             'q >= 3')

    def rules(self, inst):
        """Rules 2 to 6 of RULES.md: window, digits, large weight, number of solutions, syndrome filter."""
        q, R = self.q, self.rate
        lines = []
        inst.weight, inst.root_type = self.omega, {a: float(self.solution.get(a, 0.0)) for a in inst.digits}
        lines.append(f'rule 2 (window): {1 - R:.4f} n syndrome coordinates of u = log2 {q} = {math.log2(q):.4f} bits '
                     f'each.  The tree runs on the {R:.4f} n information coordinates plus the exactly matched ones'
                     + ('' if inst.accounting == 'many-solutions' else ', runs of the tree = permutations'))
        if len(inst.digits) < q:                                # the lists use a subset of Z_q, the labels all of it
            inst.field_class_of = {a: min(a, q - a) for a in range(q)}
            if inst.syndromes:
                lines.append(f'the lists use {len(inst.digits)} of the {q} digits.  A syndrome filter sees labels over '
                             f'all of Z_{q}: one price per class {{a, -a}} of Z_q')
        lines += self.digit_rules(inst)
        lines.append(f'rule 5 (solutions): error weight ω = {self.omega:.6f}, expected number of solutions '
                     f'2^({self.log_solutions:+.6f} n)')
        for j in sorted(inst.syndromes):
            lines.append('rule 6 (syndrome filter): the top join matches the rest of the syndrome by distance with '
                         f'the filter {inst.syndromes[j].describe()}'
                         + (', and it also matches label units exactly (mixed join)' if inst.tree.mixed else '')
                         + '.  Near or far follows from the weight (see the filter column of the report)')
        return lines

    def digit_rules(self, inst):
        """Rule 3: the non-zero digits are alike unless Tree(nonzero='free').  Rule 4: above the weight (q-1)/q the
        tree works on e - (1,...,1), so that 0 is the most frequent digit."""
        q = self.q
        inst.alike = inst.tree.nonzero != 'free'
        if inst.tree.nonzero is None:                               # the default, written out
            inst.tree = replace(inst.tree, nonzero='alike')
        if inst.alike:
            line = 'rule 3 (digits): all non-zero digits alike'
        else:
            line = 'rule 3 (digits): full joint types, free window type, free prices'
        if not is_prime(q):
            line += (f'.  q = {q} is not a prime: mod({q}) stands for the addition of F_{q} (same numbers of digit '
                     'pairs)')
        lines = [line]
        top = max(inst.tree.representation_joins, default=None)
        if self.omega > (q - 1) / q and top is not None:
            inst.shift = {top: 1}
            lines.append(f'rule 4 (large weight): ω = {self.omega:.4f} > (q-1)/q = {(q - 1) / q:.4f}: the tree works '
                         'on e - (1,...,1)')
        return lines

    def variables(self, inst, template):
        """The window, the exact units ell_0..ell_{d-1}, and ell_d if the top join matches exactly (an exact or mixed
        top join)."""
        d, R = inst.depth, self.rate
        slots = self.window_variables(inst, template)
        slots += [Variable(('ell', j), 0.0, 1 - R) for j in range(d)]
        if inst.tree.joins[d].exact:
            template['ell_d'] = 0.0
            slots.append(Variable(('ell_d',), 0.0, 1 - R))
        return slots

    def paper_name(self, inst, path):
        """ℓ^(i) in bits (ℓ^(d) of an exact or mixed top join: ell_d), ω₁ (window.weight), and the type p^(d)
        of e₁ on the window (window.type).  The type is given per digit while the model holds class totals, and with
        rule 4 it is the type of e₁ - (1,...,1)."""
        d = inst.depth
        if path == ('ell_d',):
            return ('ell_d',), inst.unit
        if path == ('window', 'weight'):
            return ('omega_1',), 1.0
        if path[:2] == ('window', 'type'):
            a = int(path[2])
            if inst.shift:                                  # the paper's digit of e₁ - (1,...,1)
                a = next(b for b in inst.digits if self.add(b, 1) == a)
                return ('p', d, a), 1.0 / class_size(inst, self.add(a, 1))
            return ('p', d, a), 1.0 / class_size(inst, a)
        return super().paper_name(inst, path)

    def window_variables(self, inst, template):
        """The error on the window: its weight (digits alike), or one share per non-zero digit (class)."""
        if inst.alike:
            template['window'] = {'weight': self.omega / 2}
            return [Variable(('window', 'weight'), 0.0, self.omega)]
        support = [a for a in reps(inst) if a != 0 and inst.root_type[a] > 0]
        template['window'] = {'type': {str(a): 0.5 * class_share(inst, inst.root_type, a) for a in support}}
        return [Variable(('window', 'type', str(a)), 0.0, 1.0) for a in support]

    def window_type(self, inst, prm, size):
        """(type of the error on the window of `size` n coordinates, weight of the error inside it)."""
        if inst.alike:                                              # the weight spread over the non-zero digits
            inside, q = prm['window']['weight'], len(inst.digits)
            return {a: 1 - inside / size if a == 0 else inside / size / (q - 1) for a in inst.digits}, inside
        shares = expand(inst, {int(a): s for a, s in prm['window']['type'].items()})     # one share per non-zero digit
        window = {a: 1 - sum(shares.values()) if a == 0 else shares.get(a, 0.0) for a in inst.digits}
        return window, size * sum(shares.values())

    def parts(self, inst, prm):
        """The window is the tree part (rule 2).  The exact units are ell_0..ell_{d-1}, and ell_d if the top join
        matches exactly.  `rest` is the syndrome part, which takes the coordinates and the weight that the window
        leaves."""
        d = inst.depth
        ell = list(prm.get('ell', [0.0] * d)) + [prm.get('ell_d', 0.0)]
        size = self.rate + sum(ell)                                    # rule 2: k + exactly matched coordinates
        root, inside = self.window_type(inst, prm, size)
        tree = TreePart('tree', size, root, 0, inside)
        exact = [ExactPart(f'ell_{j}', x, {0: 1.0}, j) for j, x in enumerate(ell) if j < d or inst.tree.joins[d].exact]
        # `rest`: with one solution the tree must find the planted error, whose type on `rest` is the complement of
        # the window type, digit by digit.  With many solutions any vector of the right weight completes a solution,
        # and the dominant type of that weight has the non-zero digits in the proportions of the error type (equally
        # likely for Decoding, only the digits of E for restricted errors, the Lee proportions for Lee).
        rest_size = 1.0 - size
        rest_weight = inst.weight - inside
        if inst.accounting == 'many-solutions':
            gamma = min(1.0, max(0.0, rest_weight / rest_size)) if rest_size > 1e-12 else 0.0
            nonzero = sum(v for a, v in inst.root_type.items() if a != 0)
            rest_type = {a: 1 - gamma if a == 0 else gamma * inst.root_type[a] / nonzero for a in inst.digits}
        else:
            rest_type = {}
            for a in inst.digits:
                if a != 0:
                    left = inst.root_type[a] - size * tree.type[a]
                    rest_type[a] = left / rest_size if rest_size > 1e-12 else 0.0
            rest_type[0] = 1.0 - sum(rest_type.values())
        return Parts(tree, exact, SyndromePart('rest', rest_size, rest_type, d, rest_weight, remainder=True))

    def constraints(self, inst, parts):
        """[isd]: the window leaves syndrome coordinates and error weight for `rest`, and with a free window type every
        digit of the error is left over."""
        rest, isd = parts.rest, ref('isd')
        out = [Constraint('syndrome coordinates: exact <= (1 - R) n', rest.size, 'model', isd),
               Constraint('error weight: window <= ω', rest.weight, 'model', isd)]
        if not inst.alike:
            for a, share in rest.type.items():
                if a != 0 and inst.root_type[a] > 0:
                    name = f'error type: share of digit {a} on the syndrome part >= 0'
                    out.append(Constraint(name, rest.size * share, 'model', isd))
        return out

    def runs(self, inst, parts):
        """[isd].  The error must have the assumed type on the window: a window of size s and type t out of the n
        coordinates of the error type t' costs  H(t') - s H(t) - (1 - s) H(t'')  permutations, t'' the type of the
        coordinates outside the window.  (For a weight spread equally over the non-zero digits this is the binomial
        count binomial(n, w) - binomial(s, w_1) - binomial(n - s, w - w_1) of the paper.)  With the accounting
        'many-solutions' the runs follow from the top list instead (model.evaluate)."""
        if inst.accounting == 'many-solutions':
            return 0.0
        window, error = parts.tree, inst.root_type
        rest = 1.0 - window.size
        if rest <= 1e-12:                                               # the window takes all coordinates
            return 0.0
        left = {a: (error[a] - window.size * window.type.get(a, 0.0)) / rest for a in error}
        return H(error) - window.size * H(window.type) - rest * H(left)
