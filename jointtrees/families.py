"""Filter families: how the parameters of a filter give the prices of the digits.  Standard library only.

A filter keeps a vector if it fits a random center (filters.py has the closed form of its exponents alpha, beta_x,
beta_y, the same for every family).  A family only says which parameters a filter has and how they give one price
per digit for the left list (z) and for the right list (z_y), in the paper's words [simplifications]:

    tilt 'opposite'   z_y = -z        the two vectors add up to the parent (representation joins)
    tilt 'same'       z_y = z         the two vectors are close (a near-neighbour syndrome filter)
    tilt 'own'        z_y free        the general filter: own prices for each list

The size u of the centers is a parameter of every filter: 'free' (optimized, the default) or a number fixed by the
user.  The base class handles u and the case "all non-zero digits alike" (rule 3 of RULES.md), and a family writes
only `variables`, `prices` and `random_start`.  docs/GUIDE.md ("Extending the library") shows a family, and
tests/test_extension.py a complete example.  The names of a family's variables are the names of its parameters
everywhere: in `result.parameters` (e.g. P.filter[3].z[-1], P.filter[3].u), in `fixed=` and in witness files.

The contract in detail:

  * `Variable(path, low, high, search)`: `path` is the place of the value inside the filter's dict, e.g. ('slope',)
    or ('z', '3') (a digit as a string, as in a witness file).  `search` names the sides of the range that only limit
    the search (the optimizer warns when a result lies there).  A side implied by the model or by a symmetry is not
    named.
  * `random_start(ctx, rng, shared)`: `shared` is one dict per start of the optimizer, the same for all filters of the
    tree.  A draw stored there (shared.setdefault(name, draw)) couples the filters of different joins.
  * With all non-zero digits alike (Decoding, rule 3) the base class replaces `variables` and `prices` by one price
    z[0] on the digit 0 for every family (the build prints a rule line when that happens to a family other than
    PriceFilter).
  * The family must be imported before a witness that names it is read: `python3 -m jointtrees verify --import
    my_filters.py FILE.json`, or define it in the package.
  * Optional methods: `check(ctx)` rejects places where the family has no meaning, and `default` gives start values.
    A start from another family (starts.embed) takes that family's prices.
"""
from dataclasses import dataclass, fields
from typing import NamedTuple

from .lee import lee_weight

RESERVED = frozenset(name for name in dir(dict) if not name.startswith('_'))    # dict methods: not parameter names

PRICE_MAX = 40.0        # a price of 40 bits makes the fraction theta equal to 0 or 1 up to 1e-12
SIGMA_MAX = 10.0        # sigma of LeeWeightFilter, in bits per unit of Lee weight: the search box
CENTER_MIN = 0.001      # size u of a free center: CENTER_MIN .. 1 - CENTER_MIN (u = 0 or 1 is no filter)


class Variable(NamedTuple):
    """One free parameter: its path (inside a filter's parameter dict, e.g. ('z', '3'), or in the model's dict), its
    range, and which sides of the range are only a search box ('low', 'high'), not implied by the model.  The
    optimizer warns when a result lies on such a side."""
    path: tuple
    low: float
    high: float
    search: tuple = ()


@dataclass
class FilterContext:
    """What a family may use about the part it filters (build.py creates it, the same for every family).

    digits     the digits of the filtered part's joint type (all of Z_q on the syndrome part)
    rep        digit -> the representative of its class.  Digits of one class share a price: the classes {a, -a} of
               the Lee metric, or of Z_q on the syndrome part when the lists use a subset of Z_q.  Without classes
               every digit represents itself
    reference  the digit whose price is fixed to 0 (only differences of prices matter): 0 if it is a digit
    alike      all non-zero digits alike (Decoding, rule 3): the base class then gives the one price of digit 0
    q          size of Z_q for a q-ary problem, else None (subset sum, LWE: the digits add as integers)
    syndrome   True for a syndrome filter (on the syndrome part of the top join), False on the tree part
    """
    digits: tuple
    rep: dict
    reference: int
    alike: bool = False
    q: int = None
    syndrome: bool = False

    @property
    def reps(self):
        """One representative per class, sorted."""
        return sorted(set(self.rep.values()))

    def value(self, j):
        """The digit j as a number: j itself, or over Z_q its signed centered value (j for j <= q/2, else j - q)."""
        return j if self.q is None or j <= self.q // 2 else j - self.q


class Filter:
    """The base of every filter family.  Subclasses register under `name` (witness files name them)."""
    name = None
    registry = {}
    tilt = 'opposite'
    u = 'free'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            Filter.registry[cls.name] = cls

    def __post_init__(self):
        """Every family: u is 'free' or a number in (0, 1), tilt is 'opposite', 'same' or 'own'."""
        check_u(self.u)
        if self.tilt not in ('opposite', 'same', 'own'):
            raise ValueError(f"{type(self).__name__}: tilt 'opposite', 'same' or 'own' (got {self.tilt!r})")

    # ---- what a family writes
    def variables(self, ctx):
        raise NotImplementedError

    def prices(self, ctx, p):
        raise NotImplementedError

    def random_start(self, ctx, rng, shared):
        return {}

    def check(self, ctx):
        """Reject a place where this family has no meaning (a ValueError that says why).  build.py calls it."""

    # ---- what the library calls (the base class handles digits alike and the center)
    def price_keys(self):
        """The price entries of this filter: z, and z_y when the second list has its own prices."""
        if self.tilt == 'own':
            return ('z', 'z_y')
        return ('z',)

    def declared(self, ctx):
        """All variables of this filter, with the center size and the rule for digits alike."""
        if ctx.alike:                               # rule 3: only the digit 0 has a price ("zero against the rest")
            if self.tilt == 'opposite':             # z[0] >= 0: swapping the two lists negates it
                out = [Variable(('z', '0'), 0.0, PRICE_MAX, ('high',))]
            else:
                out = [Variable((key, '0'), -PRICE_MAX, PRICE_MAX, ('low', 'high')) for key in self.price_keys()]
        else:
            out = list(self.variables(ctx))
            clash = sorted({v.path[0] for v in out} & RESERVED)
            if clash:
                raise ValueError(f'{type(self).__name__}: the parameter names {clash} are names of dict methods, '
                                 'choose others')
        if self.u == 'free':
            out.append(Variable(('u',), CENTER_MIN, 1 - CENTER_MIN, ('low', 'high')))
        return out

    def price_pair(self, ctx, p):
        """(z, z_y) on ctx.digits."""
        if not ctx.alike:
            return self.prices(ctx, p)
        z = {j: float(p['z'].get('0', 0.0)) if j == 0 else 0.0 for j in ctx.digits}
        if self.tilt == 'opposite':
            return z, {j: -z[j] for j in z}
        if self.tilt == 'same':
            return z, dict(z)
        return z, {j: float(p['z_y'].get('0', 0.0)) if j == 0 else 0.0 for j in ctx.digits}

    def start(self, ctx, rng, shared):
        """Start values of all declared variables: {path: value}."""
        if not ctx.alike:
            out = dict(self.random_start(ctx, rng, shared))
        elif self.tilt == 'opposite':
            out = {('z', '0'): rng.uniform(.1, 1.5)}
        else:
            zero = shared.setdefault('zero_price', rng.uniform(.5, 8.)) if ctx.syndrome else rng.uniform(.1, 1.5)
            out = {(key, '0'): zero for key in self.price_keys()}
        if self.u == 'free':
            out[('u',)] = rng.uniform(0.2, 0.8)
        return out

    def default(self, path):
        """The start value of a variable when nothing else is known."""
        if path == ('u',):
            return 0.5
        if path == ('z', '0') and self.tilt == 'opposite':
            return 0.8
        return 0.0

    def exponents(self, J, parameters, q=None, alike=False, size=1.0):
        """[filter exponents] of this filter on a part of `size` n positions whose pairs of digits have the joint type
        J = {(i, j): share}: the center size u, the center count alpha, the list reductions beta_x and beta_y, and the
        fractions theta_ij of [closed form].  `parameters` are the filter's parameters as a witness writes them, e.g.
        {'u': 0.3, 'z': {'1': 0.8}} (prices keyed by the digit as a string, the digit 0 has price 0).  q: the size of
        Z_q for digits of Z_q (None for integer digits).  alike: all non-zero digits alike (rule 3), only the digit 0
        has a price.

        >>> J = {(0, 0): .4, (0, 1): .1, (1, 0): .1, (1, 1): .4}
        >>> e = PriceFilter(u=.5).exponents(J, {'z': {'1': 1.}})
        >>> round(e['u'], 12), round(e['alpha'], 6), round(e['beta_x'], 6)
        (0.5, 0.016341, 0.003208)
        """
        from .filters import filter_exponents
        digits = tuple(sorted({i for i, _ in J} | {j for _, j in J}))
        context = FilterContext(digits, {a: a for a in digits}, 0 if 0 in digits else digits[0], alike=alike, q=q)
        record = filter_exponents(size, J, self, context, parameters)
        return {key: record[key] for key in ('u', 'alpha', 'beta_x', 'beta_y', 'theta')}

    def to_dict(self):
        return {'family': self.name, **{f.name: getattr(self, f.name) for f in fields(self)}}

    def describe(self):
        center = 'free center size' if self.u == 'free' else f'center size {self.u:g}'
        return f'{self.name} ({self.tilt} tilt, {center})'


def filter_from_dict(d):
    """The filter family of a tree in a witness file (Filter.to_dict)."""
    if d is None:
        return None
    if d.get('family') not in Filter.registry:
        raise ValueError(f"unknown filter family {d.get('family')!r}, known: {', '.join(sorted(Filter.registry))}")
    arguments = {k: v for k, v in d.items() if k != 'family'}
    cls = Filter.registry[d['family']]
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(arguments) - known)
    if unknown:
        raise ValueError(f"filter family {d['family']!r}: unknown fields {unknown}, its fields are {sorted(known)}")
    return cls(**arguments)


def check_u(u):
    if not (u == 'free' or (isinstance(u, (int, float)) and not isinstance(u, bool) and 0 < u < 1)):
        raise ValueError(f"u (the center size): 'free' or a number between 0 and 1 (got {u!r})")


# ------------------------------------------------------------------------------------------------- the families
@dataclass(frozen=True)
class PriceFilter(Filter):
    """One price z_j per digit [simplifications], the filter of the paper.  With the tilt 'opposite' the second list
    gets -z (representation joins), with 'same' it gets +z, and with 'own' its own prices z_y (the general filter, for
    syndrome filters).  Digits of one class share a price, and the reference digit has price 0."""
    name = 'prices'
    tilt: str = 'opposite'
    u: object = 'free'

    def variables(self, ctx):
        return [Variable((key, str(r)), -PRICE_MAX, PRICE_MAX, ('low', 'high'))
                for key in self.price_keys() for r in ctx.reps if r != ctx.reference]

    def prices(self, ctx, p):
        def price(key, j):
            return float(p[key].get(str(ctx.rep[j]), 0.0))
        z = {j: price('z', j) for j in ctx.digits}
        if self.tilt == 'opposite':
            return z, {j: -z[j] for j in z}
        if self.tilt == 'same':
            return z, dict(z)
        return z, {j: price('z_y', j) for j in ctx.digits}

    def random_start(self, ctx, rng, shared):                # price = slope * digit for small digits
        slope = rng.uniform(.1, 1.5)
        out = {}
        for key in self.price_keys():
            for r in ctx.reps:
                if r != ctx.reference:
                    out[key, str(r)] = slope * ctx.value(r) if abs(ctx.value(r)) <= 10 else rng.uniform(-2.0, 2.0)
        return out


@dataclass(frozen=True)
class LeeWeightFilter(Filter):
    """The syndrome filter of the Lee metric [lee metric]: the price of digit j is -sigma * wt_L(j), the same on both
    lists (same tilt), sigma >= 0.  The positions where both digits have a small Lee weight are the most likely to lie
    inside a center.  Two parameters with the center size."""
    name = 'lee-weight'
    tilt = 'same'
    u: object = 'free'

    def check(self, ctx):
        if ctx.q is None:
            raise ValueError('LeeWeightFilter: a Lee weight needs digits in Z_q (a q-ary problem)')
        if ctx.alike:
            raise ValueError('LeeWeightFilter: with all non-zero digits alike every filter has one price on the '
                             "digit 0.  Use LeeDecoding, or Decoding with nonzero='free'")

    def variables(self, ctx):
        return [Variable(('sigma',), 0.0, SIGMA_MAX, ('high',))]

    def prices(self, ctx, p):
        a = {j: -p['sigma'] * lee_weight(j % ctx.q, ctx.q) for j in ctx.digits}
        return a, dict(a)

    def random_start(self, ctx, rng, shared):
        return {('sigma',): rng.uniform(0.05, 1.0)}

    def default(self, path):
        return 0.3 if path == ('sigma',) else super().default(path)
