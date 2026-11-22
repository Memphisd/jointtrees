"""The algorithm: a `Tree`, written as its joins.  The problem is a Problem (problems/), and everything else is
derived.

    may_ozerov = Tree([Join(), Join(), Join(exact=False, syndrome=PriceFilter(tilt='own'))])
    optimize(Decoding(q=2, rate=0.42), may_ozerov)

A tree of depth d has the joins 0, 1, ..., d, bottom to top.  Join 0 merges two halves into a base list (the
meet-in-the-middle of the paper, or the 7-dissection), join i combines two lists of level i - 1 into a list of level
i, and join d is the top join.  Each join is a `Join`.  The table of RULES.md (section "Joins") says which
combinations of its fields exist and what the paper calls them.
"""
from dataclasses import dataclass, fields, replace
from typing import Optional

from .families import Filter, filter_from_dict


@dataclass(frozen=True)
class Join:
    """One join.

    exact       True: matches label units exactly.  Below the top their number ell_i is optimized, and at the top
                ell_d is what the problem defines: the rest of the labels for subset sum, `ell_d` for LWE, a variable
                for decoding.  False: matches no label units exactly (ell_d = 0) and needs a syndrome filter.
    filter      the filter family on the positions the tree enumerates (decoding: the window filter), or None.
    syndrome    decoding only, top join only: the filter family on the syndrome coordinates that the window leaves
                (the syndrome part), which the top join matches by distance, or None.
    split       True: a split join.  The two children live on disjoint halves of the positions (no representations,
                no consistency loss, no filter on the tree part).
    dissection  join 0 only: the base lists by the 7-dissection [EZ23] instead of two halves.

    The paper's names: exact join (exact, no filter), filtered join (exact + filter), nearest-neighbour join
    (exact=False + syndrome), mixed join (exact + syndrome, top join only), split join.
    """
    exact: bool = True
    filter: Optional[Filter] = None
    syndrome: Optional[Filter] = None
    split: bool = False
    dissection: bool = False

    def __post_init__(self):
        for name in ('filter', 'syndrome'):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Filter):
                raise ValueError(f'Join.{name}: a filter family such as PriceFilter(...) (got {value!r})')
        if not self.exact and self.syndrome is None:
            raise ValueError('Join(exact=False) matches no labels at all: a join that does not match exactly needs a '
                             'syndrome filter (a nearest-neighbour join)')
        if self.split and self.filter is not None:
            raise ValueError('Join(split=True, filter=...): a split join has no joint type to filter')
        if self.split and self.syndrome is not None and self.exact:
            raise ValueError('Join(split=True, syndrome=...): a split top join with a syndrome filter matches no label '
                             'units exactly, write exact=False')
        if self.dissection and (self.split or self.syndrome is not None or not self.exact or self.filter is not None):
            raise ValueError('Join(dissection=True): the 7-dissection builds base lists with exact labels only')

    @property
    def kind(self):
        """The paper's name of this join."""
        if self.split:
            return 'split join' + (' with syndrome filter' if self.syndrome else '')
        if self.syndrome is not None:
            name = 'mixed join' if self.exact else 'nearest-neighbour join'
            return name + (' with window filter' if self.filter else '')
        return 'filtered join' if self.filter else 'exact join'

    def to_dict(self):
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value != f.default:
                out[f.name] = value.to_dict() if isinstance(value, Filter) else value
        return out

    @staticmethod
    def from_dict(d):
        unknown = sorted(set(d) - {f.name for f in fields(Join)})
        if unknown:
            raise ValueError(f'join: unknown fields {unknown}')
        return Join(**{k: filter_from_dict(v) if k in ('filter', 'syndrome') else v for k, v in d.items()})


@dataclass(frozen=True)
class Tree:
    """A tree: its joins 0..d (a list of `Join`), and the choices that hold for the whole tree.

    joins        Join objects for the joins 0, ..., d (bottom to top).  `Tree.same_joins` builds the common case.
    digits       the digits the lists may use, not only the digits of the solution.  More digits allow more
                 representations and change the result (subset sum: (0, 1), (-1, 0, 1), (-1, 0, 1, 2)).  None: the
                 problem's default (the digits of the solution, for decoding the digits of the error and the sums
                 of two of them).
    nonzero      Decoding only: 'alike' (default) treats all non-zero digits alike (rule 3).  'free': full joint
                 types, one price per digit, a free window type (needs a prime q, needed for a large weight over
                 F_q, q >= 3).
    accounting   'budget' (one representation of the solution must survive), 'repetitions' (subtrees are repeated
                 when representations are missing [EZ23]) or 'many-solutions' (the number of runs follows from the size
                 of the top list, and a deficit of representations lowers the top list instead of being a condition).
    joint_types  'free': the pair fractions of every join are free numbers.  'maxent': the free numbers are the
                 types of the levels, and each join uses the joint type of maximum entropy with them [largest
                 entropy].
    types        LeeDecoding with joint_types='maxent': 'free' or 'boltzmann' (the level types are Boltzmann types,
                 one mean Lee weight each, the family of the Lee-BJMM lists).
    parts        the number of parts the positions of the tree are partitioned into [partitioning]: each
                 part has its own joint types and filter parameters at every join, the label units are shared.
                 1: no partitioning.
    """
    joins: tuple
    digits: Optional[tuple] = None
    nonzero: Optional[str] = None
    accounting: str = 'budget'
    joint_types: str = 'free'
    types: str = 'free'
    parts: int = 1

    def __post_init__(self):
        joins = tuple(self.joins)
        if not joins or not all(isinstance(j, Join) for j in joins):
            raise ValueError('Tree.joins: a list of Join objects for the joins 0, 1, ..., d (at least join 0)')
        object.__setattr__(self, 'joins', joins)
        if self.digits is not None:
            object.__setattr__(self, 'digits', tuple(self.digits))
        if joins[0].split or joins[0].filter is not None:
            raise ValueError('join 0 (the base merge of two halves) has no filter on the tree part and is not a split '
                             'join.  Put the filter on join 1 or higher')
        if any(j.dissection for j in joins[1:]):
            raise ValueError('dissection: only join 0 builds base lists')
        if self.accounting not in ('budget', 'repetitions', 'many-solutions'):
            raise ValueError(f"accounting: 'budget', 'repetitions' or 'many-solutions' (got {self.accounting!r})")
        if self.nonzero not in (None, 'alike', 'free'):
            raise ValueError(f"nonzero: 'alike' or 'free' (got {self.nonzero!r})")
        if self.joint_types not in ('free', 'maxent'):
            raise ValueError("joint_types: 'free' (pair fractions) or 'maxent' (types of the levels)")
        if self.types not in ('free', 'boltzmann'):
            raise ValueError("types: 'free' or 'boltzmann'")
        if not isinstance(self.parts, int) or self.parts < 1:
            raise ValueError('parts: the number of parts of the positions (1: no partitioning)')
        for i, j in enumerate(joins[:-1]):
            if j.syndrome is not None:
                raise ValueError(f'join {i}: a syndrome filter (a nearest-neighbour or mixed join) is possible at '
                                 'the top join only')

    # ---- derived views (what build.py and the problems read)
    @property
    def depth(self):
        return len(self.joins) - 1

    @property
    def mixed(self):
        """True if the top join is a mixed join (exact and by distance)."""
        top = self.joins[-1]
        return top.exact and top.syndrome is not None

    @property
    def dissection(self):
        return self.joins[0].dissection

    @property
    def representation_joins(self):
        return [i for i in range(1, len(self.joins)) if not self.joins[i].split]

    # ---- construction
    @classmethod
    def same_joins(cls, depth, filter=None, top=None, **options):
        """Join 0 exact, and joins 1..depth of the same form, each with its own parameters: a filter of the family
        `filter` at each of them (None: exact joins).  `top`, if given, replaces the top join.
        Tree.same_joins(depth=5, filter=PriceFilter()) is the paper's filtered tree (algorithms.filtered)."""
        if not isinstance(depth, int) or depth < 0:
            raise ValueError('depth: the number of joins above the base merge (0, 1, 2, ...)')
        joins = [Join()] + [Join(filter=filter) for _ in range(depth)]
        if top is not None:
            joins[depth] = top
        return cls(joins, **options)

    def to_dict(self):
        out = {'joins': [j.to_dict() for j in self.joins]}
        for f in fields(self):
            if f.name != 'joins' and getattr(self, f.name) != f.default:
                value = getattr(self, f.name)
                out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, d):
        """A tree as written in a witness file (to_dict)."""
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f'tree: unknown fields {unknown}.  The fields are {[f.name for f in fields(cls)]}')
        if 'joins' not in d:
            raise ValueError('tree: "joins", the list of the joins 0, 1, ..., d, is missing')
        return cls([Join.from_dict(j) for j in d['joins']], **{k: v for k, v in d.items() if k != 'joins'})

    def replace(self, **changes):
        """The same tree with some fields changed, e.g. tree.replace(parts=2) or tree.replace(digits=(-1, 0, 1))."""
        return replace(self, **changes)

    def __repr__(self):
        options = ', '.join(f'{f.name}={getattr(self, f.name)!r}' for f in fields(self)
                            if f.name != 'joins' and getattr(self, f.name) != f.default)
        groups = []                                         # equal neighbours written as [Join(...)] * k
        for j in self.joins:
            if groups and groups[-1][0] == j:
                groups[-1][1] += 1
            else:
                groups.append([j, 1])
        joins = ' + '.join(f'[{join_repr(j)}]' + (f' * {k}' if k > 1 else '') for j, k in groups)
        return f'Tree({joins}' + (', ' + options if options else '') + ')'


def join_repr(join):
    args = []
    for f in fields(join):
        value = getattr(join, f.name)
        if value != f.default:
            args.append(f'{f.name}={value!r}')
    return f"Join({', '.join(args)})"
