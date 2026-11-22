"""The parts of a solution and of its label, and what each part contributes at each join.  Standard library only.

The model never looks at a problem as a whole.  The solution and its label are cut into parts
(problem.parts), and every quantity of a join is the sum over the parts of a local contribution, a `Share`.  The
types of a part are computed top-down: on the top level it has the type of the solution (or of the target), and each
join it takes part in determines its type one level down.  Three kinds of parts exist:

    TreePart      the coordinates the tree enumerates: the whole vector (subset sum, LWE) or the window of a
                  decoding problem.  Split by a joint type at every representation join, halved at a split join.
    ExactPart     ell_j label units that join j matches exactly.  Above join j both children carry the units.
    SyndromePart  decoding: the syndrome coordinates that the window leaves, with the error weight that it leaves
                  (`rest` in the model).  The top join checks them: by distance if it has a syndrome filter (the
                  two labels must differ by a vector of this weight), and always by the final weight check.

What a part contributes at join i, the fields of a `Share` (all exponents per n, in bits):

    r         representations of the solution gained: the tree part at a representation join, [lemma 1] (a)
    c         consistency loss, [lemma 1] (b): every part that is split by a joint type at this join
    c_label   loss from the target condition: exact units at their join (u bits each)
    final     the weight condition on the syndrome part (u - H(t) per coordinate): the final check, -log2 Q, which
              every candidate of the top list must pass [many solutions].  It is kept apart from c_label, so that
              L_d is the size of the top list before it
    ell       label bits matched exactly at this join (they shorten the list of candidate pairs), exact units only
    K         constraints that the part puts on the representations of the solution [budget]: a part above its own
              join fixes both children.  Counted once per join of this level (2^(d-i) copies)
    J         joint type of the two children of this part at this join (what a filter sees)
    child     (type, size) of the part one level down, None at the join that introduces it.  At join 0 the children
              are the two halves of a base list, stored as level -1.

A part marked `remainder` takes what the others leave (`rest`, or the label units of the top join of subset sum).
Its size is not a free parameter, and its sign is a constraint of the problem, not a domain constraint.  The parts of
an instance are built by its problem (problems/: problem.parts).  A part's name is its kind and, for a part of
several, its index: 'tree', 'tree_2', 'ell_0', 'rest' (`split_name`).
"""
from dataclasses import dataclass, field

from .build import child_type_of, expand
from .entropy import H
from .filters import marginals
from .jointtypes import channel, classes, field_channel, joint_type, loss, maxent_joint_type, representations


@dataclass
class Share:
    r: float = 0.0
    c: float = 0.0
    c_label: float = 0.0
    ell: float = 0.0
    final: float = 0.0
    K: float = 0.0
    J: dict = None
    child: tuple = None
    residual: float = 0.0           # maxent joint types: deviation of J from the required sums (0 inside the domain)


@dataclass
class Part:
    name: str
    size: float                     # fraction of n
    type: dict                      # digit frequencies on the top level
    intro: int                      # the join that introduces the part (0 = the base merge)
    weight: float = None            # decoding problems: weight of the error on the part (fraction of n)
    types: dict = field(default_factory=dict)       # level -> type, filled top-down by model.contributions
    sizes: dict = field(default_factory=dict)       # level -> size
    remainder: bool = False         # takes what the other parts leave: its sign is a constraint of the problem

    kind = 'part'

    def share(self, i, inst, prm):
        raise NotImplementedError


@dataclass
class TreePart(Part):
    kind = 'tree'

    def share(self, i, inst, prm):
        t, size = self.types[i], self.sizes[i]
        if i == 0 or inst.tree.joins[i].split:              # base merge or split join: the children are the halves
            return Share(child=(t, size / 2))
        if inst.joint_types == 'maxent':                    # the child type is given, J is the max-entropy table
            given = prm['joins'][i]['child'][self.name]
            if 'mean' in given:                             # types='boltzmann': the Boltzmann type of that Lee weight
                child = inst.problem.boltzmann(given['mean'], inst)
            else:
                child = child_type_of(inst, t, expand(inst, {int(a): s for a, s in given.items()}))
            if min(child.values()) < -1e-12:
                raise ValueError(f'join {i}: the type of level {i - 1} (p[{i - 1}]) has a negative share')
            J, residual = maxent_joint_type(t, child, inst.digits, inst.problem.add, inst.shift.get(i, 0))
            if not J:
                raise ValueError('no joint type: the child type cannot produce the parent type at all')
            return Share(J=J, r=size * representations(J, t), c=size * loss(J), child=(child, size),
                         residual=residual)
        J = joint_type(t, prm['joins'][i]['fractions'][self.name], classes(inst, inst.shift.get(i, 0)))
        return Share(J=J, r=size * representations(J, t), c=size * loss(J), child=(marginals(J)[0], size))


@dataclass
class ExactPart(Part):
    kind = 'exact'

    def share(self, i, inst, prm):
        size, u = self.sizes[i], inst.unit
        if i == self.intro:                                 # matched here: a pair survives w.p. 2^(-size u n)
            return Share(c_label=size * u, ell=size * u)
        return Share(K=2 ** (inst.depth - i) * size * u, child=({0: 1.0}, size))     # fixed on both children


@dataclass
class SyndromePart(Part):
    kind = 'syndrome'

    def share(self, i, inst, prm):
        """At the top join: the joint type of the two labels if a syndrome filter matches them by distance (their
        difference has type t, and the joint type is None at an exact top join), and the final weight check."""
        t, size, u = self.types[i], self.sizes[i], inst.unit
        J = None
        if i in inst.syndromes:
            # the labels are uniform over Z_q, also if the lists use fewer digits
            if inst.q is None or len(inst.digits) == inst.q:
                J = channel(t, inst.digits, inst.problem.add)
            else:
                J = field_channel(t, inst.q)
        return Share(J=J, final=size * (u - H(t)))


@dataclass
class Parts:
    """The parts of an instance at given parameters: the tree part, the exact units ell_0..ell_d, and for a
    decoding problem the syndrome part `rest`.

    `tree` is the tree part as the problem defines it (the whole vector, or the window of a decoding problem).  With
    partitioning, `trees` holds its parts (`partition`), and without, `trees` is [tree].  The model works on `trees`,
    and the problem's permutations and constraints see the whole `tree`."""
    tree: TreePart
    exact: list
    rest: SyndromePart = None
    trees: list = None

    def __post_init__(self):
        if self.trees is None:
            self.trees = [self.tree]

    def __iter__(self):
        yield from self.trees
        yield from self.exact
        if self.rest is not None:
            yield self.rest


def partition(inst, prm, parts):
    """[partitioning] The tree part cut into P parts of sizes eps_1..eps_P (fractions of the tree part, the last takes
    the rest), fixed at random and the same for every list.  Every part has the top-level type of the whole tree part
    (one of polynomially many random partitions achieves this) and its own joint types and filter parameters.  The
    label units and the syndrome part are shared.  Problem-independent: the problem only defines the tree part."""
    names = inst.tree_parts
    if len(names) == 1:
        return parts
    whole = parts.tree
    eps = [prm['parts'][name]['size'] for name in names[:-1]]
    eps.append(1.0 - sum(eps))
    parts.trees = [TreePart(name, e * whole.size, dict(whole.type), whole.intro,
                            None if whole.weight is None else e * whole.weight) for name, e in zip(names, eps)]
    return parts


def split_name(name):
    """The kind and the index of a part's name: 'tree_2' -> ('tree', 2), 'ell_0' -> ('ell', 0), 'tree' -> ('tree',
    None), 'rest' -> ('rest', None)."""
    kind, _, index = name.partition('_')
    if not index:
        return kind, None
    return kind, int(index)


def label(inst, name):
    """The name of a part of the model as the report, the constraints and the paper write it: the window (decoding) or
    the tree part, part k (partitioning), the syndrome part, the label bits ell[j] and ell_d."""
    kind, index = split_name(name)
    if kind == 'tree' and index is None:
        return 'window' if inst.rate is not None else 'tree part'
    if kind == 'tree':
        return f'part {index}'
    if kind == 'ell':
        return 'ell_d' if index == inst.depth else f'ell[{index}]'
    if kind == 'rest':
        return 'syndrome part'
    return name
