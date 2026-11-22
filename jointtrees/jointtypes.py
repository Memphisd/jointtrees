"""Joint types and [lemma 1].  Standard library only, pure functions, no knowledge of trees.

A joint type J = {(a, b): share} is the distribution of the digit pairs (a, b) of two vectors x, y that are added:
share of the positions where x has a and y has b.  Its row sums are the type of x, its column sums the type of y,
and the sums over the pairs with a + b = s give the type of the parent z = x + y.  Two quantities follow from it
[lemma 1]:

    representations(J, parent)   r = H(J) - H(parent)          z of type `parent` has 2^(r n) writings z = x + y
    loss(J)                      c = H(p_x) + H(p_y) - H(J)     a random pair (x, y) has joint type J w.p. 2^(-c n)

The model holds a joint type as pair fractions: for every parent digit s, how its positions are split over the
classes of pairs a + b = s (`pair_classes`).  `joint_type` turns the fractions into J.

    >>> table = pair_classes((-1, 0, 1), lambda a, b: a + b, 0)
    >>> J = joint_type({0: .5, 1: .5}, {'0': {'0+0': .9, '-1+1': .1}, '1': {'0+1': 1.}}, table)
    >>> sorted(J.items())
    [((-1, 1), 0.025), ((0, 0), 0.45), ((0, 1), 0.25), ((1, -1), 0.025), ((1, 0), 0.25)]
    >>> round(representations(J, {0: .5, 1: .5}), 6), round(loss(J), 6)
    (0.784498, 0.226374)
"""
from .entropy import H
from .filters import marginals


def pair_classes(digits, add, shift):
    """How a parent digit s is written as a pair (a, b) with add(a, b) (+ shift) = s.  Pairs whose sum is not a
    digit of the alphabet do not occur.  The two orders (a, b), (b, a) form one class 'a+b' and get one fraction.

        table['pairs'][s]         {class name: [ordered pairs]}
        table['parent'][s]        the key under which the fractions of parent s are stored: str(s)
        table['fractions'][key]   the class names of that key, simplest pair first (it takes the remainder)
    """
    pairs = {}
    for a in digits:
        for b in digits:
            s = add(add(a, b), shift) if shift else add(a, b)
            if s in digits:
                pairs.setdefault(s, {}).setdefault('%d+%d' % (min(a, b), max(a, b)), []).append((a, b))

    def simplicity(name, s):
        """The smallest a^2 + b^2 of the pairs (a, b) of a class: the simplest pair comes first."""
        return min((a * a + b * b, (a, b)) for a, b in pairs[s][name])
    names = {str(s): sorted(pairs[s], key=lambda name: simplicity(name, s)) for s in pairs}
    return {'pairs': pairs, 'parent': {s: str(s) for s in pairs}, 'fractions': names}


def pair_classes_alike(digits, add):
    """The same for a field with q >= 3 elements when all non-zero digits are alike (rule 3): pairs are grouped by
    their zeros, and every non-zero parent v uses the same two fractions (key 'v'):

        parent 0:   '0+0'                and  'i+(-i)'    (q - 1 pairs of non-zero digits)
        parent v:   '0+v' (v+0 and 0+v)  and  'i+(v-i)'   (q - 2 pairs of non-zero digits)
    """
    pairs = {s: {} for s in digits}
    for a in digits:
        for b in digits:
            s = add(a, b)
            if s == 0:
                name = '0+0' if a == b == 0 else 'i+(-i)'
            else:
                name = '0+v' if 0 in (a, b) else 'i+(v-i)'
            pairs[s].setdefault(name, []).append((a, b))
    return {'pairs': pairs, 'parent': {s: '0' if s == 0 else 'v' for s in digits},
            'fractions': {'0': ['0+0', 'i+(-i)'], 'v': ['0+v', 'i+(v-i)']}}


def classes(inst, shift):
    """The pair classes of an instance (cached on it): alike digits over F_q, q >= 3, or the general table."""
    if shift not in inst.cache:
        if inst.alike and len(inst.digits) >= 3:            # no shift for q >= 3 (Decoding.check)
            inst.cache[shift] = pair_classes_alike(inst.digits, inst.problem.add)
        else:
            inst.cache[shift] = pair_classes(inst.digits, inst.problem.add, shift)
    return inst.cache[shift]


def joint_type(parent, fractions, table):
    """The joint type J of a representation join: of the positions with parent digit s (share parent[s]), the
    fraction fractions[key of s][class] is written as a pair of that class, equally over its ordered pairs."""
    J = {}
    for s, mass in parent.items():
        if mass > 0:
            for name, fraction in fractions[table['parent'][s]].items():
                for pair in table['pairs'][s][name]:
                    J[pair] = J.get(pair, 0.0) + mass * fraction / len(table['pairs'][s][name])
    return J


def channel(t, digits, add):
    """The joint type of the two labels on the syndrome part at a top join with a syndrome filter: the two labels are
    uniform and their difference has type t, J(a, b) = t(a - b) / q.  [neighbour join]"""
    difference = {(a, b): next(x for x in digits if add(b, x) == a) for a in digits for b in digits}
    return {pair: t.get(x, 0.0) / len(digits) for pair, x in difference.items()}


def field_channel(t, q):
    """The same when the lists use only some digits of Z_q (restricted errors): the labels are still uniform over all
    of Z_q, J(a, b) = t(a - b) / q for a, b in Z_q.  Only the non-zero entries are stored."""
    J = {}
    for b in range(q):
        for x, share in t.items():
            if share > 0:
                J[(b + x) % q, b] = J.get(((b + x) % q, b), 0.0) + share / q
    return J


def maxent_joint_type(parent, child, digits, add, shift=0, tolerance=1e-13, limit=3000):
    """The joint type of maximum entropy with row sums = child, column sums = child and anti-diagonal sums (over
    the pairs with add(a, b) (+ shift) = s) = parent: the typical joint type of the standard join, which keeps a
    pair when its sum has the parent type [largest entropy].  Iterative proportional fitting.

    Returns (J, residual): the largest deviation of a row, column or anti-diagonal sum of J from its target.  It is
    below `tolerance` when a joint type with these types exists.  Otherwise (a digit of the child type has no partner
    that produces a digit of the parent type) it measures the unmatched mass, and the model turns it into a
    constraint, so that the optimizer is led back into the domain instead of being stopped by an error."""
    def parent_digit(a, b):
        return add(add(a, b), shift) if shift else add(a, b)
    present = [a for a in digits if child.get(a, 0.0) > 0]
    pairs = [(a, b) for a in present for b in present if parent.get(parent_digit(a, b), 0.0) > 0]
    rows, cols, antis = {}, {}, {}
    for (a, b) in pairs:
        rows.setdefault(a, []).append((a, b))
        cols.setdefault(b, []).append((a, b))
        antis.setdefault(parent_digit(a, b), []).append((a, b))
    unmatched = max([child[a] for a in present if a not in rows]
                    + [parent[s] for s in parent if parent[s] > 0 and s not in antis] + [0.0])
    if not pairs:
        return {}, max(unmatched, 1.0)
    # the same scaling with numpy is 30 times faster.  verify needs only the standard library: without numpy the
    # loop below runs
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        return _maxent_numpy(np, pairs, rows, cols, antis, child, parent, unmatched, tolerance, limit)
    J = {(a, b): child[a] * child[b] for (a, b) in pairs}
    total = sum(J.values())
    for pair in J:
        J[pair] /= total
    residual = 1.0
    for _ in range(limit):
        residual = 0.0
        for target, groups in ((child, rows), (child, cols), (parent, antis)):
            for key, group in groups.items():
                mass = sum(J[pair] for pair in group)
                residual = max(residual, abs(mass - target[key]))
                if mass > 0:
                    factor = target[key] / mass
                    for pair in group:
                        J[pair] *= factor
        if residual < tolerance and unmatched == 0.0:
            break
    return J, max(residual, unmatched)


def _maxent_numpy(np, pairs, rows, cols, antis, child, parent, unmatched, tolerance, limit):
    """The scaling of maxent_joint_type on arrays: rows, columns and anti-diagonals are index vectors."""
    index = {pair: k for k, pair in enumerate(pairs)}
    groups = []
    for target, table in ((child, rows), (child, cols), (parent, antis)):
        keys = list(table)
        ids = np.zeros(len(pairs), dtype=int)
        for g, key in enumerate(keys):
            for pair in table[key]:
                ids[index[pair]] = g
        groups.append((ids, np.array([target[key] for key in keys]), len(keys)))
    J = np.array([child[a] * child[b] for (a, b) in pairs])
    J /= J.sum()
    residual = 1.0
    for _ in range(limit):
        residual = 0.0
        for ids, target, count in groups:
            mass = np.bincount(ids, weights=J, minlength=count)
            residual = max(residual, float(np.max(np.abs(mass - target))))
            factor = np.where(mass > 0, target / np.where(mass > 0, mass, 1.0), 1.0)
            J = J * factor[ids]
        if residual < tolerance and unmatched == 0.0:
            break
    return {pair: float(J[k]) for k, pair in enumerate(pairs)}, max(residual, unmatched)


def representations(J, parent):
    """[lemma 1] (a): a parent of type P has 2^(r n) representations with joint type J, r = H(J) - H(P)."""
    return H(J) - H(parent)


def loss(J):
    """[lemma 1] (b): a random pair of children is consistent with J with probability 2^(-c n),
    c = I(X;Y) = H(p_x) + H(p_y) - H(J)."""
    x, y = marginals(J)
    return H(x) + H(y) - H(J)
