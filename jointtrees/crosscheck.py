"""A second, separately written implementation of the tree equations for subset sum and LWE.

Written from the paper without model.py: the formulas [center 1/2] for centers of size 1/2 (filters.py uses [filter
exponents]), and the list recursion written out.  It reads a witness file in format 3 (Witness.to_dict) itself: the
joint types J^(i) as matrices, the label bits ℓ^(i), the prices z and center sizes u, and not the problem classes.
With partitioning every part has its own joint types and filters, and the entropies, representations, consistency
losses and filter exponents of a join are the sums over the parts, each weighted by the size of its part.  Returns
tau, the L_i and Delta_i, or None if the witness is not SubsetSum or LWE or uses what `layout` does not cover (verify
then says so).  Standard library only.
"""
import math


def ent(values):
    return -sum(v * math.log2(v) for v in values if v > 0)


def hb(x):
    return ent([x, 1 - x]) if 0 < x < 1 else 0.0


def filtered(J, f, center):
    """alpha, beta_x, beta_y of a filter with opposite tilt on the joint type J = {(a, b): share}."""
    digits = sorted({a for a, _ in J})
    z = {a: f['z'].get(str(a), 0.0) for a in digits}

    def theta(c):
        """The share of the positions of each pair inside the center, for the offset c."""
        return {(a, b): 1 / (1 + 2 ** (-(c + z[a] - z[b]))) for (a, b) in J}
    p = {a: sum(m for (x, _), m in J.items() if x == a) for a in digits}
    if center == 0.5:                                            # [center 1/2]
        t = theta(0.0)
        alpha = sum(m * (1 - hb(t[pr])) for pr, m in J.items() if pr[0] != pr[1])
        dl = {a: sum(m * (2 * t[x, y] - 1) for (x, y), m in J.items() if x == a) for a in digits}
        beta = ent(p.values()) - (ent([p[a] + dl[a] for a in digits]) + ent([p[a] - dl[a] for a in digits])) / 2
        return alpha, beta, beta
    target = f['u'] if center == 'free' else center              # the size of the center: a parameter if free
    c, lo, hi = 0.0, -200.0, 200.0
    for _ in range(200):                                         # bisection on the center size
        c = (lo + hi) / 2
        lo, hi = (c, hi) if sum(m * theta(c)[pr] for pr, m in J.items()) < target else (lo, c)
    t = theta(c)
    size = sum(m * t[pr] for pr, m in J.items())
    betas = []
    for side in (0, 1):
        inside = {a: sum(m * t[pr] for pr, m in J.items() if pr[side] == a) for a in digits}
        betas.append(hb(size) - sum(p[a] * hb(inside[a] / p[a]) for a in digits if p[a] > 0))
    return hb(size) - sum(m * hb(t[pr]) for pr, m in J.items()), betas[0], betas[1]


def solution_and_top(problem):
    """(type of the solution, label bits of the whole label or None, label bits of the top join or None)."""
    if problem['name'] == 'SubsetSum':
        return {0: 1 - problem['weight'], 1: problem['weight']}, 1, None
    if problem['name'] == 'LWE':
        return {int(a): v for a, v in problem['key'].items()}, None, problem['ell_d']
    return None


def layout(tree):
    """(d, the center size of the filter of every join 1..d or None, dissection, accounting), or None if the tree has
    something this implementation does not cover (split joins, syndrome filters, maximum-entropy joint types, another
    filter family or tilt, many solutions)."""
    joins = tree['joins']
    if (tree.get('joint_types', 'free') != 'free' or tree.get('accounting') == 'many-solutions'
            or any(j.get('split') or j.get('syndrome') or not j.get('exact', True) for j in joins)):
        return None
    out = []
    for j in joins[1:]:
        f = j.get('filter')
        if f is None:
            out.append(None)
        elif f['family'] == 'prices' and f.get('tilt', 'opposite') == 'opposite':
            out.append(f.get('u', 'free'))
        else:
            return None
    return len(joins) - 1, out, joins[0].get('dissection', False), tree.get('accounting', 'budget')


def sum_tree(witness):
    """(tau, L_i, Delta_i) of a witness in format 3, or None if it is not covered (see `layout`).  The 7-dissection
    uses the summed entropy D_0 of the level-0 lists, also on parts."""
    problem = solution_and_top(witness['problem'])
    shape = layout(witness['tree']) if problem is not None else None
    if shape is None:
        return None
    d, filters, dissection, accounting = shape
    root, label_bits, ell_d = problem
    prm, digits = witness['parameters'], witness['tree']['digits']
    ell = list(prm['ell'])
    ell.append(ell_d if ell_d is not None else label_bits - sum(ell))         # the top join takes what is left
    if 'parts' in prm:                                                       # partitioning: the parts 1..P
        groups = [prm['parts'][str(k)] for k in range(1, witness['tree']['parts'] + 1)]
    else:
        groups = [prm]
    sizes = [group.get('size', 0.0) for group in groups]
    sizes[-1] = 1 - sum(sizes[:-1])                                          # the last part takes the rest
    D, c, r, joints_of = [0.0] * (d + 1), [0.0] * (d + 1), [0.0] * (d + 1), []
    for size, group in zip(sizes, groups):
        types, joints = {d: root}, {}
        for i in range(d, 0, -1):
            rows = group['J'][str(i)]
            joints[i] = {(a, b): rows[x][y] for x, a in enumerate(digits) for y, b in enumerate(digits)
                         if rows[x][y] > 0}
            sums = {}
            for (a, b), m in joints[i].items():
                sums[a + b] = sums.get(a + b, 0.0) + m
            if max(abs(sums.get(s, 0.0) - types[i].get(s, 0.0)) for s in set(sums) | set(types[i])) > 1e-9:
                raise ValueError(f'J[{i}]: the sums of its pairs are not the type of level {i}')
            types[i - 1] = {a: sum(m for (x, _), m in joints[i].items() if x == a) for a in {x for x, _ in joints[i]}}
        for i in range(d + 1):
            D[i] += size * ent(types[i].values())
        for i in range(1, d + 1):
            c[i] += size * (2 * ent(types[i - 1].values()) - ent(joints[i].values()))
            r[i] += size * (ent(joints[i].values()) - ent(types[i].values()))
        joints_of.append(joints)
    L = [D[0] - ell[0]]
    for i in range(1, d + 1):
        L.append(2 * L[-1] - ell[i] - c[i])
    surplus = sum(2 ** (d - i) * r[i] for i in range(1, d + 1))
    delta = [0.0] * (d + 1)
    if accounting == 'repetitions':
        delta = [max(0.0, sum((2 ** (d - j) - 1) * ell[j] for j in range(i + 1)) - surplus) for i in range(d)]
        delta.append(delta[-1])
    if dissection:
        M = witness['memory_bound']
        M0 = D[0] / 4 if M is None else min(M, D[0] / 4)
        costs = [max(D[0] / 2, 2 * (D[0] - M0) / 3, L[0]) + delta[0]]
    else:
        costs = [max(D[0] / 2, L[0]) + delta[0]]
    for i in range(1, d + 1):
        center = filters[i - 1]
        Lin = L[i - 1]
        if center is None:
            cost = max(Lin, 2 * Lin - ell[i])
        else:
            a = bx = by = 0.0
            for size, group, joints in zip(sizes, groups, joints_of):          # summed over the parts
                part_a, part_bx, part_by = filtered(joints[i], group['filter'][str(i)], center)
                a, bx, by = a + size * part_a, bx + size * part_bx, by + size * part_by
            cost = max(Lin + a - bx, Lin + a - by, 2 * Lin - ell[i] + a - bx - by)
        costs.append(cost + delta[i])
    return max(costs), L, delta
