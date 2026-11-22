"""Starts for one tree made from the optimum of another (steps of optimization strategies).

    from jointtrees.starts import deeper, embed, into_parts, with_digits
    start = embed(result, problem, other_tree)              another tree: new features switched off
    start = deeper(result, problem, tree_d_plus_1)          one more join at the bottom
    start = into_parts(result, parts=2)                      the same parameters in every part
    start = with_digits(result, problem, tree_more_digits)   more digits, at share 0
    optimize(problem, new_tree, warm=start)

Each takes a Result, a Witness or parameters (paper form) and returns parameters (paper form) for the new tree,
which can be inspected or edited before they are used.  `evaluate(problem, new_tree, start).time` shows how good
the start is: `into_parts` and `with_digits` keep the time of the old optimum, and `deeper` does not, since the new
join changes the lists below it.  `optimize(..., warm=old_result)` with a result or witness of another tree uses
`embed` by itself.
"""
import copy

from .build import build, context, model_variables
from .families import PRICE_MAX
from .model import evaluate
from .parameters import LEVELS, Parameters, filter_place, from_model, put, to_model
from .record import Record


def parameters_of(start):
    """A copy of the free parameters (paper form) of a Result, a Witness or parameters."""
    P = getattr(start, 'parameters', start)
    if not isinstance(P, dict):
        raise ValueError('a start: a Result, a Witness or parameters (paper form)')
    return copy.deepcopy(P)


def embed(start, problem, tree, memory_bound=None):
    """Parameters for `tree` from those of another tree (a Result or a Witness, or parameters): what matches by name is
    kept, and everything new starts at its neutral value, so that a tree that contains the old one starts at the old
    time: a new filter with prices 0 (it filters nothing) and center size 1/2, a new ℓ^(d) of size 0.  Where a filter
    changed its family or its tilt, the new filter starts with the old filter's prices and center size (e.g.
    LeeWeightFilter -> PriceFilter('own'), or the opposite tilt -> own prices)."""
    P = parameters_of(start)
    inst = build(problem, tree, memory_bound)
    old = getattr(start, 'instance', None)                  # a Result or a Witness: the instance of the old tree
    d = inst.depth
    if old is not None and old.depth == d:                  # filters whose family or tilt changed: their prices
        old_prm = to_model(old, start.parameters, strict=False)
        for i in range(d + 1):
            for syndrome, new_flt, old_flt in ((False, inst.filters.get(i), old.filters.get(i)),
                                               (True, inst.syndromes.get(i), old.syndromes.get(i))):
                if new_flt is None or old_flt is None:
                    continue
                if (type(new_flt), new_flt.tilt) == (type(old_flt), old_flt.tilt):
                    continue
                names = ['rest'] if syndrome else inst.tree_parts
                for name in names:
                    f_old = old_prm['joins'][i].get('filter', {}).get(name)
                    if f_old is None:
                        continue
                    z, z_y = old_flt.price_pair(context(old, syndrome), f_old)
                    ctx = context(inst, syndrome)
                    keep = [0] if ctx.alike else [r for r in ctx.reps if r != ctx.reference]
                    new = Record(z={r: z.get(r, 0.0) for r in keep})
                    if new_flt.tilt == 'own':
                        new['z_y'] = {r: z_y.get(r, 0.0) for r in keep}
                    new['u'] = f_old.get('u', old_flt.u if old_flt.u != 'free' else 0.5)
                    put(P, filter_place(inst, name, i), new)
    return from_model(inst, to_model(inst, P, strict=False))[0]


def into_parts(start, parts=2, sizes=None):
    """The parameters of a tree without partitioning, written for `parts` parts that all have the joint types, level
    types and filters of the one part (so the time is the same): the start for Tree(..., parts=parts).  `sizes`: the
    sizes of the parts, all of them or all but the last (default: equal parts)."""
    P = parameters_of(start)
    if 'parts' in P:
        raise ValueError('into_parts: these parameters are already partitioned')
    sizes = [1.0 / parts] * parts if sizes is None else list(sizes)
    if len(sizes) == parts - 1:
        sizes.append(1.0 - sum(sizes))
    if len(sizes) != parts or min(sizes) < 0 or abs(sum(sizes) - 1) > 1e-9:
        raise ValueError(f'into_parts: {parts} sizes >= 0 that sum to 1, or the first {parts - 1} (got {sizes})')
    d = len(P['ell'])
    out = Parameters({key: value for key, value in P.items() if key not in LEVELS})
    for key in ('p', 'lee_weight'):                         # the top level is shared by the parts
        if d in P.get(key, {}):
            out[key] = {d: P[key].pop(d)}
    own = {key: P[key] for key in LEVELS if P.get(key)}
    out['parts'] = {k: Record({**({'size': size} if k < parts else {}), **copy.deepcopy(own)})
                    for k, size in enumerate(sizes, 1)}
    return out


def deeper(start, problem, tree, new='balanced', share=0.98):
    """Parameters for `tree`, which has one join more than the tree of `start`: a new lowest representation join
    is inserted below the old one, and the joins above it move up by one level.

    new='balanced'  the new join writes every digit of its parent as the simplest pair with that sum (0 = 0 + 0,
                    1 = 0 + 1, 2 = 1 + 1, ...) at `share`, and the other pairs share the rest equally.  Its filter (if
                    the tree has one) starts switched off
    new='copy'      the new join gets the pair shares and the filter of the old lowest join
    ℓ: the label bits ℓ of the old lowest lists are split.  The new lowest lists get as many as the new join
    has representations (r, at most ℓ), and the lists above them the rest, max(0, ℓ - r).  If r <= ℓ the old
    levels keep their list sizes.  With maximum-entropy joint types the new level gets half the non-zero shares of
    the level above it (Boltzmann types: 0.55 of its Lee weight)."""
    if new not in ('balanced', 'copy'):
        raise ValueError("deeper: new='balanced' or new='copy'")
    P = parameters_of(start)
    inst = build(problem, tree)
    d = inst.depth
    pos = min(tree.representation_joins) if tree.representation_joins else None
    if new == 'copy' and pos is not None and inst.shift.get(pos + 1, 0) != inst.shift.get(pos, 0):
        raise ValueError("deeper(new='copy'): the old lowest join works on e - (1,...,1) (rule 4) and the new one does "
                         "not, so their pairs differ.  Use new='balanced'")
    if len(P['ell']) != d - 1:
        raise ValueError(f'deeper: the tree must have one join more than the start (the start has {len(P["ell"])} '
                         f'joins above join 0, the tree {d})')
    # the joins from pos up and the levels from pos - 1 up move up by one, and the new join and level start empty
    # (with new='copy' the new join has the joint type and filter of the old lowest join)
    for group in [P] + list(P.get('parts', {}).values()):
        for key, at in (('J', pos), ('filter', pos), ('p', pos - 1), ('lee_weight', pos - 1)):
            if key not in group:
                continue
            copied = new == 'copy' and key in ('J', 'filter') and at in group[key]
            entry = copy.deepcopy(group[key][at]) if copied else None
            group[key] = {(i + 1 if i >= at else i): v for i, v in group[key].items()}
            if entry is not None:
                group[key][at] = entry
            group[key] = dict(sorted(group[key].items()))
    before = P['ell'][pos - 1]                              # bits of the old lowest join, split below
    P['ell'].insert(pos - 1, before)
    prm = to_model(inst, P, strict=False)
    for r in model_variables(inst).remainders:              # the pair shares of the new join
        if r.join == pos:
            fractions = prm['joins'][pos]['fractions'][r.part][r.parent]
            if new == 'copy':
                fractions.update(copy.deepcopy(prm['joins'][pos + 1]['fractions'][r.part][r.parent]))
                continue
            others = [c for c in fractions if c != r.first]
            for c in fractions:
                if not others:
                    fractions[c] = 1.0
                elif c == r.first:
                    fractions[c] = share
                else:
                    fractions[c] = (1 - share) / len(others)
    for name, child in prm['joins'][pos].get('child', {}).items():    # maximum-entropy joint types: the new level
        above = prm['joins'][pos + 1]['child'][name]
        if 'mean' in child:
            child['mean'] = 0.55 * above['mean']
        else:
            for digit in child:
                child[digit] = above.get(digit, 0.0) / 2
    if before is not None:
        r = evaluate(inst, prm).joins[pos]['r']             # bits: the representations of the new join
        prm['ell'][pos - 1] = min(r, before) / inst.unit
        if pos < len(prm['ell']):
            prm['ell'][pos] = max(0.0, before - r) / inst.unit
    return from_model(inst, prm)[0]


def with_digits(start, problem, tree):
    """Parameters for `tree`, whose lists use more digits than the tree of `start`: the new digits have the share 0
    in every joint type and level type (the time stays the same), and their prices continue the prices of the two
    outermost old digits on their side in a straight line (clipped to ±40)."""
    P = parameters_of(start)
    inst = build(problem, tree)
    digits = inst.digits

    def joint_type(J):
        missing = set(J) - set(digits)
        if missing:
            raise ValueError(f'with_digits: the new tree lacks the digits {sorted(missing)} of the start')
        return {a: {b: J.get(a, {}).get(b, 0.0) for b in digits} for a in digits}

    def widen(node, key=None):
        if key == 'J':                                      # {i: joint type}: the new digits at share 0
            return {i: joint_type(J) for i, J in node.items()}
        if key == 'p':                                      # {i: level type}: the new digits at share 0
            return {i: {a: t.get(a, 0.0) for a in sorted(set(t) | set(digits))} for i, t in node.items()}
        if key in ('z', 'z_y'):
            return extend(node, digits)
        if isinstance(node, dict):
            return type(node)({k: widen(v, k if isinstance(k, str) else None) for k, v in node.items()})
        return node
    return from_model(inst, to_model(inst, widen(P), strict=False))[0]


def extend(prices, digits):
    """The prices of new digits on a straight line through the two outermost old ones of each side (the reference
    digit 0, whose price 0 is not listed, counts as an old digit)."""
    out = dict(prices)
    if 0 in digits and 0 not in out:
        out[0] = 0.0
    old = sorted(k for k in out if isinstance(k, int))
    if len(old) < 2:
        return out
    for a in digits:
        if a in out:
            continue
        if a > old[-1]:
            x0, x1 = old[-2], old[-1]
        elif a < old[0]:
            x0, x1 = old[1], old[0]
        else:
            continue
        slope = (out[x1] - out[x0]) / (x1 - x0)
        out[a] = max(-PRICE_MAX, min(PRICE_MAX, out[x1] + slope * (a - x1)))
    if 0 not in prices:
        out.pop(0, None)
    return dict(sorted(out.items()))
