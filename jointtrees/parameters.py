"""The parameters of a tree under the paper's names, and their conversion to and from the model's dict.  Standard
library only.

    P = result.parameters                (or Witness.load(path).parameters): the free parameters

docs/GUIDE.md ("The parameters, named as in the paper") lists the names (P.ell[i], P.ell_d, P.J[i][a][b], P.p[i],
P.filter[i], P.syndrome_filter, P.omega_1, P.parts[k], ...), and RULES.md ("The parameters of a tree") the place of
each in the model's dict.  P.ell is the list of ℓ^(0), ..., ℓ^(d-1).  A quantity per join or level (J, p,
lee_weight, filter) is a dict keyed by the index i with the free entries only, so that P.J[2] is J^(2).  The
syndrome filter of the top join is one record.

The model's dict.  The model and the optimizer work on their own dict (build.variables documents it): pair fractions
instead of joint types (the share of the positions with parent digit s that are written as a + b: the free numbers
the optimizer moves, which keep the sums of J right by construction), label units instead of bits, class totals
instead of per-digit shares, one entry per join.  `from_model` and `to_model` convert.  Each problem names its own
variables (Problem.paper_name).  The variables of the tree (joint types, level types, filters, parts) are named here,
and a filter family's variables keep their own names.
"""
import copy
import re

from .build import build, class_size, context, model_variables
from .jointtypes import classes
from .model import evaluate
from .parts import split_name
from .record import Record

LEVELS = ('J', 'p', 'lee_weight', 'filter')     # the quantities per join or level i, {i: value}, own to every part


class Parameters(Record):
    """The free parameters of a tree on a problem, under the paper's names (module docstring).  P.ell_d is None where
    ℓ^(d) is not free."""
    __slots__ = ()

    def __getattr__(self, name):
        if name == 'ell_d' and name not in self:                # ℓ^(d) where it is not free
            return None
        if name in LEVELS and name not in self and 'parts' in self:
            n = len(self['parts'])
            raise AttributeError(f'this tree has {n} parts: use P.parts[k].{name}, k = 1..{n}')
        return super().__getattr__(name)

    def __str__(self):
        return show(self)


# ---------------------------------------------------------------------------------------- paths ('filter[3].u')
TOKEN = re.compile(r'\s*(?:\.?\s*([A-Za-z_]\w*)|\[\s*(-?\d+)\s*\])')


def parse(text):
    """'parts[1].J[2][0][1]' -> ('parts', 1, 'J', 2, 0, 1)."""
    out, k = [], 0
    text = text.strip()
    if text.startswith('P.'):
        text = text[2:]
    while k < len(text):
        m = TOKEN.match(text, k)
        if not m or m.end() == k:
            raise ValueError(f'{text!r}: cannot read the name at {text[k:]!r}.  Write it as after P., e.g. filter[3].u '
                             'or J[2][0][1]')
        name, number = m.groups()
        out.append(name if name is not None else int(number))
        k = m.end()
    return tuple(out)


def text_of(path):
    """('parts', 1, 'J', 2, 0, 1) -> 'parts[1].J[2][0][1]'."""
    out = ''
    for key in path:
        if isinstance(key, int):
            out += f'[{key}]'
        elif out:
            out += '.' + key
        else:
            out += key
    return out


def lookup(P, path):
    """The value at `path`, or None if there is none."""
    node = P
    for key in path:
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            return None
        if node is None:
            return None
    return node


def put(P, path, value):
    """Set the value at `path`, creating on the way a Record where a name follows and a dict where an index or a
    digit follows."""
    node = P
    for key, following in zip(path[:-1], path[1:]):
        if key not in node:
            node[key] = Record() if isinstance(following, str) else {}
        node = node[key]
    node[path[-1]] = value


def digit_key(key):
    """'-1' -> -1 (dict keys of the model and of JSON are strings)."""
    return int(key) if isinstance(key, str) and re.fullmatch(r'-?\d+', key) else key


def node(prm, path):
    """The entry of a nested dict at `path`."""
    for key in path:
        prm = prm[key]
    return prm


def set_node(prm, path, value):
    for key in path[:-1]:
        prm = prm[key]
    prm[path[-1]] = value


# ------------------------------------------------------------------------------- the names of the model's places
def part_place(inst, name):
    """Where the parameters of a part of the model live in the paper form: () for the tree part without
    partitioning, ('parts', k) for part k."""
    kind, k = split_name(name)
    if kind != 'tree':
        raise ValueError(f'part {name!r} has no joint types or level types')
    if k is None:
        return ()
    return ('parts', k)


def filter_place(inst, name, i):
    """The place of the filter of part `name` at join i.  The syndrome filter (part 'rest') is at the top join."""
    if name == 'rest':
        return ('syndrome_filter',)
    return part_place(inst, name) + ('filter', i)


def model_place(inst, path):
    """(place in the paper form, scale) of a variable of the model's dict: paper value = scale * model value.  None
    for a pair fraction (the joint type, in the paper form, is made of them)."""
    named = inst.problem.paper_name(inst, path)
    if named is not None:
        return named
    head = path[0]
    if head == 'parts' and len(path) == 3:
        return ('parts', split_name(path[1])[1], path[2]), 1.0
    if head == 'joins':
        i, kind, name = path[1], path[2], path[3]
        if kind == 'fractions':
            return None
        if kind == 'child':
            if path[4] == 'mean':
                return part_place(inst, name) + ('lee_weight', i - 1), 1.0
            a = digit_key(path[4])
            return part_place(inst, name) + ('p', i - 1, a), 1.0 / class_size(inst, a)
        if kind == 'filter':
            return filter_place(inst, name, i) + tuple(digit_key(k) for k in path[4:]), 1.0
    return tuple(digit_key(k) for k in path), 1.0           # a problem's own parameter: under its own name


# ------------------------------------------------------------------------------------------------ the conversion
def from_model(inst, prm, ev=None):
    """(P, derived) of the model's dict `prm`: P the free parameters under the paper's names, `derived` the values that
    follow from them, the problem and the tree and are not free (ℓ^(d), the level types, κ, ω₁, the size of the
    last part, a center size that the tree fixes, the prices of a family with other parameters), in the same
    layout."""
    ev = evaluate(inst, prm) if ev is None else ev
    d = inst.depth
    P, D = Parameters(ell=[None] * d), Record()
    variables = model_variables(inst)
    for slot in variables.slots:                                # the free parameters
        place = model_place(inst, slot.path)
        if place is not None:
            try:
                value = node(prm, slot.path)
            except (KeyError, IndexError, TypeError):
                continue
            put(P, place[0], place[1] * value)
    # the level types: the full type where a type is a parameter, the others derived.  The top type p[d] is shared
    # by the parts and written once.
    for name, part in ev.by_part.items():
        place = part_place(inst, name)
        for i, t in enumerate(part['types']):
            where = (place if i < d else ()) + ('p', i)
            put(P if lookup(P, where) is not None else D, where, ev.paper_type(i, t))
        if len(ev.by_part) > 1 and lookup(P, place + ('size',)) is None:
            put(D, place + ('size',), part['size'])
    for r in variables.remainders:                              # joint types made of pair fractions
        J = ev.joins[r.join]['J'].get(r.part)
        if J is not None:
            put(P, part_place(inst, r.part) + ('J', r.join), ev.nested(J))
    if 'ell_d' not in P:
        D['ell_d'] = ev.ell_d
    if ev.kappa is not None:
        D['kappa'] = ev.kappa
        if 'omega_1' not in P:
            D['omega_1'] = ev.omega_1
    # filters: a center size that the tree fixes, and the prices of a family with other parameters
    for i, join in enumerate(ev.joins):
        for record in join['filters']:
            if record['kind'] == 'none':
                continue
            name, place = record['part'], filter_place(inst, record['part'], i)
            if name == 'rest':
                flt, ctx = inst.syndromes[i], context(inst, True)
            else:
                flt, ctx = inst.filters[i], context(inst, False)
            if flt.u != 'free':
                put(D, place + ('u',), float(flt.u))
            if not ctx.alike and lookup(P, place + ('z',)) is None:
                z, z_y = flt.price_pair(ctx, prm['joins'][i]['filter'][name])
                others = [r for r in ctx.reps if r != ctx.reference]       # the reference digit has price 0
                put(D, place + ('z',), {r: z[r] + 0.0 for r in others})   # + 0.0: no -0.0
                if flt.tilt == 'own':
                    put(D, place + ('z_y',), {r: z_y[r] + 0.0 for r in others})
    return P, D


def to_model(inst, P, strict=True):
    """The model's dict of the free parameters `P` (paper form).  strict: every free parameter must be given (a witness
    file, `evaluate`).  Otherwise missing ones keep their start values and other entries are ignored (a start made for
    another tree)."""
    variables = model_variables(inst)
    prm, missing = copy.deepcopy(variables.template), []
    for slot in variables.slots:
        place = model_place(inst, slot.path)
        if place is None:
            continue
        value = lookup(P, place[0])
        if value is None:
            missing.append(text_of(place[0]))
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'parameters: {text_of(place[0])} is {value!r}, not a number')
        else:
            set_node(prm, slot.path, float(value) / place[1])
    for r in variables.remainders:                              # pair fractions from the joint types
        J = lookup(P, part_place(inst, r.part) + ('J', r.join))
        if J is None:
            missing.append(text_of(part_place(inst, r.part) + ('J', r.join)))
            continue
        table = classes(inst, inst.shift.get(r.join, 0))
        parents = [s for s, key in table['parent'].items() if key == r.parent]
        mass = {c: sum(J.get(a, {}).get(b, 0.0) for s in parents for a, b in table['pairs'][s].get(c, []))
                for c in table['fractions'][r.parent]}
        total = sum(mass.values())
        if total > 0:
            node(prm, r.path).update({c: m / total for c, m in mass.items()})
    if strict and missing:
        raise ValueError(f'parameters: {", ".join(sorted(set(missing)))} missing')
    return prm


def checked(inst, P, derived=None, tolerance=1e-9):
    """(model's dict, evaluation, free parameters, derived values) of the free parameters P, which must give every
    free parameter of the tree and nothing else, with joint types that fit their levels.  `derived`, if given (the
    block of a file), must agree with the values that follow.  A ValueError names everything that is wrong."""
    prm = to_model(inst, P)
    ev = evaluate(inst, prm)
    P2, D2 = from_model(inst, prm, ev)
    bad = conflicts(inst, P, ev, tolerance) or mismatches(P, P2, tolerance) + mismatches(derived or {}, D2, tolerance)
    if bad:
        texts = [f'{name} {placed(name, P2, D2) if reason == "unknown" else text}' for name, reason, text in bad]
        more = f' (and {len(texts) - 6} more)' if len(texts) > 6 else ''
        raise ValueError('parameters: ' + '.  '.join(texts[:6]) + more)
    return prm, ev, P2, D2


def placed(name, P2, D2):
    """What to say about a value of a file that is not a value of the tree where the file has it: the block it
    belongs to, if it is a value of the tree."""
    path = parse(name)
    free, derived = lookup(P2, path) is not None, lookup(D2, path) is not None
    if derived and not free:
        return 'is a value that follows from the free parameters: it belongs under "derived" (or leave it out)'
    if free and not derived:
        return 'is a free parameter: it belongs under "parameters"'
    return 'is not a value of this tree'


def type_text(t):
    """A type as '(a: share, ...)', without the shares below 5e-7."""
    return '(' + ', '.join(f'{a}: {v:.6f}' for a, v in sorted(t.items()) if abs(v) > 5e-7) + ')'


def conflicts(inst, P, ev, tolerance=1e-9):
    """[(name, 'conflict', text)] for the joint types of P whose pairs a + b = s do not sum to the type of their level:
    the row sums of the next joint type above it, or at the top the type the problem (or the window) gives.  Checked
    from the top down, so the first entry is usually the one that was changed."""
    out, add = [], inst.problem.add
    for name in inst.tree_parts:
        place = part_place(inst, name)
        Js = lookup(P, place + ('J',)) or {}
        joins = sorted(i for i in Js if isinstance(i, int) and i >= 1 and Js[i] is not None)
        for n in range(len(joins) - 1, -1, -1):
            i, J = joins[n], Js[joins[n]]
            sums = {}
            for a, row in J.items():
                for b, v in row.items():
                    sums[add(a, b)] = sums.get(add(a, b), 0.0) + v
            above = Js[joins[n + 1]] if n + 1 < len(joins) else None
            if above is not None:
                target = {a: sum(row.values()) for a, row in above.items()}
                source = 'the rows of the joint type above it give'
            else:
                target, source = ev.paper_type(i, ev.by_part[name]['types'][i]), f'the type of level {i} is'
            if max(abs(sums.get(s, 0.0) - target.get(s, 0.0)) for s in set(sums) | set(target)) > tolerance:
                out.append((text_of(place + ('J', i)), 'conflict',
                            f'has pairs a + b = s that sum to {type_text(sums)}, but {source} {type_text(target)} '
                            '(change the joint types of a part together)'))
    return out


def mismatches(given, computed, tolerance=1e-9, path=()):
    """[(name, reason, text)] for the numbers of `given` that differ from `computed` by more than `tolerance` (reason
    'differs'), that `computed` does not have ('unknown'), or that are not numbers ('not a number')."""
    out = []
    if isinstance(given, dict):
        for key, value in given.items():
            sub = computed.get(key) if isinstance(computed, dict) else None
            if value is not None:
                out += mismatches(value, sub, tolerance, path + (key,))
        return out
    if isinstance(given, list):
        for key, value in enumerate(given):
            sub = computed[key] if isinstance(computed, list) and key < len(computed) else None
            if value is not None:
                out += mismatches(value, sub, tolerance, path + (key,))
        return out
    if isinstance(given, (int, float)) and not isinstance(given, bool):
        if not isinstance(computed, (int, float)):
            return [(text_of(path), 'unknown', 'is not a value of this tree')]
        if abs(given - computed) > tolerance:
            return [(text_of(path), 'differs', f'is {given!r}, the model gives {computed!r}')]
        return out
    return out + [(text_of(path), 'not a number', f'is {given!r}, not a number')]


def at_search_bounds(inst, prm):
    """The variables of `prm` that lie on a pure search-box bound (see build.variables): the optimum may lie
    outside."""
    out = []
    for slot in model_variables(inst).slots:
        value = prm
        try:
            for key in slot.path:
                value = value[key]
        except (KeyError, IndexError, TypeError):
            continue
        for side, bound in (('low', slot.low), ('high', slot.high)):
            if side in slot.search and abs(value - bound) <= 1e-7 * max(1.0, abs(bound)):
                place = model_place(inst, slot.path)
                if place is None:
                    name, scale = '.'.join(map(str, slot.path)), 1.0
                else:
                    name, scale = text_of(place[0]), place[1]
                out.append(f'{name} is at its search bound {bound * scale:g}')
    return out


ORDER = ('ell', 'ell_d', 'omega_1', 'kappa', 'lee_weight', 'p', 'J', 'filter', 'syndrome_filter', 'parts', 'size', 'u',
         'z', 'z_y', 'sigma')


def order_key(item):
    """The place of an entry of a paper-form dict: the names of ORDER first, then the integer keys (digits, indices)
    in increasing order, then the other names."""
    rank = ORDER.index(item[0]) if item[0] in ORDER else len(ORDER)
    if isinstance(item[0], int):
        return rank, 0, item[0], ''
    return rank, 1, 0, str(item[0])


def ordered(P):
    """The entries of a paper-form dict in the order of the paper (ORDER first, then the others)."""
    return sorted(P.items(), key=order_key)


# --------------------------------------------------------------------------------------------- printing, listing
def fmt(value):
    if value is None:
        return 'none'
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def show(P, indent=''):
    """The parameters as text, one quantity per line."""
    lines = []
    for key, value in ordered(P):
        if key == 'parts' and isinstance(value, dict):
            for k, part in value.items():
                lines.append(f'{indent}part {k}:')
                lines.append(show(part, indent + '  '))
            continue
        if isinstance(value, list) or (key in LEVELS and isinstance(value, dict)):
            entries = sorted(value.items()) if isinstance(value, dict) else enumerate(value)
            for i, entry in entries:
                if key == 'J':
                    digits = list(entry)
                    lines.append(f'{indent}J[{i}]   rows and columns: the digits {digits}')
                    if len(digits) <= 9:
                        lines += [f'{indent}    ' + ' '.join(f'{entry[a][b]:9.6f}' for b in digits) for a in digits]
                elif isinstance(entry, dict):
                    shown = ', '.join(f'{k}: {show_value(v)}' for k, v in ordered(entry))
                    lines.append(f'{indent}{key}[{i}]  {shown}')
                else:
                    lines.append(f'{indent}{key}[{i}]  {fmt(entry)}')
        elif isinstance(value, dict):
            lines.append(f'{indent}{key}  ' + ', '.join(f'{k}: {show_value(v)}' for k, v in ordered(value)))
        else:
            lines.append(f'{indent}{key}  {fmt(value)}')
    return '\n'.join(lines)


def show_value(v):
    if isinstance(v, dict):
        return '(' + ', '.join(f'{k}: {fmt(x)}' for k, x in v.items()) + ')'
    return fmt(v)


# ------------------------------------------------------------------------------------------------- the table
def names(inst):
    """The names of the free parameters of an instance, in the order of the model's variables.  A joint type, whose
    entries the optimizer moves through its shares per parent digit, appears once as 'J[i]'."""
    out = []
    for slot in model_variables(inst).slots:
        place = model_place(inst, slot.path)
        if place is None:
            name = text_of(part_place(inst, slot.path[3]) + ('J', slot.path[1]))
        else:
            name = text_of(place[0])
        if name not in out:
            out.append(name)
    return out


def parameter_names(problem, tree, memory_bound=None):
    """The names of the free parameters of `tree` on `problem` (a joint type: 'J[i]'), as fixed= takes them."""
    return names(build(problem, tree, memory_bound))


def leaf_paths(node, path=()):
    """The paths of the numbers in a paper-form dict."""
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return [path]
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return []
    return [leaf for k, v in items if v is not None for leaf in leaf_paths(v, path + (k,))]


def leaf_values(tree):
    """{path: number} of a paper-form dict."""
    return {leaf: lookup(tree, leaf) for leaf in leaf_paths(tree)}


def meaning(path, problem=None):
    """What the value at `path` is, in words (a problem's own parameters: its `meanings`)."""
    prefix, rest = '', path
    if path[:1] == ('parts',) and len(path) > 2:
        prefix, rest = f' (part {path[1]})', path[2:]
    head = rest[0]
    if head == 'ell':
        return f'label bits matched exactly at join {rest[1]} (bits per n)' + prefix
    if head == 'ell_d':
        return 'label bits matched exactly at the top join (bits per n)' + prefix
    if head == 'omega_1':
        return 'weight of the error on the window (fraction of n)' + prefix
    if head == 'kappa':
        return 'size of the window R + ℓ, ℓ = sum of the ℓ^(i) / log2 q (fraction of n)' + prefix
    if head == 'p':
        digit = f': share of the digit {rest[2]}' if len(rest) > 2 else ''
        return f'type of level {rest[1]}{digit}' + prefix
    if head == 'J':
        return (f'joint type of the join into level {rest[1]}' if len(rest) > 1 else 'joint type') + prefix
    if head == 'lee_weight':
        return f'relative Lee weight of the type of level {rest[1]}' + prefix
    if head == 'size':
        return 'share of the positions of the tree part in this part (decoding: of the window)' + prefix
    if problem is not None and head in getattr(problem, 'meanings', {}):
        return problem.meanings[head] + prefix
    if head in ('filter', 'syndrome_filter'):
        if head == 'filter':
            where, tail = f'filter of join {rest[1]}', rest[2:]
        else:
            where, tail = 'syndrome filter of the top join', rest[1:]
        key = tail[0] if tail else ''
        if key in ('z', 'z_y'):
            side = 'first list' if key == 'z' else 'second list'
            what = f'price of the digit {tail[1]} ({side})' if len(tail) > 1 else f'prices of the digits ({side})'
        else:
            what = {'u': 'center size u', 'sigma': 'σ', '': 'parameters'}.get(key, key)
        return f'{what} of the {where}' + prefix
    return ''


def fixed_by_problem(inst, D):
    """The paths of the derived values D (at the template) that the problem or the tree fixes: those that are equal
    at a second point, where every variable has moved a tenth of the way to a point of its range (spread by the
    golden ratio, so that no two variables move alike) and every pair fraction is 0.03 off its first class."""
    variables = model_variables(inst)
    moved = copy.deepcopy(variables.template)
    for k, slot in enumerate(variables.slots):
        if 'fractions' not in slot.path:
            value = node(moved, slot.path)
            target = slot.low + (slot.high - slot.low) * (0.3 + 0.4 * ((0.618 * k) % 1))
            set_node(moved, slot.path, value + 0.1 * (target - value))
    for r in variables.remainders:
        fractions = node(moved, r.path)
        for c in fractions:
            if c == r.first:
                fractions[c] = 1 - 0.03 * (len(fractions) - 1) / len(fractions)
            else:
                fractions[c] = 0.03 / len(fractions)
    try:
        second = leaf_values(from_model(inst, moved)[1])
    except (ValueError, ZeroDivisionError, OverflowError):    # the second point is outside the domain of the formulas
        return set()
    return {leaf for leaf, v in leaf_values(D).items() if leaf in second and abs(v - second[leaf]) < 1e-15}


def is_digit_dict(node):
    """True for a non-empty dict of numbers keyed by digits: a type or the prices of a filter."""
    if not isinstance(node, dict) or not node:
        return False
    return all(isinstance(k, int) and isinstance(v, (int, float)) for k, v in node.items())


def parameter_table(problem, tree, memory_bound=None):
    """The parameters of `tree` on `problem` as a table: every name (as `optimize(..., fixed=...)` takes it), whether
    it is free, follows from the free ones (derived) or is fixed by the problem or the tree (given), its range, and
    what it is."""
    inst = build(problem, tree, memory_bound)
    variables = model_variables(inst)
    rows, free = [], set()
    for slot in variables.slots:
        place = model_place(inst, slot.path)
        if place is None:
            continue
        free.add(place[0])
        lo, hi = sorted((slot.low * place[1], slot.high * place[1]))
        rows.append((text_of(place[0]), 'free', f'{lo:g} .. {hi:g}', meaning(place[0], problem)))
    for r in variables.remainders:
        J = part_place(inst, r.part) + ('J', r.join)
        rows.append((text_of(J), 'free', 'shares',
                     meaning(J) + ', the optimizer moves the shares of the pairs per digit of the parent'))
    try:
        P, D = from_model(inst, variables.template)
    except (ValueError, ZeroDivisionError, OverflowError):    # the template is outside the domain of the formulas
        P, D = Parameters(), Record()
    given = fixed_by_problem(inst, D)

    def walk(node, path, kind):
        if node is None or path[-2:-1] == ('J',):               # a joint type: one row (above)
            return
        if isinstance(node, (int, float)):
            if path not in free:
                rows.append((text_of(path), 'given' if path in given else kind, '', meaning(path, problem)))
            return
        if is_digit_dict(node) and path[-1] not in LEVELS:
            derived = [k for k in node if path + (k,) not in free]          # a type or prices: one row
            if len(derived) == len(node):
                if all(path + (k,) in given for k in derived):
                    kind_of_type = 'given'
                else:
                    kind_of_type = 'derived' if kind == 'free' else kind
                rows.append((text_of(path), kind_of_type, '', meaning(path)))
            else:                                       # the entries of a free type that follow from the others
                rows.extend((text_of(path + (k,)), 'given' if path + (k,) in given else 'derived', '',
                             meaning(path + (k,))) for k in derived)
            return
        items = node.items() if isinstance(node, dict) else enumerate(node)
        for k, v in items:
            walk(v, path + (k,), kind)
    walk(P, (), 'free')
    walk(D, (), 'derived')
    seen, unique = set(), []
    for row in rows:                                    # a joint type, or a derived entry of a free type, appears once
        if row[0] not in seen:
            seen.add(row[0])
            unique.append(row)
    width = max(len(r[0]) for r in unique) if unique else 10
    lines = [f'parameters of {tree!r} on {problem!r}',
             'free: optimized (fixed= holds it, bounds= narrows its range).  derived: follows from the free values '
             '(fixed= or bounds= on it become constraints).  given: fixed by the problem or the tree', '']
    lines += [f'{name:{width}s}  {kind:7s}  {rng:17s}  {text}' for name, kind, rng, text in unique]
    return '\n'.join(lines)
