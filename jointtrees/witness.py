"""Witness files: a problem, a tree, a memory bound and the parameters that reach the claimed exponents (JSON).

    w = Witness.load(path)
    w.problem, w.tree, w.memory_bound, w.parameters (the free parameters, parameters.py), w.claimed, w.fixed,
    w.bounds, w.notes
    w.evaluate(), w.check(), w.report(), w.save(path)

docs/GUIDE.md ("Witness files") describes the format, "jointtrees witness 3".  When a file is read, its fields are
checked, the parameters are used, and everything else is checked against the model (parameters.checked): a joint
type that does not fit its level, a value under the wrong block, or a derived value that does not follow is an error
that names it.  The JSON form of the parameters (to_json, from_json, dumps) is at the end of this module.
"""
import json
import os
from dataclasses import dataclass, field

from .build import build
from .model import evaluate
from .parameters import (LEVELS, Parameters, checked, conflicts, digit_key, from_model, mismatches, ordered, text_of,
                         to_model)
from .problems import problem_from_dict
from .record import Record
from .report import report
from .tree import Tree

FORMAT = 'jointtrees witness 3'
FIELDS = ('format', 'library', 'problem', 'tree', 'memory_bound', 'fixed', 'bounds', 'claimed', 'parameters', 'derived',
          'notes')
REQUIRED = ('format', 'problem', 'tree', 'parameters')
CASES = ('both zero', 'one zero', 'both nonzero, sum zero', 'both nonzero, sum nonzero')


@dataclass
class Witness:
    """A problem, a tree, a memory bound and the free parameters (paper form) that reach the claimed exponents.
    `derived` holds the "derived" block of a file as read (the values that follow from the parameters).  It is only
    checked, never used.  `claimed`: computed from the parameters when it is None.  `fixed` and `bounds`: what
    optimize held or bounded, as optimize(fixed=..., bounds=...) takes them."""
    problem: object
    tree: object
    memory_bound: float = None
    parameters: Parameters = None
    claimed: dict = None
    notes: dict = field(default_factory=dict)
    derived: dict = None
    fixed: dict = None
    bounds: dict = None

    # ---- reading and writing
    @classmethod
    def load(cls, path):
        with open(path) as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, data):
        """A witness from the data of a file in format 3.  It is checked: its fields, the joint types that must fit
        their levels, the derived values that must follow from the parameters (a ValueError names what does not)."""
        if data.get('format') != FORMAT:
            raise ValueError(f"witness: the format is {data.get('format')!r}, and this library reads {FORMAT!r} only")
        unknown = sorted(set(data) - set(FIELDS))
        if unknown:
            raise ValueError(f'witness: unknown fields {unknown}.  The fields of {FORMAT!r} are {", ".join(FIELDS)}, '
                             'and free-form notes go into "notes"')
        missing = [key for key in REQUIRED if key not in data]
        if missing:
            raise ValueError(f'witness: the fields {missing} are missing')
        for key in ('parameters', 'derived', 'fixed', 'bounds', 'notes'):
            if key in data and not isinstance(data[key], dict):
                raise ValueError(f'witness: "{key}" is an object {{name: value}}, got {data[key]!r}')
        for key in ('parameters', 'derived'):
            where = null_in(data.get(key, {}))
            if where is not None:
                raise ValueError(f'{key}: {where} is null.  Leave out what has no value')
        digits = data['tree'].get('digits')
        if digits is None:
            raise ValueError('tree: a witness lists the digits of the lists (the rows and columns of the joint types)')
        if list(digits) != sorted(digits):
            raise ValueError(f'tree digits {digits}: write them in increasing order (the rows and columns of the joint '
                             'types follow them)')
        problem, tree = problem_from_dict(data['problem']), Tree.from_dict(data['tree'])
        bound = data.get('memory_bound')
        inst = build(problem, tree, bound)
        parameters = from_json(data['parameters'], inst, top=True)
        derived = from_json(data['derived'], inst) if 'derived' in data else None
        checked(inst, parameters, derived)
        return cls(problem, inst.tree, bound, parameters, data.get('claimed'), dict(data.get('notes', {})), derived,
                   data.get('fixed'), data.get('bounds'))

    def to_dict(self):
        """The data of the file (format 3): the parameters, and the values that follow from them under "derived".  The
        parameters are checked first."""
        from . import __version__                         # the package imports this module: a cycle at import time
        inst = self.instance
        _, evaluation, _, derived = checked(inst, self.parameters)
        claimed, notes = claimed_of(evaluation), dict(self.notes)
        if self.claimed is not None and any(abs(self.claimed.get(k, v) - v) > 1e-9 for k, v in claimed.items()):
            notes['claimed_before'] = self.claimed          # the parameters were changed: the old claim stays a note
        out = {'format': FORMAT, 'library': f'jointtrees {__version__}', 'problem': self.problem.to_dict(),
               'tree': inst.tree.to_dict(), 'memory_bound': self.memory_bound}
        if self.fixed:
            out['fixed'] = self.fixed
        if self.bounds:
            out['bounds'] = self.bounds
        out.update(claimed=claimed, parameters=to_json(self.parameters, inst), derived=to_json(derived, inst))
        if notes:
            out['notes'] = notes
        return out

    def save(self, path):
        """Write the file (format 3).  The parameters are checked first, and an existing file is replaced only when
        the new one is complete."""
        text = dumps(self.to_dict()) + '\n'
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path + '.part', 'w') as handle:
            handle.write(text)
        os.replace(path + '.part', path)

    # ---- the model
    @property
    def instance(self):
        """The instance (build.Instance) of the problem, the tree and the memory bound, built anew at every use."""
        return build(self.problem, self.tree, self.memory_bound)

    def model_parameters(self, strict=True):
        """The model's dict of the parameters (parameters.to_model)."""
        return to_model(self.instance, self.parameters, strict)

    def evaluate(self):
        """The evaluation (model.Evaluation) at the parameters, which are checked first (parameters.checked)."""
        return checked(self.instance, self.parameters)[1]

    def check(self, tolerance=1e-9):
        """[(name, text)] for every value that the model does not reproduce: in the parameters (a joint type whose
        sums do not give the level above, the remainder of a type) and in the derived block of the file."""
        inst = self.instance
        prm = to_model(inst, self.parameters)
        ev = evaluate(inst, prm)
        P, D = from_model(inst, prm, ev)
        found = conflicts(inst, self.parameters, ev, tolerance)
        found = found or mismatches(self.parameters, P, tolerance) + mismatches(self.derived or {}, D, tolerance)
        return [(name, text) for name, _, text in found]

    def report(self):
        """The report of the evaluation (report.py), with the claimed time and the legend."""
        return report(self.instance, self.evaluate(), (self.claimed or {}).get('time'), legend=True)


def claimed_of(evaluation):
    """The claimed exponents of a witness: time and memory in bits, and F (the time in base q) when a label unit is
    log2 q bits."""
    out = {'time': evaluation.time, 'memory': evaluation.memory}
    if evaluation.unit != 1.0:
        out['F'] = evaluation.F
    return out


def witness_files(path):
    if os.path.isfile(path):
        return [path]
    return sorted(os.path.join(root, f) for root, _, files in os.walk(path) for f in files if f.endswith('.json'))


def null_in(data, path=()):
    """The name of the first null in JSON data, or None if there is none."""
    if data is None:
        return text_of(path)
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = enumerate(data)
    else:
        return None
    for key, value in items:
        found = null_in(value, path + (digit_key(key),))
        if found is not None:
            return found
    return None


# --------------------------------------------------------------------------------- the JSON form of the parameters
def alike_cases(inst):
    """With all non-zero digits alike over F_q, q >= 3: the case of every pair of digits (the paper's cases)."""
    add = inst.problem.add

    def case(a, b):
        if a == 0 and b == 0:
            return CASES[0]
        if a == 0 or b == 0:
            return CASES[1]
        return CASES[2] if add(a, b) == 0 else CASES[3]
    return case


def uses_cases(inst):
    return inst.alike and len(inst.digits) >= 3


def is_levels(path):
    """True for the place of a quantity per join or level (J, p, lee_weight, filter), also in a part (parts[k].J)."""
    if not path or path[-1] not in LEVELS:
        return False
    return len(path) == 1 or (len(path) == 3 and path[0] == 'parts')


def to_json(value, inst, path=()):
    """The paper form as JSON data.  Digit and index keys are strings, and a quantity per join or level is an object
    keyed by the index.  A joint type is the matrix of its rows (rows and columns in the order of the tree's digits),
    or with the non-zero digits alike one entry per case."""
    if is_levels(path) and path[-1] == 'J':
        return {str(i): joint_type_json(J, inst) for i, J in sorted(value.items())}
    if isinstance(value, dict):
        return {str(k): to_json(v, inst, path + (k,)) for k, v in ordered(value)}
    if isinstance(value, list):
        return [to_json(v, inst, path + (i,)) for i, v in enumerate(value)]
    return value


def joint_type_json(J, inst):
    digits = inst.digits
    if uses_cases(inst):
        case, out = alike_cases(inst), {}
        for a in digits:
            for b in digits:
                out.setdefault(case(a, b), J[a][b])
        return {name: out[name] for name in CASES if name in out}
    return [[J[a][b] for b in digits] for a in digits]


def from_json(data, inst, path=(), top=False):
    """The paper form of JSON data (to_json, backwards).  It also takes the paper form itself.  top=True: the
    parameters of a file, as Parameters."""
    if is_levels(path):
        return levels(data, inst, path)
    if path == ('ell',) and not (isinstance(data, list) and len(data) == inst.depth):
        raise ValueError(f'ell: ℓ^(0), ..., ℓ^(d-1) of a tree of depth {inst.depth}, a list of {inst.depth} '
                         f'numbers (ℓ^(d) is "ell_d"), got {data!r}')
    if isinstance(data, dict):
        items = {digit_key(k): from_json(v, inst, path + (digit_key(k),)) for k, v in data.items()}
        if top:
            return Parameters(items)
        return Record(items) if all(isinstance(k, str) for k in items) else items
    if isinstance(data, list):
        return [from_json(v, inst, path + (i,)) for i, v in enumerate(data)]
    return data


def levels(data, inst, path):
    """A quantity per join or level, {i: value}, from an object keyed by the index i = 0..d."""
    name, d = text_of(path), inst.depth
    if not isinstance(data, dict):
        raise ValueError(f'{name}: an object keyed by the index i = 0..{d}, e.g. {{"{d}": ...}}, got {data!r}')
    out = {}
    for key, entry in data.items():
        i = digit_key(key)
        if not isinstance(i, int) or not 0 <= i <= d:
            raise ValueError(f'{name}: the key {key!r} is not an index 0..{d}')
        if path[-1] == 'J':
            out[i] = joint_type_of(entry, inst, f'{name}[{i}]')
        else:
            out[i] = from_json(entry, inst, path + (i,))
    return dict(sorted(out.items()))


def joint_type_of(data, inst, name='joint type'):
    """A joint type of a file: rows {a: {b: share}} over the tree's digits, from a matrix (rows and columns in the order
    of the digits), from rows {a: {b: share}}, or with the non-zero digits alike from one entry per case.  It must be
    symmetric, its entries >= 0 and their sum 1."""
    digits = inst.digits
    if isinstance(data, dict) and data and all(isinstance(digit_key(k), int) for k in data):     # rows {a: {b: share}}
        rows = {digit_key(a): {digit_key(b): v for b, v in row.items()} for a, row in data.items()}
        unknown = sorted((set(rows) | {b for row in rows.values() for b in row}) - set(digits))
        if unknown:
            raise ValueError(f'{name}: the digits {unknown} are not digits of the tree {list(digits)}')
        J = {a: {b: float(rows.get(a, {}).get(b, 0.0)) for b in digits} for a in digits}
    elif isinstance(data, dict):                                # one entry per case
        unknown = sorted(set(data) - set(CASES))
        if unknown:
            raise ValueError(f'{name}: unknown cases {unknown}.  The cases are {list(CASES)}')
        case = alike_cases(inst)
        J = {a: {b: float(data.get(case(a, b), 0.0)) for b in digits} for a in digits}
    else:
        if len(data) != len(digits) or any(len(row) != len(digits) for row in data):
            raise ValueError(f'{name}: a {len(digits)} x {len(digits)} matrix (rows and columns: the digits '
                             f'{list(digits)})')
        J = {a: {b: float(v) for b, v in zip(digits, row)} for a, row in zip(digits, data)}
    a, b = max(((a, b) for a in digits for b in digits), key=lambda ab: abs(J[ab[0]][ab[1]] - J[ab[1]][ab[0]]))
    if abs(J[a][b] - J[b][a]) > 1e-12:
        raise ValueError(f'{name}: J[{a}][{b}] = {J[a][b]!r} and J[{b}][{a}] = {J[b][a]!r} differ (a joint type is '
                         'symmetric)')
    low = min((J[a][b], a, b) for a in digits for b in digits)
    if low[0] < -1e-9:                                          # rounding of the optimizer's shares: 1e-12
        raise ValueError(f'{name}: J[{low[1]}][{low[2]}] = {low[0]!r} is negative')
    total = sum(sum(row.values()) for row in J.values())
    if abs(total - 1) > 1e-9:
        raise ValueError(f'{name}: its entries sum to {total:.12f}, not 1')
    return J


def is_scalar(value):
    return value is None or isinstance(value, (int, float, str, bool))


def dumps(data, indent=0):
    """JSON with one type, one dict of prices, one list of numbers or one matrix row per line.  A dict longer than
    400 characters, such as a type over Z_47, is written one entry per line."""
    pad, inner = ' ' * indent, ' ' * (indent + 1)
    flat = json.dumps(data)
    if isinstance(data, dict):
        if len(flat) <= 100 or (len(flat) <= 400 and all(is_scalar(v) for v in data.values())):
            return flat
        items = [f'{inner}{json.dumps(str(k))}: {dumps(v, indent + 1)}' for k, v in data.items()]
        return '{\n' + ',\n'.join(items) + '\n' + pad + '}'
    if isinstance(data, list):
        if len(flat) <= 100 or (len(flat) <= 400 and all(is_scalar(v) for v in data)):
            return flat
        if all(isinstance(v, list) and all(is_scalar(x) for x in v) for v in data):
            return '[' + (',\n' + inner).join(json.dumps(v) for v in data) + ']'
        items = [f'{inner}{dumps(v, indent + 1)}' for v in data]
        return '[\n' + ',\n'.join(items) + '\n' + pad + ']'
    return flat
