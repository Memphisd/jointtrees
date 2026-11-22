"""For maintainers: write the generated blocks of RULES.md (constraints, search bounds, references) from the code.

    python3 tools/rules_table.py          rewrites the blocks between <!-- name:begin --> and <!-- name:end -->
    python3 tools/rules_table.py --check  exit code 1 if RULES.md is not up to date (used by tests/test_library.py)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from jointtrees import build, families                                  # noqa: E402
from jointtrees.references import LITERATURE, PAPER                     # noqa: E402

# name pattern as produced by model.constraints (i = a level or join), formula, reference key or a sentence, group
CONSTRAINTS = [
    ('saturation: L_i <= D_i - fixed_i',
     'a list of level i, 1 <= i < d, has at most as many elements as there are vectors of its type with its fixed '
     'label bits (at level 0 this holds by construction)', 'saturation', 'model'),
    ('saturation: L_d <= D_d - fixed_d', "the same for the top list, only with `accounting='many-solutions'`",
     'many solutions', 'model'),
    ('representation budget >= 0', "sum_i 2^(d-i) r_i >= sum_parts K_part, only with `accounting='budget'`", 'budget',
     'model'),
    ('memory: L_half <= M', 'the two halves of a base list are stored (not with `dissection`)', 'tree cost', 'model'),
    ('memory: L_i <= M', 'every list below the top is stored (i < d)', 'tree cost', 'model'),
    ('7-dissection: D_0/7 <= M', 'the 7-dissection needs memory 2^(D_0 n / 7)', 'dissection', 'model'),
    ('syndrome coordinates: exact <= (1 - R) n',
     'size of the syndrome part = 1 - R - (sum_j ell[j] + ell_d)/log2 q >= 0', 'isd', 'model'),
    ('error weight: window <= ω', 'weight of the syndrome part = ω - ω₁ >= 0', 'isd', 'model'),
    ('error type: share of digit X on the syndrome part >= 0',
     "free window type (TypedDecoding and its subclasses, or `nonzero='free'`): every digit the error has is left over "
     'for the syndrome part', 'isd', 'model'),
    ('label bits left for the top join: ell_d >= 0',
     'ell_d = 1 - sum_{j<d} ell[j] >= 0, the label bits left for the top join (SubsetSum, while LWE gives `ell_d`)',
     'list sizes', 'model'),
    ('size of X >= 0', 'ell[j] and ell_d (label units), part sizes', None, 'domain'),
    ('weight of X <= its size', 'the syndrome part: its error weight is at most its size', None, 'domain'),
    ('type of X: share of digit X >= 0',
     'the type of the window (or of the tree part): 0 <= ω₁ <= κ, or every share of a free window type', None,
     'domain'),
    ('join i X: pair fractions >= 0',
     'every pair fraction (the share of a parent digit written as a pair of a class) is between 0 and 1, and they sum '
     'to 1 by construction', None, 'domain'),
    ('join i X: level type shares >= 0',
     "joint_types='maxent': the share of the digit 0 (the remainder of the level type) is >= 0", None, 'domain'),
    ('join i: joint type exists (scaling residual = 0)',
     "joint_types='maxent': a joint type with the two level types exists (the iterative scaling converges). The value "
     "is -max(0, residual - 1e-10), exactly 0 inside the domain, so that the optimizer's finite differences do not see "
     'the scaling noise', None, 'domain'),
    ('L_i >= 0', 'lists below the top are not empty', None, 'domain'),
]
BOUNDS = [
    ('ell[j] of LWE (labels unlimited below the top)', '0 .. UNLIMITED_LABELS', 'the model has no upper bound'),
    ('sigma (`LeeWeightFilter`)', '0 .. SIGMA_MAX', 'sigma >= 0 by its definition'),
    ('z[0] of an opposite-tilt filter with all non-zero digits alike', '0 .. PRICE_MAX',
     'swapping the two lists negates it'),
    ('digit prices z, z_y (`PriceFilter`)', '-PRICE_MAX .. PRICE_MAX',
     'at 40 bits a fraction theta is 0 or 1 up to 1e-12'),
    ('center size u of a filter (u free)', 'CENTER_MIN .. 1 - CENTER_MIN',
     'the offset of the closed form is solved from it'),
]


def blocks():
    rows = ['| constraint (value >= 0) | meaning | group | from |', '|---|---|---|---|']
    for name, formula, key, group in CONSTRAINTS:
        rows.append(f"| `{name}` | {formula} | {group} | {PAPER[key] if key else 'range of the quantity'} |")
    constants = {name: getattr(families, name) for name in ('SIGMA_MAX', 'PRICE_MAX', 'CENTER_MIN')}
    constants['UNLIMITED_LABELS'] = build.UNLIMITED_LABELS
    bounds = ['| variable | search box | note |', '|---|---|---|']
    for variable, box, note in BOUNDS:
        for name, value in constants.items():
            box = box.replace(name, f'{value:g}')
        bounds.append(f'| {variable} | {box} | {note} |')
    references = ['| name used in the code | paper |', '|---|---|']
    references += [f'| [{name}] | {where} |' for name, where in PAPER.items()]
    references += ['', '| key | work |', '|---|---|'] + [f'| [{key}] | {text} |' for key, text in LITERATURE.items()]
    return {'constraints': '\n'.join(rows), 'bounds': '\n'.join(bounds), 'references': '\n'.join(references)}


def render(text):
    for name, block in blocks().items():
        pattern = re.compile(f'(<!-- {name}:begin -->\n).*?(<!-- {name}:end -->)', re.S)
        if not pattern.search(text):
            raise SystemExit(f'RULES.md has no block {name}')
        text = pattern.sub(lambda m: m.group(1) + block + '\n' + m.group(2), text)
    return text


def pattern_of(name):
    """A constraint name of model.constraints as its pattern in the table above."""
    if name.startswith('7-dissection'):
        return name
    name = re.sub(r'(L|D|fixed)_\d+', r'\1_i', name)
    name = re.sub(r'^join \d+ [^:]+:', 'join i X:', name)
    name = re.sub(r'^join \d+:', 'join i:', name)
    name = re.sub(r'share of digit \S+', 'share of digit X', name)
    name = re.sub(r'^type of the [^:]+:', 'type of X:', name)
    return re.sub(r'^(size|weight) of .+? (>=|<=)', r'\1 of X \2', name)


if __name__ == '__main__':
    path = os.path.join(HERE, 'RULES.md')
    with open(path) as handle:
        text = handle.read()
    new = render(text)
    if '--check' in sys.argv:
        sys.exit(0 if new == text else 1)
    with open(path, 'w') as handle:
        handle.write(new)
    print('RULES.md updated' if new != text else 'RULES.md is up to date')
