"""Human-readable report of an evaluation: configuration, applied rules, levels, joins, filters, cost terms,
constraints, legend.  Every exponent is per n, in bits, and a set of coordinates is a fraction of n."""
import math
import re

from .parts import label
from .references import ref

LEGEND = f"""legend (exponents per n in bits, sets of coordinates as fractions of n, a label unit is log2 q bits)
  D_i       entropy of the tree part on level i: there are 2^(D_i n) vectors of its type
  L_i       size of a list of level i < d.  L_d: the top list, 2^(L_d n) candidates per run that pass every check
            except the final weight check on the syndrome part, apart from the planted solution.  A negative L_d means
            fewer than one false candidate per run
  log Q     a candidate passes the final weight check on the syndrome part with probability Q (decoding problems only)
  type      the type p^(i) of a level (with rule 4, from the shifted join up: of e - (1,...,1), as in the paper)
  Delta_i   repetitions of the joins of level i and above (deficit of representations)
  r_i       representations gained at join i
  c_i       consistency loss: every part split by a joint type at join i
  c_label   loss from the target condition at join i: exact label bits (ell_i)
  ell_i     label bits matched exactly at join i (the parameters ell[i], and ell_d in the row of the top join)
  alpha, beta_x, beta_y   filter exponents of join i, summed over its filtered parts (one product center per part).
            u: the center size of a filter
  cost terms   input: read a list (L).  label matches: pairs that agree on the ell_i label bits (2L - ell).
            routing (left/right list): L + alpha - beta.  candidates: pairs that share a center and agree on the
            labels (2L - ell + alpha - beta_x - beta_y).  output L_i: write the list.  dissection: the 7-dissection
            ({ref('dissection')})"""


def round_up(value, digits=4):
    """Exponents are upper bounds: shown values are rounded up."""
    return math.ceil(value * 10 ** digits - 1e-9) / 10 ** digits


def show_type(t):
    return '(' + ', '.join(f'{digit}: {share:.4f}' for digit, share in sorted(t.items()) if share > 5e-5) + ')'


def filter_text(inst, record):
    """'window: prices, u 0.5000' or 'syndrome part: near-neighbour filter, u 0.5100'."""
    if record['alpha'] < 1e-9:
        off = ' (inactive: alpha = 0)'
    elif record['alpha'] < 1e-4:
        off = f" (nearly inactive: alpha = {record['alpha']:.1e})"
    else:
        off = ''
    if record['part'] != 'rest':
        return f"{label(inst, record['part'])}: {record['kind']}, u {record['u']:.4f}{off}"
    agree = sum(m for (a, b), m in record['J'].items() if a == b)       # share of positions where the labels agree
    kind = 'near' if agree > 1 / inst.q else 'far'
    return f"{label(inst, record['part'])}: {kind}-neighbour filter, u {record['u']:.4f}{off}"


def grouped(records):
    """The constraints in groups whose names differ only in a number (one per digit, one per level): [(name, [c])]."""
    groups = []
    for c in records:
        pattern = re.sub(r'(?<=_)-?\d+|(?<=digit )-?\d+|(?<=join )\d+|(?<=level )\d+', 'X', c.name)
        if groups and groups[-1][0] == pattern:
            groups[-1][1].append(c)
        else:
            groups.append((pattern, [c]))
    return groups


def constraints_to_show(records, tolerance=1e-6):
    """[(name, constraint)] of the given constraints, grouped by their names.  A group of up to three is shown in
    full.  A larger group shows its violated constraints, or else its tightest, under a name that counts the group."""
    out = []
    for pattern, group in grouped(records):
        if len(group) <= 3:
            out += [(c.name, c) for c in group]
            continue
        violated = [c for c in group if c.value < -tolerance]
        for c in violated or [min(group, key=slack)]:
            out.append((f'{pattern} ({len(group)} constraints), tightest {c.name}', c))
    return out


def slack(constraint):
    return constraint.value


def constraint_lines(records, tolerance=1e-6):
    """The model constraints by name with their slack (more than three of one kind: one line with the tightest), the
    domain constraints as one line."""
    lines = ['constraints of the model (slack >= 0.  A slack above -1e-6 is zero up to rounding, the tolerance of '
             'verify)']
    for name, c in constraints_to_show([c for c in records if c.group == 'model'], tolerance):
        flag = 'VIOLATED' if c.value < -tolerance else ''
        lines.append(f'  {flag:8s} {name:52s} {c.value:+.3e}   {c.reference}')
    domain = [c for c in records if c.group == 'domain']
    if domain:
        worst = min(domain, key=slack)
        lines.append(f'  domain constraints (sizes, weights, fractions, list sizes >= 0, weight <= size): '
                     f'{len(domain)}, tightest: {worst.name} ({worst.value:+.2e})')
        lines += [f'  VIOLATED {c.name} by {-c.value:.3e}' for c in domain if c.value < -tolerance]
    return lines


def report(inst, ev, claimed=None, legend=False):
    q, d = inst.q or len(inst.digits), inst.depth
    lines = []
    in_base_q = inst.rate is not None and q > 2

    def base_q(v):
        """The exponent in base q, for the decoding problems over F_q, q > 2."""
        if not in_base_q:
            return ''
        return f'   = {v / math.log2(q):.6f} in base {q} (rounded up: {round_up(v / math.log2(q)):.4f})'
    bound = 'none' if inst.memory_bound is None else f'M = {inst.memory_bound:g}'
    lines += ['configuration (defaults filled in)', '  ' + repr(inst.problem), '  ' + repr(inst.tree),
              f'  memory bound {bound}, exponents per n in bits, a label unit is {inst.unit:.4f} bit'
              + ('s' if inst.unit != 1 else '')]
    lines += ['applied rules'] + ['  ' + rule for rule in inst.rules]
    lines.append(f'time    2^({ev.time:.9f} n)   rounded up: {round_up(ev.time):.4f}' + base_q(ev.time))
    lines.append(f'memory  2^({ev.memory:.9f} n)   rounded up: {round_up(ev.memory):.4f}' + base_q(ev.memory))
    runs = f'runs of the tree 2^({ev.runs:.6f} n)'
    top_list = f'L_{d} = {ev.levels[d]["L"]:+.6f}'
    if inst.accounting == 'many-solutions':
        runs += (f' = max(0, -L_{d} - log Q + min(0, log N)) with {top_list}, log Q = {-ev.final_check:+.6f}, '
                 f'log N = {inst.log_solutions:+.6f}')
    elif inst.rate is not None:
        runs += f' (permutations), top list {top_list}, log Q = {-ev.final_check:+.6f}'
    else:
        runs += f', top list {top_list}'
    lines.append(runs + f'   representation budget {ev.budget:+.6f}')
    if claimed is not None:
        lines.append(f'claimed time {claimed:.9f}, difference {ev.time - claimed:+.1e}')
    parts = list(ev.levels[0].get('types', {'tree': None}))
    if len(parts) > 1:
        whole = sum(ev.part_sizes.values())
        lines.append('parts   ' + ', '.join(f'{label(inst, name)}: {size / whole:.4f} of the positions'
                                            for name, size in ev.part_sizes.items()))
    if ev.shift is not None:
        lines.append(f'rule 4: from join {ev.shift[0]} up the types are those of e - (1,...,1)')
    lines.append('\nlevel      D_i      L_i  Delta_i   type p^(i)' + (' of each part' if len(parts) > 1 else ''))
    for i in reversed(range(len(ev.levels))):
        v = ev.levels[i]
        row = f"{i:4d}  {v['D']:8.5f} {v['L']:8.5f} {v['delta']:8.5f}   "
        if len(parts) == 1:
            lines.append(row + show_type(ev.paper_type(i, v['type'])))
        else:
            lines.append(row + '   '.join(f'{label(inst, name)} {show_type(ev.paper_type(i, t))}'
                                         for name, t in v['types'].items()))
    lines.append('\njoin      r_i      c_i  c_label    ell_i    alpha   beta_x   beta_y   filters')
    for i, j in enumerate(ev.joins):
        filters = ', '.join(filter_text(inst, x) for x in j['filters'] if x['kind'] != 'none') or 'none'
        numbers = '  '.join(f'{j[key]:7.5f}' for key in ('r', 'c', 'c_label', 'ell', 'alpha', 'beta_x', 'beta_y'))
        lines.append(f'{i:4d}  {numbers}   {filters}')
    top = max(ev.terms.values())
    lines.append('\ncost terms (repetitions and runs included), * = decides the time')
    lines += [f"  {'*' if v >= top - 1e-7 else ' '} {k:40s} {v + ev.runs:.9f}" for k, v in ev.terms.items()]
    lines += [''] + constraint_lines(ev.records)
    lines.append('the parameters with the names of the paper: print(result.parameters) or '
                 'print(Witness.load(file).parameters).  All names: parameter_table(problem, tree)')
    if legend:
        lines += ['', LEGEND]
    return '\n'.join(lines)
