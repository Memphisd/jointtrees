"""Verification of a witness.  Standard library only (no optimizer needed).

Independent of the formulas in model.py and filters.py:
  * alpha of every filter: the minimum over all admissible fractions [admissible] for the witness's center digit
    counts, by iterative proportional fitting of the maximum-entropy coupling Q(i, j, b) (b = inside the center):
    alpha = H(J) + h(u) - H(Q).  Must equal the closed form [closed form] to 1e-6 per n.  The scaling loses precision
    for large alphabets with large prices, so the best-converged state is kept, and a slow scaling runs longer.
  * beta_x, beta_y of every filter: exact counting with log-gamma multinomials at n = 1e12.  Must equal [filter
    exponents] to 1e-6 per n.
  * trees without a rate (subset sum, LWE): the whole tree by a second implementation (crosscheck.py), to 1e-9 (for
    trees deeper than 20 levels 2^d 1e-15: the list recursion doubles rounding errors at every level).
Shared with the model (regression): the time against the claimed time (1e-9), and every constraint of
model.constraints (>= -1e-6): the model constraints are listed by name, the domain constraints as one line.  Every
value of the parameters that follows from the free ones (level types, joint types, ℓ^(d), ...) is checked against
the model.  A variable on a pure search-box bound of the optimizer is reported with the level 'bound' (the optimum
may lie outside).
"""
import math

from .crosscheck import sum_tree
from .entropy import H, h
from .filters import marginals
from .model import evaluate
from .parameters import at_search_bounds
from .parts import label
from .report import constraints_to_show, slack
from .witness import Witness


def max_entropy_alpha(J, inside_x, inside_y, iterations=20000):
    """(alpha, error) of a filter by iterative proportional fitting of the maximum-entropy coupling, and the largest
    deviation of its (i, j) marginals from J."""
    px, py = marginals(J)
    u = sum(inside_x.values())
    Q = {(i, j, b): m * (u if b else 1 - u) for (i, j), m in J.items() for b in (0, 1)}
    targets = ({(i, b): inside_x[i] if b else px[i] - inside_x[i] for i in px for b in (0, 1)},
               {(j, b): inside_y[j] if b else py[j] - inside_y[j] for j in py for b in (0, 1)})
    error, best = 1.0, None
    for _ in range(iterations):
        for (i, j), m in J.items():                                  # fit the (i, j) marginal
            total = Q[i, j, 0] + Q[i, j, 1]
            if total > 0:
                Q[i, j, 0], Q[i, j, 1] = Q[i, j, 0] * m / total, Q[i, j, 1] * m / total
        for side, target in enumerate(targets):                      # fit the (i, b) and (j, b) marginals
            sums = {}
            for (i, j, b), v in Q.items():
                key = (i if side == 0 else j, b)
                sums[key] = sums.get(key, 0.0) + v
            for (i, j, b) in Q:
                key = (i if side == 0 else j, b)
                if sums[key] > 0:
                    Q[i, j, b] *= target[key] / sums[key]
        error = max(abs(Q[i, j, 0] + Q[i, j, 1] - m) for (i, j), m in J.items())
        # keep the best state: with many digits and large prices the scaling can lose precision and drift.  Stop after
        # 2000 rounds without improvement, unless far from converged (over F_157 the error stalls)
        if best is None or error < best[0]:
            best = (error, H(J) + h(u) - H(Q), 0)
        else:
            best = (best[0], best[1], best[2] + 1)
            if best[2] > 2000 and best[0] < 1e-12:
                break
        if error < 1e-15:
            break
    return best[1], best[0]


def log2_multinomial(counts):
    """log2 of the multinomial coefficient of the counts (log-gamma, so the counts may be large and not integers)."""
    return (math.lgamma(sum(counts) + 1) - sum(math.lgamma(c + 1) for c in counts)) / math.log(2)


def counted_beta(p, inside, u, n=1e12):                             # n = 1e9 is off by 1.7e-6 for 157 digits
    """beta of one list by exact counting at n positions."""
    digits = [d for d in p if p[d] > 0]
    inside_count = log2_multinomial([inside[d] * n for d in digits])
    outside_count = log2_multinomial([(p[d] - inside[d]) * n for d in digits])
    return -(inside_count + outside_count - log2_multinomial([p[d] * n for d in digits])) / n


def check_filter(i, s, name=None):
    """The checks of one filter record of the evaluation: alpha by iterative proportional fitting, beta by counting."""
    size = s['size']
    where = (f'join {i} {name or s["part"]} ({s["kind"]} filter, {s["second_list"]} prices for the second list, '
             f'{size:.4f} n)')
    if size <= 0:
        return [('ok', f'{where}: empty part')]
    alpha, u = s['alpha'] / size, s['u']
    ipf, error = max_entropy_alpha(s['J'], s['inside_x'], s['inside_y'])
    if error > 1e-9:                    # large prices on a small part: the scaling is slow, so it goes on longer
        ipf, error = max_entropy_alpha(s['J'], s['inside_x'], s['inside_y'], iterations=200000)
    # the comparisons are per n, like every exponent the witness claims (a part of size s contributes s times its
    # value per position)
    out = [] if 0 < u < 1 else [('FAIL', f'{where}: center size u = {u} not in (0, 1)')]
    if error * size > 1e-6:
        out.append(('FAIL', f'{where}: IPF did not converge ({error:.1e} per position)'))
    out.append(('ok' if abs(alpha - ipf) * size <= 1e-6 else 'FAIL',
                f'{where}: alpha {s["alpha"]:.9f} per n = {alpha:.9f} per position of the part, IPF minimum {ipf:.9f} '
                f'(diff {(alpha - ipf) * size:+.1e} per n)'))
    px, py = marginals(s['J'])
    for side, p, inside in (('x', px, s['inside_x']), ('y', py, s['inside_y'])):
        beta, counted = s['beta_' + side] / size, counted_beta(p, inside, u)
        out.append(('ok' if abs(beta - counted) * size <= 1e-6 else 'FAIL',
                    f'{where}: beta_{side} {s["beta_" + side]:.9f} per n = {beta:.9f} per position, exact count '
                    f'{counted:.9f}'))
    if s['alpha'] < max(s['beta_x'], s['beta_y']) - 1e-12:
        out.append(('FAIL', f'{where}: alpha < beta'))
    return out


def verify(witness):
    """(passed, [(level, message)]) for a Witness or the data of a witness file.  The level is 'ok', 'warning',
    'bound' (a variable on a search bound) or 'FAIL'."""
    try:
        witness = witness if isinstance(witness, Witness) else Witness.from_dict(witness)
        inst = witness.instance
        prm = witness.model_parameters()
        ev = evaluate(inst, prm)
    except Exception as error:
        return False, [('FAIL', f'evaluation failed: {error!r}')]
    out = []
    bad = witness.check()
    if bad:
        more = f' (and {len(bad) - 4} more)' if len(bad) > 4 else ''
        out.append(('FAIL', 'parameters: ' + '.  '.join(f'{name} {text}' for name, text in bad[:4]) + more))
    else:
        out.append(('ok', 'parameters: the joint types, types and derived values of the file agree with the model'))
    try:
        second = sum_tree(witness.to_dict())
    except Exception as error:
        return False, out + [('FAIL', f'second implementation failed: {error!r}')]
    if second is None:
        out.append(('ok', 'independent tree (crosscheck.py): not covered by the second implementation (it covers '
                          'subset sum and LWE trees with free joint types and filters PriceFilter(opposite))'))
    else:
        tau, lists, deltas = second
        gaps = ([abs(tau - ev.time)] + [abs(a - b['L']) for a, b in zip(lists, ev.levels)]
                + [abs(a - b['delta']) for a, b in zip(deltas, ev.levels)])
        tolerance = max(1e-9, 2 ** inst.depth * 1e-15)      # the list recursion doubles rounding errors per level
        out.append(('ok' if max(gaps) <= tolerance else 'FAIL',
                    f'independent tree (crosscheck.py): tau {tau:.12f}, agrees to {max(gaps):.1e}'))
    out += claims(witness.claimed, ev)
    for name, c in constraints_to_show([c for c in ev.records if c.group == 'model' or c.value < -1e-6]):
        reference = f'   ({c.reference})' if c.reference else ''
        out.append(('ok' if c.value >= -1e-6 else 'FAIL', f'{name}: slack {c.value:+.2e}{reference}'))
    domain = [c for c in ev.records if c.group == 'domain']
    if domain:
        worst = min(domain, key=slack)
        out.append(('ok' if worst.value >= -1e-6 else 'FAIL',
                    f'{len(domain)} domain constraints (tightest: {worst.name}, {worst.value:+.1e})'))
    for i, join in enumerate(ev.joins):
        for s in join['filters']:
            if s['kind'] != 'none':
                out += check_filter(i, s, label(inst, s['part']))
    out += [('bound', text) for text in at_search_bounds(inst, prm)]
    return all(level != 'FAIL' for level, _ in out), out


def claims(claimed, ev):
    """The checks of the claimed exponents against the evaluation (1e-9)."""
    if not claimed or claimed.get('time') is None:
        return [('warning', f'time {ev.time:.12f}, but the file claims no time, so none was compared')]
    out = [('ok' if abs(ev.time - claimed['time']) <= 1e-9 else 'FAIL',
            f"time {ev.time:.12f}, claimed {claimed['time']:.12f}")]
    if 'F' in claimed:
        out.append(('ok' if abs(ev.F - claimed['F']) <= 1e-9 else 'FAIL',
                    f"time in base q: F = {ev.F:.12f}, claimed {claimed['F']:.12f}"))
    if claimed.get('memory') is not None:
        out.append(('ok' if abs(ev.memory - claimed['memory']) <= 1e-9 else 'FAIL',
                    f"memory {ev.memory:.12f}, claimed {claimed['memory']:.12f}"))
    return out
