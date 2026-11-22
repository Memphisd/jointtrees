"""Filters.  Standard library only.  Paper locations in brackets, their numbers are in references.py.

A filter keeps a vector if it fits a random center: a random set of u n positions on which the vector's digits have
prescribed counts.  For a pair of digits (i, j) at the same position of the two vectors of a join, the most likely
share of such positions inside the center is

    theta_ij = sigmoid2(offset + a_i + b_j)                                                  [closed form]

with one price a_i per digit for the left list and b_j for the right list (in bits, and a digit with a high price
is frequent inside the center), and an offset that sets the center size u = sum_ij J_ij theta_ij.  The size is the
parameter (f['u'], or the size fixed in the filter), and the offset is solved from it.  How the parameters of a
filter give the prices is the filter's family (families.py).  The exponents below are the same for every family.
"""
from .entropy import h, sigmoid2


def center_offset(flt, f, J, a, b):
    """The offset that gives the center its size u: the parameter f['u'] if the size is free, otherwise the size
    fixed in the filter.  (For u = 1/2 with opposite prices and a symmetric J the offset is 0 [center 1/2], and the
    bisection finds it like any other.)"""
    u = f['u'] if flt.u == 'free' else flt.u

    def size(c):
        return sum(m * sigmoid2(c + a[i] + b[j]) for (i, j), m in J.items())
    lo, hi = -200.0, 200.0                           # u is increasing in the offset: bisection to machine precision
    for _ in range(200):
        lo, hi = ((lo + hi) / 2, hi) if size((lo + hi) / 2) < u else (lo, (lo + hi) / 2)
    return (lo + hi) / 2


def marginals(J):
    """Digit frequencies of the left and of the right vector of a joint type J = {(a, b): share}."""
    x, y = {}, {}
    for (a, b), m in J.items():
        x[a], y[b] = x.get(a, 0.0) + m, y.get(b, 0.0) + m
    return x, y


def filter_exponents(size, J, flt=None, ctx=None, f=None):
    """[filter exponents] on a part of `size` n positions with joint type J:

        u      = sum_ij J_ij theta_ij                                  center size
        alpha  = h(u) - sum_ij J_ij h(theta_ij)                        a consistent pair shares a center w.p. 2^-alpha n
        beta_x = h(u) - sum_i p_i h(X_i / p_i),  X_i = sum_j J_ij theta_ij  a left vector fits a center w.p. 2^-beta_x n
        beta_y   the same with the columns

    `flt` is the filter family (families.py), `ctx` its FilterContext, `f` its parameters.  flt None: no filter.

    >>> from .families import PriceFilter, FilterContext
    >>> J = {(0, 0): .4, (0, 1): .1, (1, 0): .1, (1, 1): .4}
    >>> e = filter_exponents(1., J, PriceFilter(u=.5), FilterContext((0, 1), {0: 0, 1: 1}, 0), {'z': {'1': 1.}})
    >>> round(e['u'], 12), round(e['alpha'], 6), round(e['beta_x'], 6)
    (0.5, 0.016341, 0.003208)
    """
    record = dict(size=size, kind='none' if flt is None else flt.name, second_list=None if flt is None else flt.tilt,
                  J=J, alpha=0.0, beta_x=0.0, beta_y=0.0, u=None, offset=None)
    if flt is None:
        return record
    a, b = flt.price_pair(ctx, f)
    c = center_offset(flt, f, J, a, b)
    theta = {(i, j): sigmoid2(c + a[i] + b[j]) for (i, j) in J}
    u = sum(m * theta[p] for p, m in J.items())
    inside_x, inside_y = {}, {}
    for (i, j), m in J.items():
        inside_x[i], inside_y[j] = inside_x.get(i, 0.0) + m * theta[i, j], inside_y.get(j, 0.0) + m * theta[i, j]
    px, py = marginals(J)

    def beta(p, inside):
        return h(u) - sum(p[i] * h(inside[i] / p[i]) for i in p if p[i] > 0)
    record.update(theta=theta, offset=c, u=u, inside_x=inside_x, inside_y=inside_y,
                  alpha=size * (h(u) - sum(m * h(theta[p]) for p, m in J.items())),
                  beta_x=size * beta(px, inside_x), beta_y=size * beta(py, inside_y))
    return record
