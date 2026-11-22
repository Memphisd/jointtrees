"""The Lee metric over Z_q.  Standard library only.

The Lee weight of a digit a is min(a, q - a).  A random vector of relative Lee weight T has, asymptotically, the
maximum-entropy type with that mean Lee weight: the Boltzmann distribution P(a) ~ 2^(-beta wt(a)) ([BKW22],
Lemma 11).  That type is the type of the error of a Lee-metric decoding problem.

    boltzmann_type(T, q)        the type {digit: share} with mean Lee weight T
    lee_gv(rate, q)             the Lee weight T of the Gilbert-Varshamov bound: H(type) = (1 - rate) log2 q
    lee_weight(a, q)
"""
import functools
import math

from .entropy import H


def lee_weight(a, q):
    return min(a % q, (-a) % q)


def boltzmann_type(T, q, cutoff=None):
    """{digit: share}, the maximum-entropy type on Z_q (digits of Lee weight <= cutoff, if given) with mean Lee
    weight T per position."""
    digits = [a for a in range(q) if cutoff is None or lee_weight(a, q) <= cutoff]
    weights = [lee_weight(a, q) for a in digits]
    lo, hi = min(weights), max(weights)
    if T <= lo + 1e-12:
        return {a: (1.0 if w == lo else 0.0) / sum(1 for w in weights if w == lo) for a, w in zip(digits, weights)}
    if T >= hi - 1e-12:
        return {a: (1.0 if w == hi else 0.0) / sum(1 for w in weights if w == hi) for a, w in zip(digits, weights)}

    def shares(beta):
        top = max(-beta * w for w in weights)
        shift = top if top > 700 else 0.0                    # exp overflows beyond 709
        z = [math.exp(-beta * w - shift) for w in weights]
        total = sum(z)
        return [x / total for x in z]

    a, b = -80.0, 80.0                                       # the mean is decreasing in beta
    for _ in range(300):
        c = (a + b) / 2
        mean = sum(w * s for w, s in zip(weights, shares(c)))
        a, b = (c, b) if mean > T else (a, c)
    return dict(zip(digits, shares((a + b) / 2)))


@functools.lru_cache(maxsize=None)
def lee_gv(rate, q):
    """The relative Lee weight T at the Gilbert-Varshamov bound: the entropy of the Boltzmann type is
    (1 - rate) log2 q."""
    target = (1 - rate) * math.log2(q)
    lo, hi = 0.0, (q - 1) / 2 if q % 2 else q / 2
    for _ in range(300):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if H(boltzmann_type(mid, q)) < target else (lo, mid)
    return (lo + hi) / 2
