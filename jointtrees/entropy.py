"""Entropies in bits per n.  Standard library only."""
import math


def H(p):
    """Entropy of a distribution (dict or list).  Zero entries contribute nothing.  H([.5, .5]) = 1."""
    return -sum(x * math.log2(x) for x in (p.values() if isinstance(p, dict) else p) if x > 0)


def h(x):
    """Binary entropy."""
    return H([x, 1 - x]) if 0 < x < 1 else 0.0


def weight_entropy(w, q):
    """log2 #(vectors of relative weight w in F_q^n) / n = h(w) + w log2(q - 1)."""
    return h(w) + w * math.log2(q - 1)


def gv(rate, q=2):
    """Gilbert-Varshamov weight: the w <= (q-1)/q with weight_entropy(w, q) = (1 - rate) log2 q (bisection)."""
    lo, hi = 0.0, (q - 1) / q
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if weight_entropy(mid, q) < (1 - rate) * math.log2(q) else (lo, mid)
    return (lo + hi) / 2


def sigmoid2(x):
    """1 / (1 + 2^-x)."""
    return 1 / (1 + 2 ** -x) if x >= 0 else 2 ** x / (1 + 2 ** x)
