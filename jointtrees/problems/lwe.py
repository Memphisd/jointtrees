"""Small LWE keys (Meet-LWE).  Standard library only."""
from .base import Problem


class LWE(Problem):
    """Small LWE keys: find the secret s with digit frequencies `key`, e.g. {-1: 1/4, 0: 1/2, 1: 1/4} (ternary of
    weight n/2), from LWE samples, as in Meet-LWE.

    The tree enumerates all n coordinates of s.  The joins below the top match as many guessed coordinates as they
    need: the labels are unlimited there, each ell_i is searched in [0, 1], and a result at this bound is reported.
    The top join matches `ell_d` label bits (ℓ^(d), default 1: one bit per coordinate).
    """
    arguments = ('key', 'ell_d')
    ell_max = None

    def __init__(self, key, ell_d=1):
        key = {int(a): float(v) for a, v in key.items()}
        if any(v < 0 for v in key.values()) or abs(sum(key.values()) - 1) > 1e-9:
            raise ValueError(f'key: the frequencies must be >= 0 and sum to 1 (they sum to {sum(key.values()):.6g})')
        if not ell_d > 0:
            raise ValueError('ell_d: the label bits matched by the top join (ℓ^(d)), > 0')
        self.key, self.ell_d = key, ell_d
        self.solution = key
