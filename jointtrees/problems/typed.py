"""Syndrome decoding with an error of a given type.  Standard library only."""
import math

from ..entropy import H
from .decoding import Decoding, is_prime


class TypedDecoding(Decoding):
    """Syndrome decoding of a random code of rate R over Z_q, q prime, with an error of a given type:
    `error` = {digit: share}, the share of the positions with each digit.

    The base of RestrictedDecoding and LeeDecoding, and usable directly for any other error type.  Against Decoding:
    the non-zero digits are not alike, so the window carries a free type (one share per non-zero digit) and the
    permutations count types instead of weights (rule 2), and there is no large-weight rule.  `omega` is ω, the
    share of the non-zero digits of the error.
    """
    arguments = ('q', 'rate', 'error')
    ring = False                    # True: any q >= 2 (only the addition mod q is used, e.g. the Lee metric over Z_q)

    def __init__(self, q, rate, error):
        if not (isinstance(q, int) and q >= 2 and (self.ring or is_prime(q))):
            raise ValueError(f'q = {q} is not a prime: an error of a given type over F_q needs a prime q (LeeDecoding '
                             'works over Z_q for any q >= 2)')
        if not 0 < rate < 1:
            raise ValueError('rate: between 0 and 1')
        error = {int(a): float(share) for a, share in error.items()}
        if any(not 0 <= a < q for a in error):
            raise ValueError(f'error: the digits are 0, ..., {q - 1}')
        if any(share < 0 for share in error.values()) or abs(sum(error.values()) - 1) > 1e-9:
            raise ValueError(f'error: the shares must be >= 0 and sum to 1 (they sum to {sum(error.values()):.6g})')
        self.q, self.rate, self.unit = q, rate, math.log2(q)
        self.error = self.solution = dict(sorted(error.items()))     # the type of the error e
        self.omega = 1.0 - self.solution.get(0, 0.0)                  # ω, the weight: the share of non-zero digits
        self.log_solutions = H(self.solution) - (1 - rate) * math.log2(q)

    def check(self, tree, digits):
        q = self.q
        if any(not 0 <= a < q for a in digits) or 0 not in digits:
            raise ValueError(f'Tree.digits: a subset of 0, ..., {q - 1} that contains 0')
        missing = sorted(a for a in self.solution if a not in digits)
        if missing:
            raise ValueError(f'Tree.digits: the digits {missing} of the error are not digits of the lists')
        if tree.nonzero == 'alike':
            raise ValueError("nonzero='alike' needs an error whose non-zero digits are equally frequent (Decoding)")

    def digit_rules(self, inst):
        """Rule 3 for an error of a given type: full joint types, free window type, free prices."""
        error = ', '.join(f'{a}: {v:.4f}' for a, v in inst.root_type.items() if v > 0)
        return [f'rule 3 (digits): the error has the type {error}.  Free window type, full joint types, free prices, '
                'and the permutations count types']
