"""Random subset sum.  Standard library only."""
from .base import Problem


class SubsetSum(Problem):
    """Random subset sum: find e in {0,1}^n with weight·n ones and <a, e> = t mod 2^n, a and t random.

    The tree enumerates all n coordinates, and the labels are the n bits of the target (ℓ_max = 1).  How many of them
    each join below the top matches is optimized, and ℓ^(d) is what they leave of the n bits.  Everything else is
    the base problem.  The digits of the lists belong to the algorithm: Tree(digits=(-1, 0, 1)) for the
    representations of Becker-Coron-Joux, (-1, 0, 1, 2) for those of Bonnetain-Bricout-Schrottenloher-Shen.  The
    default is the digits of the solution, 0 and 1.
    """
    arguments = ('weight',)

    def __init__(self, weight=0.5):
        if not 0 < weight < 1:
            raise ValueError('weight: the share of ones in the solution, between 0 and 1')
        self.weight = weight
        self.solution = {0: 1 - weight, 1: weight}
