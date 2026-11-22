"""Restricted syndrome decoding.  Standard library only."""
from .typed import TypedDecoding


class RestrictedDecoding(TypedDecoding):
    """Restricted syndrome decoding [BBPSWW23]: the non-zero entries of e lie in a subgroup E of F_q^* (here: a set
    of non-zero digits), each element of E equally often, and e has weight W n.

    E.g. q = 157, E = (1, 156) = {+-1}.  Everything else is TypedDecoding.  The lists use by default the digits 0,
    E and the sums of two elements of E (E0 + E+ in the notation of [BBPSWW23]), and Tree(digits=...) chooses others.
    """
    arguments = ('q', 'rate', 'E', 'weight')

    def __init__(self, q, rate, E, weight):
        E = tuple(sorted({int(e) % q for e in E}))
        if not E or 0 in E:
            raise ValueError('E: non-zero digits of F_q')
        if not 0 < weight <= 1:
            raise ValueError('weight: the share of non-zero entries of the error, in (0, 1]')
        error = {0: 1 - weight}
        for e in E:
            error[e] = weight / len(E)
        super().__init__(q, rate, error)
        self.E, self.weight = E, weight
