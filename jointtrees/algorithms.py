"""The classical algorithms as Trees.  Each function returns a Tree, and the model and the optimizer are the same for
all.

    from jointtrees.algorithms import may_ozerov
    optimize(Decoding(q=2, rate=0.42), may_ozerov(2))

A new algorithm is a new function here (or in your own script): write down its joins.
"""
from .families import PriceFilter
from .tree import Join, Tree


def nearest_neighbour(exact=False):
    """The top join of the ISD trees with a syndrome filter: own prices for each list, free center.  exact=True: the
    mixed join (it also matches label units exactly)."""
    return Join(exact=exact, syndrome=PriceFilter(tilt='own'))


# ----------------------------------------------------------------------------------------- subset sum and LWE
def hgj(depth, **options):
    """Howgrave-Graham and Joux (Eurocrypt 2010): exact joins, the lists use the digits 0, 1."""
    return Tree.same_joins(depth, digits=(0, 1), **options)


def bcj(depth, **options):
    """Becker, Coron and Joux (Eurocrypt 2011): exact joins, the lists use the digits -1, 0, 1."""
    return Tree.same_joins(depth, digits=(-1, 0, 1), **options)


def bbss(depth, **options):
    """Bonnetain, Bricout, Schrottenloher and Shen (Asiacrypt 2020): exact joins, the lists use the digits -1, 0, 1,
    2."""
    return Tree.same_joins(depth, digits=(-1, 0, 1, 2), **options)


def meet_lwe(depth, digits=(-2, -1, 0, 1, 2), **options):
    """Meet-LWE (May, Crypto 2021): exact joins on the given digits."""
    return Tree.same_joins(depth, digits=digits, **options)


def filtered(depth, digits, parts=1, **options):
    """The filtered tree of the paper: every join 1..depth filters with one price per digit (opposite tilt, free
    center).  parts=2 partitions the positions [partitioning]."""
    return Tree.same_joins(depth, filter=PriceFilter(), digits=digits, parts=parts, **options)


# -------------------------------------------------------------------------------------------------- decoding
def bjmm(depth, **options):
    """Becker, Joux, May and Meurer [BJMM12]: exact joins, the top join checks the weight on the rest."""
    return Tree.same_joins(depth, **options)


def may_ozerov(depth, **options):
    """May and Ozerov [MO15]: exact joins below, the top join matches the rest by nearest neighbours."""
    return Tree.same_joins(depth, top=nearest_neighbour(), **options)


def mixed_top(depth, window=None, **options):
    """The decoders of the paper ([isd]): exact joins below, a mixed top join (exact on ell_d coordinates and a
    syndrome filter on the rest).  `window` adds a filter on the window at the top join."""
    return Tree.same_joins(depth, top=Join(filter=window, syndrome=PriceFilter(tilt='own')), **options)


def wagner(depth, **options):
    """Wagner's tree for many solutions: split joins only, and the runs follow from the size of the top list
    (accounting 'many-solutions')."""
    return Tree([Join()] + [Join(split=True) for _ in range(depth)], **{'accounting': 'many-solutions', **options})
