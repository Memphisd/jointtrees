"""Ternary decoding with large weight (Wave).  Run:  python3 examples/ternary_large_weight.py      (under a minute)

Above the weight 2/3 the tree works on e - (1,...,1) (rule 4), which needs one price per digit: nonzero='free'."""
from jointtrees import Decoding, Join, Tree, optimize
from jointtrees.algorithms import wagner

hardest = Decoding(q=3, rate=0.3690702464, weight=1.0)   # e in F_3^n of weight n (no zeros), rate 1 - log_3(2)
wave    = Decoding(q=3, rate=0.5, weight=0.894)          # a Wave instance: many solutions

# one split join, then two representation joins
two_levels = Tree([Join(), Join(split=True), Join(), Join()], accounting='many-solutions', nonzero='free')
optimize(hardest, two_levels)                   # 0.2304   (published, without cancellations: 0.243)
optimize(wave, wagner(6, nonzero='free'))       # 0.01515 here, the paper's witness 0.01506 (published: 0.0151)
