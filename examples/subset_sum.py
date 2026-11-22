"""Random subset sum.  Run:  python3 examples/subset_sum.py      (under a minute)

The digits of the lists belong to the algorithm (Tree.digits), and the problem is the same for every tree."""
from jointtrees import SubsetSum, filtered, optimize
from jointtrees.algorithms import bbss, bcj

subset_sum = SubsetSum()        # e in {0,1}^n of weight n/2 with <a, e> = t mod 2^n

optimize(subset_sum, bcj(3), starts=4)              # 0.28915 (BCJ11: 0.291, which fixes ell_i = r_i)
optimize(subset_sum, bbss(3), starts=4)             # 0.2829  (BBSS20: 0.2830)

# the paper's filtered tree: every join filters with one price per digit (opposite tilt, free center)
tree = filtered(depth=3, digits=(-1, 0, 1))
optimize(subset_sum, tree, starts=4)                # 0.2844  ([alphabets table])

# the tree of [main theorem] (digits -3..3, depth 6, two parts) is in tests/witnesses/subsetsum_pm3_d6_two.json:
#   python3 -m jointtrees evaluate tests/witnesses/subsetsum_pm3_d6_two.json        # 0.261965
