"""Time-memory trade-offs: any tree, plus a memory bound.  Run:  python3 examples/tradeoffs.py   (about 3 minutes)"""
import os

from jointtrees import Decoding, Join, PriceFilter, SubsetSum, Tree, optimize

# subset sum: base lists by the 7-dissection, subtrees repeated when representations are missing
filtered = Tree([Join(dissection=True)] + [Join(filter=PriceFilter())] * 3, digits=(-1, 0, 1, 2),
                accounting='repetitions')
for M in (0.10, 0.15):                          # 0.5757, 0.4393
    optimize(SubsetSum(), filtered, memory_bound=M)

# decoding under a memory bound: base lists by the 7-dissection, a nearest-neighbour top join, subtrees repeated when
# representations are missing.  Random starts take long, and a warm start from the stored witness re-optimizes it
# quickly
here = os.path.dirname(os.path.abspath(__file__))
stored = os.path.join(here, '..', 'tests', 'witnesses', 'tree_nn_d2_full_M0.040.json')
code = Decoding(q=2, rate=0.442, weight='gv')
top = Join(exact=False, syndrome=PriceFilter(tilt='own'))
nearest = Tree([Join(dissection=True), Join(), top], accounting='repetitions')
optimize(code, nearest, memory_bound=0.04, starts=2, warm=stored)   # 0.1023
