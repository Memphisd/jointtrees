"""Decoding random linear codes over F_q.  Run:  python3 examples/decoding.py      (about 1 minute)

The code (the problem) stays, and only the Tree (the algorithm) changes."""
import os

from jointtrees import Decoding, Join, PriceFilter, Tree, optimize
from jointtrees.algorithms import bjmm, may_ozerov, mixed_top

binary = Decoding(q=2, rate=0.42, weight='gv')      # e in F_2^n of Gilbert-Varshamov weight with H e = s

optimize(binary, bjmm(3))                           # 0.101832   exact joins, the top join checks the weight
optimize(binary, may_ozerov(2))                     # 0.096646   top join by nearest neighbours

# filters on the window do not help binary decoding: the same value as may_ozerov(2)
optimize(binary, Tree([Join(), Join(filter=PriceFilter()),
                       Join(exact=False, filter=PriceFilter(), syndrome=PriceFilter(tilt='own'))]))

ternary = Decoding(q=3, rate=0.466, weight='gv/2')  # e in F_3^n at half the Gilbert-Varshamov distance
optimize(ternary, may_ozerov(2))                    # 0.072906 bits = 0.0460 in base 3

# the paper's decoders ([isd]): the mixed top join (exact on some syndrome coordinates, nearest neighbours on the rest)
# at the worst rate 0.494 of F_11 at half distance, started from the stored witness: 0.141947 ([decoders table])
stored = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'witnesses', 'q11_d1_mixed_half.json')
optimize(Decoding(q=11, rate=0.494, weight='gv/2'), mixed_top(1), starts=2, warm=stored)
