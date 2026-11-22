"""Small LWE keys by Meet-LWE.  Run:  python3 examples/lwe.py      (about 30 minutes)"""
from jointtrees import LWE, filtered, optimize
from jointtrees.algorithms import meet_lwe

ternary   = LWE(key={-1: 1/4, 0: 1/2, 1: 1/4})                          # ternary secret of weight n/2
kyber     = LWE(key={-2: 1/16, -1: 4/16, 0: 6/16, 1: 4/16, 2: 1/16})    # centered binomial B(2): Kyber768 / Kyber1024

optimize(ternary, meet_lwe(3), starts=4)                                # 0.3425
optimize(ternary, meet_lwe(5), starts=3)                                # 0.3193  (published: 0.3218)
optimize(ternary, filtered(depth=4, digits=(-2, -1, 0, 1, 2)), starts=2)  # with filters
# deep trees need many starts and a warm start from a shallower tree.  The paper's witnesses are in the
# paper's repository.
