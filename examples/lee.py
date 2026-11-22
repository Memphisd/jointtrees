"""Decoding in the Lee metric [lee table].  Run:  python3 examples/lee.py      (about 20 minutes)

The Lee metric lives on Z_q for any q >= 2.  The tree of the paper: exact joins with Boltzmann level types, and a
mixed top join with the Lee syndrome filter (price -sigma * Lee weight on both lists)."""
from jointtrees import Join, LeeDecoding, LeeWeightFilter, Tree, optimize

code = LeeDecoding(q=4, rate=0.46, weight='gv/2')
tree = Tree([Join(), Join(), Join(), Join(syndrome=LeeWeightFilter())], joint_types='maxent', types='boltzmann')
optimize(code, tree, starts=8)                  # 0.094901 (the same as own free prices on both lists)
