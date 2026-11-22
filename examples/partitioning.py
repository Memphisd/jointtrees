"""Partitioning the positions into parts [partitioning].  Run:  python3 examples/partitioning.py   (about 12 minutes)

Tree(parts=2) cuts the positions the tree enumerates into two parts, fixed at random and the same for every list.
Each part has its own joint types and filters at every join, and the label bits are shared.  It works for every
problem: the problem only says which positions the tree enumerates.  The one-part optimum copied into both parts
(`into_parts`) has the same exponent.  At depth 4 with the digits -1..2 the two-part optimum is not found in
reasonable time from that copy or from random starts, so the two-part run starts from the stored witness of that
optimum, the end point of a longer search."""
import os

from jointtrees import SubsetSum, evaluate, filtered, optimize
from jointtrees.starts import into_parts

problem = SubsetSum()
one = optimize(problem, filtered(depth=4, digits=(-1, 0, 1, 2)), starts=6, quiet=True)
print(one.time)                                     # 0.2726
tree = filtered(depth=4, digits=(-1, 0, 1, 2), parts=2)
copied = evaluate(problem, tree, into_parts(one))   # the optimum in both parts
print(copied.time)                                  # the same exponent

stored = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'witnesses', 'subsetsum_d4_two.json')
two = optimize(problem, tree, warm=stored, starts=0, quiet=True)
print(two.time)                                     # 0.2707 ([alphabets table])
print(two.parameters.parts[1].size)                 # ε, the share of the positions in part 1
two.save('partitioned_d4.json')                     # python3 -m jointtrees verify partitioned_d4.json
