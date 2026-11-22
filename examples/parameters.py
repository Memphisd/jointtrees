"""The parameters under the paper's names: read them, fix them, start from them, store them.
Run:  python3 examples/parameters.py      (about 1 minute)"""
from jointtrees import SubsetSum, Witness, evaluate, filtered, optimize, parameter_table

problem = SubsetSum()
tree = filtered(depth=3, digits=(-1, 0, 1))
print(parameter_table(problem, tree))               # every name, free or derived, its range and its meaning

result = optimize(problem, tree, starts=4, quiet=True)
P = result.parameters                               # the free parameters
print(P)                                            # one quantity per line
print(P.ell[0], P.J[2][-1][1], P.filter[3].u, P.filter[3].z[1])   # ℓ^(0), J^(2)_{-1,1}, u and z_1 of join 3
ev = result.evaluation                              # everything that follows from them
print(ev.p[1], ev.L, ev.T, ev.tau, ev.time)         # p^(1), L^(i), T^(i), τ, time

# fix a value while the rest is optimized: a free value is held, a derived one becomes a constraint
held = optimize(problem, tree, starts=0, warm=result, fixed={'filter[3].u': 0.25, 'p[1][0]': 0.7}, quiet=True)
print(held.time, held.parameters.filter[3].u, held.evaluation.p[1][0])

# edit the parameters and evaluate them
P2 = held.parameters
P2.filter[3].u = 0.3
print(evaluate(problem, tree, P2).time)

# a witness file: the parameters under "parameters", what follows from them under "derived"
result.save('filtered_d3.json')
w = Witness.load('filtered_d3.json')                # python3 -m jointtrees verify filtered_d3.json
print(w.parameters.ell, w.evaluate().time)
