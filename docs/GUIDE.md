# jointtrees: guide

Time exponents of representation trees for subset sum, LWE with small keys, decoding over F_q, ternary decoding with
large weight, restricted errors and the Lee metric, with filtered joins and partitioning. A problem is one class, an
algorithm is one `Tree` written as its joins, and `optimize` returns the time exponent with every intermediate
quantity. The parameters carry the names of the paper (`ell[i]`, `J[i][a][b]`, `p[i]`, `filter[i].u`, …), and a
result is stored as a witness file that `verify` checks without the optimizer.

Code for "Representations as Joint Types: Filtered Joins for Subset Sum, LWE and Decoding".

## Install and run

```sh
git clone https://github.com/Memphisd/jointtrees.git && cd jointtrees
python3 -m venv .venv && . .venv/bin/activate              # a fresh Python environment (Python 3.9 or later)
pip install .                                             # jointtrees, numpy and scipy
python3 examples/decoding.py                              # BJMM, May-Ozerov, the mixed top join
python3 -m jointtrees verify tests/witnesses              # check stored results (2 minutes)
python3 -m jointtrees evaluate tests/witnesses/subsetsum_pm3_d6_two.json    # the report of the tree of [main theorem]
python3 tests/test_library.py                             # tests without the optimizer (3 minutes)
python3 tests/test_optimizer.py                           # tests of the optimizer (5 minutes)
```

Without installing: `PYTHONPATH=. python3 examples/decoding.py` in an environment that has numpy and scipy.

Places in the paper are written as names in brackets, e.g. [main theorem]. `jointtrees/references.py` and the table
at the end of RULES.md give their numbers in the paper.

## A first example

```python
from jointtrees import Decoding, Tree, Join, PriceFilter, optimize

code = Decoding(q=2, rate=0.42, weight='gv')        # find e in F_2^n of Gilbert-Varshamov weight with H e = s

# May-Ozerov: join 0 and join 1 are exact joins, the top join matches the rest of the syndrome by nearest
# neighbours (a syndrome filter with its own prices for each list)
may_ozerov = Tree([Join(), Join(), Join(exact=False, syndrome=PriceFilter(tilt='own'))])

result = optimize(code, may_ozerov)                 # time 2^(0.096646 n), prints the report
P = result.parameters                               # the free parameters, named as in the paper
P.ell[0], P.omega_1, P.J[2][0][1], P.syndrome_filter.u
result.evaluation.L, result.evaluation.p[1]         # what follows from them: list sizes, level types, ...
result.save('may_ozerov.json')                      # python3 -m jointtrees verify may_ozerov.json
```

`examples/parameters.py` shows reading, fixing, editing and storing the parameters in one minute.

## The three things you write

**1. The problem** (`jointtrees/problems/`, one class each). Its arguments are the parameters of the problem and
nothing else.

| problem | the problem |
|---|---|
| `SubsetSum(weight=1/2)` | e ∈ {0,1}^n with weight·n ones and ⟨a, e⟩ = t mod 2^n. The labels are the n bits of t (ℓ_max = 1): the joins below the top match ℓ^(0), …, ℓ^(d−1) of them, and ℓ^(d) is what they leave |
| `LWE(key, ell_d=1)` | the secret of an LWE instance with digit frequencies `key`, e.g. `{-1: 1/4, 0: 1/2, 1: 1/4}`. The top join matches the given ℓ^(d) = `ell_d` label bits, and the joins below match as many as they need (no bound) |
| `Decoding(q, rate, weight='gv')` | syndrome decoding over F_q in the Hamming metric. `weight` is a number, `'gv'` or `'gv/2'`, kept as given, and `.omega` is the weight ω as a number |
| `RestrictedDecoding(q, rate, E, weight)` | the non-zero entries of the error lie in E, e.g. `E=(1, 156)` = {±1} over F_157 |
| `LeeDecoding(q, rate, weight='gv')` | Lee metric over Z_q, any q ≥ 2. `weight` is the relative Lee weight δ: a number, `'gv'` or `'gv/2'`. `.delta` is δ as a number, and `.omega` the share of the non-zero digits |
| `TypedDecoding(q, rate, error)` | decoding with an error of a given type `{digit: share}` (q prime), the base of the two above |

`problem.replace(rate=0.43)` gives the same problem with one argument changed (rate sweeps, memory curves).

**2. The tree** (`jointtrees/tree.py`): its joins 0, 1, …, d from the bottom to the top. Join 0 merges two halves
into a base list, join d is the top join. The **depth** d is the number of joins above join 0, so a tree of depth d
has d + 1 joins (`Tree.same_joins(depth=3)` has the joins 0, 1, 2, 3). Each join is a `Join` with these fields:

| field | meaning |
|---|---|
| `exact=True` | matches label units exactly (bits for subset sum and LWE, syndrome coordinates for decoding, how many is optimized). `False` only together with a syndrome filter |
| `filter=None` | a filter on the positions the tree enumerates (for decoding: the window filter) |
| `syndrome=None` | decoding, top join only: a filter on the syndrome coordinates that the window leaves, matched by distance (nearest neighbours) |
| `split=False` | a split join: the children live on disjoint halves, no representations |
| `dissection=False` | join 0 only: base lists by the 7-dissection |

The joins of the paper:

```python
Join()                                                       # exact join
Join(filter=PriceFilter())                                   # filtered join (one price per digit, opposite tilt,
                                                             # free center size)
Join(exact=False, syndrome=PriceFilter(tilt='own'))          # nearest-neighbour join
Join(syndrome=PriceFilter(tilt='own'))                       # mixed join: exact and nearest neighbours (top join only)
Join(filter=PriceFilter(), syndrome=PriceFilter(tilt='own')) # the same with a window filter (top join only)
Join(split=True)                                             # split join
Join(dissection=True)                                        # join 0 by the 7-dissection
```

A combination the paper has no join for is rejected with a message (RULES.md, section "Joins"). The rest of a
`Tree` holds what is the same for all joins:

| field | values | meaning |
|---|---|---|
| `digits` | `None`, a tuple | the digits the **lists** may use, which change the result (subset sum: `(0, 1)`, `(-1, 0, 1)`, `(-1, 0, 1, 2)`). `None`: the digits of the solution (decoding: of the error and the sums of two of them) |
| `accounting` | `'budget'`, `'repetitions'`, `'many-solutions'` | one representation of the solution must survive, or subtrees are repeated when representations are missing [EZ23], or (many solutions) the runs follow from the size of the top list, which a deficit of representations lowers |
| `parts` | 1, 2, … | partitioning: the positions of the tree cut into this many parts, each with its own joint types and filters |
| `nonzero` | `None`, `'alike'`, `'free'` | `Decoding` only: all non-zero digits alike (default), or full joint types and one price per digit (needs a prime q, and is needed above the weight (q−1)/q) |
| `joint_types` | `'free'`, `'maxent'` | the joint types are parameters, or each join uses the joint type of maximum entropy with the level types (then the level types are the parameters, with fewer variables for large alphabets) |
| `types` | `'free'`, `'boltzmann'` | `LeeDecoding` with `joint_types='maxent'`: the level types are Boltzmann types (one relative Lee weight each) |

Shortcuts: `Tree.same_joins(depth, filter=None, top=None, **fields)` gives join 0 exact and the joins 1..depth of
the same form, each with its own parameters: a filter of the family `filter` at each of them (None: exact joins), and
`top` in place of the top join if given. `jointtrees.algorithms` has the classical trees: `hgj`, `bcj`, `bbss`,
`meet_lwe`, `filtered`, `bjmm`, `may_ozerov`, `mixed_top`, `wagner`. The paper's filtered tree can also be imported
from `jointtrees` directly: `optimize(SubsetSum(), filtered(depth=4, digits=(-1, 0, 1, 2)))`.

**3. The filters** (`jointtrees/families.py`). A filter family says how the parameters of a filter give one price
per digit for each of the two lists. The exponents of a filter follow from the same closed form for every family.

| family | prices | parameters | used for |
|---|---|---|---|
| `PriceFilter(tilt='opposite')` | one price z_a per digit, the second list −z | `z`, `u` | the joins of the paper |
| `PriceFilter(tilt='same')` | one price per digit, the same on both lists | `z`, `u` | near neighbours |
| `PriceFilter(tilt='own')` | own prices for each list (the general filter) | `z`, `z_y`, `u` | syndrome filters |
| `LeeWeightFilter()` | −σ times the Lee weight, the same on both lists | `sigma`, `u` | the Lee syndrome filter ([lee metric]) |

Every family takes `u='free'` (the default: the center size is optimized) or a number that fixes it. The center
size u is the share of the positions of the filtered part (the positions the tree enumerates at that join, or the
syndrome coordinates of a syndrome filter) that a center contains.

## The parameters, named as in the paper

`result.parameters` (and `Witness.load(path).parameters`) are the **free** parameters of the tree, as plain Python
data. `ell` is the list of ℓ^(0), …, ℓ^(d−1). A quantity per join or level (`J`, `p`, `lee_weight`, `filter`) is a
dict keyed by the index i with the free entries only, so that `P.J[2]` is J^(2). Types and prices are dicts over
digits. A filter, the syndrome filter of the top join and a part are records whose entries are also attributes.

| paper | name | where it is free | meaning |
|---|---|---|---|
| ℓ^(i), i < d | `ell[i]` | always | label bits matched exactly at join i (bits per n) |
| ℓ^(d) | `ell_d` | decoding with an exact or mixed top join (otherwise `P.ell_d` is None, and the value is derived or given) | label bits matched exactly at the top join (bits per n) |
| J^(i) | `J[i][a][b]` | `joint_types='free'`, representation joins 1..d | joint type of the join into level i: share of the positions where the left child has digit a and the right child digit b |
| p^(i) | `p[i][a]` | p^(d) for decoding with free digits (the type of e₁ on the window), and p^(i), i < d, with `joint_types='maxent'` | type of level i |
| | `lee_weight[i]` | `types='boltzmann'` | p^(i) is the Boltzmann type of this relative Lee weight (the mean Lee weight per position, between 0 and ⌊q/2⌋) |
| ω₁ | `omega_1` | decoding with the non-zero digits alike | weight of the error on the window (fraction of n) |
| u, z, z^y | `filter[i].u`, `.z[a]`, `.z_y[a]`, `.sigma` | the filtered joins | the filter of join i (decoding: the window filter) |
| | `syndrome_filter` | a top join with a syndrome filter | the syndrome filter of the top join: `.u`, `.z[a]`, `.z_y[a]`, `.sigma` |
| ε | `parts[k].size` | k < P | partitioning: share of the positions of the tree part in part k (decoding: of the window). `parts` is a dict with the keys 1..P |
| | `parts[k].J`, `.p`, `.filter`, `.lee_weight` | partitioning | the same per part. ℓ, ω₁, the syndrome filter and p^(d) are shared |

Every value of the model, free or not, is on the evaluation (`result.evaluation`, see below). The `derived` block of
a witness file holds the values that follow from the free parameters, the problem and the tree and are not free:
ℓ^(d) where it is not free, the level types p^(i) that are not free, κ = R + ℓ (decoding: the size of the window,
ℓ = Σ ℓ^(i)/log₂ q), ω₁ with free digits, the size of the last part, a center size that the tree fixes, and the prices
of a family with other parameters. `print(result.parameters)` writes one quantity per line.
`parameter_table(problem, tree)` lists every name with its range and meaning before anything is optimized (free,
derived, or given: fixed by the problem or the tree), and `parameter_names(problem, tree)` returns the free ones as a
list.

Prices: only differences of prices matter, so the reference digit (0, or the smallest digit if 0 is not a digit of
the lists) has price 0 and is not listed. With all non-zero digits alike every filter has the one price `z[0]` of the
digit 0, and the other digits have price 0.

Two conventions of the paper hold here too. With a digit symmetry (the Lee classes {a, −a} of a syndrome filter, or the
classes of Z_q of a syndrome filter when the lists use a subset of Z_q) a price is given for one digit of each class and
holds for the whole class. A window filter has one price per digit. With rule 4 (large weight, RULES.md) the types from the shifted join up are those of e − (1, …, 1).

**Fixing and bounding.** `optimize(problem, tree, fixed={...}, bounds={...})` takes the names as strings, written
as after `P.`:

```python
optimize(problem, tree, fixed={'filter[3].u': 0.1, 'ell[0]': 0.2})          # free values: held
optimize(problem, tree, fixed={'J[2][0][0]': 0.6, 'p[1][0]': 0.8})          # derived values: constraints
optimize(problem, tree, bounds={'ell[0]': (0.2, None), 'parts[2].p[3][0]': (None, 0.5)})
optimize(problem, tree, fixed={'ell': P.ell, 'parts[1]': P.parts[1]})        # a whole group: every free value in it
```

A free value is held exactly. A derived value becomes a constraint, which the solver meets to about 10⁻⁹. A group
holds the free values it gives. A whole joint type (`{'J[2]': M}`) holds all its entries, and so the type of its
level too. A name that names nothing, a share outside [0, 1], or a value that the problem or the tree fixes (such as
ℓ^(d) at a nearest-neighbour top join) is an error before the search starts. The fixed values and bounds are stored
in the witness, and `polish` keeps them. ℓ^(d) is named `ell_d` here too, for example `bounds={'ell_d': (None, 0.3)}`.

## The evaluation

`evaluate(problem, tree, parameters, memory_bound=None)` (or `result.evaluation`, `Witness.evaluate()`) returns every
quantity of the model, under the paper's names. The parameters must give every free parameter of the tree and
nothing else, and a joint type must sum to the type of its level (otherwise a ValueError names what is wrong).
Parameters that violate a constraint of the model are evaluated where the formulas allow it, and `feasible` is then
False. Parameters for which a formula has no value (a negative share in a level type, a joint type that cannot
exist) are a ValueError that names the level. `result.parameters` holds the free values and `result.evaluation`
every value of the model, so a free quantity such as J^(i) is in both under the same name. In the parameters a
quantity per join or level is a dict of its free entries (`P.J[2]`), in the evaluation a list over i = 0..d
(`result.evaluation.J[2]`, None where a join has no joint type).


| name | meaning |
|---|---|
| `time`, `memory` | the exponents in bits: time 2^(time·n), and `time = tau + runs` |
| `tau` | τ = max_i (T^(i) + Δ^(i)), the cost of one run of the tree |
| `runs` | log₂ of the runs of the whole tree (decoding: the permutations) |
| `F` | the time in base q (decoding over F_q: q^(F n) = 2^(time·n)), which the paper's tables give |
| `L[i]`, `D[i]`, `Delta[i]` | list sizes, entropies of the tree part, repetitions per level |
| `T[i]`, `r[i]`, `c[i]` | per join: cost (without Δ), representations, consistency loss |
| `ell[i]`, `ell_d` | label bits matched exactly at join i < d, and at the top join |
| `alpha[i]`, `beta_x[i]`, `beta_y[i]` | filter exponents of join i (summed over the filtered parts) |
| `p[i]`, `J[i]` | level types and joint types (with partitioning: `parts[k].p`, `parts[k].J`) |
| `kappa`, `omega_1` | decoding: size of the window and weight of the error on it |
| `parts[k]` | partitioning: `size`, `p`, `J`, `D`, `r`, `c`, `alpha`, `beta_x`, `beta_y` of part k |
| `constraints`, `terms`, `budget` | every constraint's slack, every cost term, the representation budget |
| `feasible`, `violations` | whether every constraint holds (to 10⁻⁶), and the ones that do not |
| `memories` | every stored list (join 0's halves or its 7-dissection, L^(0) .. L^(d−1)). `memory` is the largest |

In the library, `memory` is always an exponent (`result.memory`, `evaluation.memory`) and `memory_bound` the bound M of
a search (`optimize(..., memory_bound=M)`, `witness.memory_bound`).

## Running the optimizer

```python
result = optimize(problem, tree, memory_bound=None, starts=8, seed=1, warm=None, fixed=None, bounds=None,
                  time_limit=None, callback=None, quiet=False, no_surplus=False)
```

- `memory_bound`: a memory bound M (memory 2^(Mn)).
- `starts`: random starts and perturbed copies of the best point, after the warm starts. `warm`: a `Result`, a
  `Witness`, the path of a witness file, parameters, or a list of them. A start from another tree goes through
  `starts.embed` (below): what matches by name is used and every new feature starts switched off, so the start keeps
  the time of the smaller tree. A start that gives none of the tree's free parameters is an error, and
  `result.starts` says how many each warm start gave.
- `time_limit`: seconds of wall-clock time. The best point found so far is returned and `result.cut` is True.
- `callback`: called after every start with its report.
- `no_surplus`: the search aid budget ≤ 0 (helps random starts of subset-sum trees).
- `result.time`, `.memory`, `.parameters`, `.evaluation`, `.starts` (per start: where it started, its time, its
  status `regular`, `not converged`, `cut` or `infeasible`, its seconds), `.cut`, `.warnings` (variables at a bound
  that only limits the search: the optimum may lie beyond it), `.save(path, **notes)`, `.witness(**notes)`.
- `polish(result, gain=1e-6, rounds=4)` restarts from the optimum until a round gains less than `gain` (with the
  result's fixed values and bounds). `result` may also be a `Witness` or the path of a witness file.
- `NoFeasiblePoint` is the error when no start reaches a feasible point. `problem.replace(rate=0.43)` and
  `tree.replace(parts=2)` change one field of a problem or a tree.
- `SearchSpace(problem, tree, memory_bound, fixed, bounds)` is the optimizer's view: the free parameters as a vector
  (`names`, `low`, `high`, `vector(start)`, `parameters(x)`, `evaluate(x)`, `constraints(x)`, `random(rng)`,
  `result(x)` to save a point as a witness), for searches with an objective of one's own.
- The optimizer is a multi-start local search on a non-convex problem with a fixed `seed`. Its result is an upper
  bound on the optimum of the model. More `starts` or a `warm` start can improve it. `optimize` raises
  `NoFeasiblePoint` if no start reaches a point that satisfies the constraints.
- Run times: a one-part tree of depth 3 to 5 with 4 to 8 starts takes seconds to a few minutes. Partitioning, large
  digit sets, maximum-entropy joint types and memory bounds take up to ten times longer.

## Starts

`jointtrees.starts` makes a start for one tree from the optimum of another:

| function | the new tree | the start |
|---|---|---|
| `embed(result, problem, tree)` | any tree that shares names with the old one | what matches is kept. A new filter starts with prices 0 and center size 1/2 (it filters nothing), a new ℓ^(d) at 0. A filter whose family or tilt changed starts with the old filter's prices |
| `deeper(result, problem, tree, new='balanced')` | one join more at the bottom | a new lowest join that writes every digit as its simplest pair (`'balanced'`) or copies the old lowest join (`'copy'`). The old lowest join's label bits split between the two |
| `into_parts(result, parts=2, sizes=None)` | partitioning | the same parameters in every part (same time) |
| `with_digits(result, problem, tree)` | more digits | the new digits at share 0 (same time), their prices on a straight line through the outer ones |

## Partitioning

`Tree(..., parts=2)` cuts the positions the tree enumerates into two parts (subset sum and LWE: all positions,
decoding: the window). Each part has on the top level the type of the tree part, and its own joint types and
filter parameters at every join, and the label bits are shared. It works for every problem, because the problem only
says which positions the tree enumerates. With the same parameters in both parts the exponent equals the
unpartitioned one (`into_parts`). With the 7-dissection at join 0, the entropy D_0 of the level-0 lists is the sum
over the parts, each weighted by its size, and the cost and memory of the dissection follow from it.

At depth 4 with the digits −1..2, the two-part optimum is not found in reasonable time from random starts, nor from
the one-part optimum copied into both parts (`warm=into_parts(one)`), which keeps the one-part exponent. The example
therefore starts from the stored witness of that optimum, the end point of a longer search, and optimizes from
there. The tree of [main theorem] is evaluated at the parameters of its witness.

```python
from jointtrees import SubsetSum, Witness, evaluate, filtered, optimize
two = optimize(SubsetSum(), filtered(depth=4, digits=(-1, 0, 1, 2), parts=2),
               warm='tests/witnesses/subsetsum_d4_two.json', starts=0)  # 0.2707 (3 minutes)
main_theorem = filtered(depth=6, digits=range(-3, 4), parts=2)          # the tree of [main theorem]
stored = Witness.load('tests/witnesses/subsetsum_pm3_d6_two.json')      # its witness
evaluate(SubsetSum(), main_theorem, stored.parameters).time             # 0.261965
```

## Reference values

What the examples and tests give (bits, rounded up). The paper's tables and theorems state the rest.

| problem | tree | exponent |
|---|---|---|
| subset sum | `bcj(3)` (digits −1, 0, 1, exact joins) | 0.2892 |
| subset sum | `bbss(3)` (digits −1, 0, 1, 2) | 0.2829 |
| subset sum | `filtered(depth=3, digits=(-1, 0, 1))` | 0.2844 |
| subset sum | `filtered(depth=4, digits=(-1, 0, 1, 2))`, one part / two parts | 0.2726 / 0.2707 |
| subset sum | digits −3..3, depth 6, two parts ([main theorem]) | 0.2620 |
| decoding F_2, R = 0.42, GV weight | `bjmm(3)` / `may_ozerov(2)` | 0.1019 / 0.0967 |
| decoding F_3, R = 0.466, half GV weight | `may_ozerov(2)` | 0.0730 bits, F = 0.0460 |
| decoding F_11, R = 0.494 (its worst rate), half GV weight | `mixed_top(1)` | 0.141947 bits |
| Lee metric Z_4, R = 0.46, half GV weight | depth 3, Boltzmann types, mixed top join with `LeeWeightFilter()` | 0.0950 bits |
| ternary, weight 1, R = 1 − log₃ 2 | split join + 2 representation joins, many solutions | 0.2304 |
| LWE ternary key of weight n/2 | `meet_lwe(5)` | 0.3193 |

## Witness files

A witness file holds the problem, the tree, the memory bound, the claimed exponents and the parameters
(`jointtrees witness 3`):

```json
{
 "format": "jointtrees witness 3",
 "library": "jointtrees 0.1",
 "problem": {"name": "Decoding", "q": 11, "rate": 0.494, "weight": "gv/2"},
 "tree": {"joins": [{}, {"syndrome": {"family": "prices", "tilt": "own", "u": "free"}}], "digits": [0, 1, …, 10], "nonzero": "alike"},
 "memory_bound": null,
 "claimed": {"time": 0.14194701408502405, "memory": 0.049672704898325266, "F": 0.041031888972830256},
 "parameters": {
  "ell": [0.05301013511068437],
  "ell_d": 0.019912604053006665,
  "omega_1": 0.012463616234346976,
  "J": {"1": {"both zero": 0.97070…, "one zero": 0.00109…, "both nonzero, sum zero": 0.00051…, "both nonzero, sum nonzero": 0.0000251…}},
  "syndrome_filter": {"u": 0.0184987…, "z": {"0": 4.1163…}, "z_y": {"0": 4.1163…}}
 },
 "derived": {"kappa": 0.5150793…, "p": {"0": {"0": 0.98166…, "1": 0.00183…, …}, "1": {"0": 0.97580…, "1": 0.00241…, …}}},
 "notes": {"note": "Decoding over F_11 at rate 0.494, an error of half the Gilbert-Varshamov weight. Tree: depth 1, …"}
}
```

- `parameters` are the free parameters of the section above. `derived` holds the values that follow from them and
  are not free, for the reader. No free value is repeated there, and a value that the parts share (the top type
  p^(d)) is written once, at the top level. When a file is read, the parameters are used and every value that
  follows from them (the sums of the joint types, the remainders of the types, the derived block) is checked against
  the model. A difference above 10⁻⁹ is an error.
- `ell` is the list ℓ^(0), …, ℓ^(d−1). `ell_d` is ℓ^(d), under `parameters` where it is free and under `derived`
  otherwise.
- A quantity per join or level (`J`, `p`, `lee_weight`, `filter`) is an object keyed by the paper's index
  (`"J": {"1": …, "2": …}` holds J^(1) and J^(2)). The syndrome filter of the top join is one object
  (`"syndrome_filter": {"u": …, "sigma": …}`). There is no null in `parameters` and `derived`.
- `fixed` and `bounds` are written only if given: what `optimize` held or bounded, which `polish` uses again. `notes`
  is written only if given and holds free-form notes. The witnesses of this repository have one note each, which
  says what the tree is.
- A joint type is a matrix whose rows (the left child's digit a) and columns (the right child's digit b) are the
  tree's `digits` in increasing order. Its entries are shares of the positions of the tree part (decoding: of the
  window). With the non-zero digits alike over F_q, q ≥ 3, it is given by the paper's cases ("both zero", "one zero",
  "both nonzero, sum zero", "both nonzero, sum nonzero"), each value the entry J_ab of every pair (a, b) of that case.
  A tree with `joint_types='maxent'` writes no joint types (they follow from the level types).
- Prices: the reference digit has price 0 and is not listed (see above), also under `derived`. With a digit
  symmetry (syndrome filters only) the price of one digit of each class holds for the class.

```python
w = Witness.load(path)                  # format 3
w.problem, w.tree, w.memory_bound, w.parameters, w.claimed, w.fixed, w.bounds, w.notes
w.evaluate(), w.check(), w.report(), w.save(path)
Witness(problem, tree, None, parameters).save(path)   # claimed is computed from the parameters
```

```sh
python3 -m jointtrees verify FILES_OR_FOLDERS        # standard library only
python3 -m jointtrees evaluate FILE                  # the report
```

`verify` (`--verbose`: every check of every file) re-evaluates a witness with the standard library only, checks
every constraint and every derived value, re-derives the filter exponents independently (α by iterative proportional
fitting, β by exact counting) and, for subset sum and LWE trees, the whole tree by a second implementation
(`crosscheck.py`) that reads the joint types, prices and label bits straight from the file.

Witnesses that name your own filter families or problems need them imported first:
`python3 -m jointtrees verify --import my_extensions.py FILE.json`.

The witnesses and the scripts behind the paper's numbers are in the paper's repository.

## Reading the report

Every exponent is per n, in bits. A set of coordinates is a fraction of n, and a label unit (a syndrome coordinate)
is log₂ q bits, printed in the header.

| column | meaning |
|---|---|
| `D_i` | entropy of the tree part on level i (summed over the parts): there are 2^(D_i n) vectors of its type |
| `L_i` | size of a list of level i < d. `L_d`: the top list before the final weight check on the syndrome part |
| runs of the tree | how often the whole tree is run (decoding: the permutations of the coordinates), added to every cost term |
| `log Q` | a candidate passes the final weight check on the syndrome part with probability Q (decoding) |
| `Delta_i` | repetitions of the joins of level i and above |
| `r_i` | representations gained at join i |
| `c_i` | consistency loss of every part split by a joint type at join i |
| `c_label` | loss from the target condition at join i: exact label bits |
| `ell_i` | label bits matched exactly at join i (ℓ^(d), `ell_d`, in the row of the top join) |
| `alpha`, `beta_x`, `beta_y` | filter exponents of join i, summed over its filtered parts. `u` is a filter's center size |

Cost terms: *input* reads a list (L). *Label matches* are the pairs that agree on the ell_i label bits (2L − ell).
*Routing* is L + α − β for each list. *Candidates* are the pairs that share a center and agree on the labels
(2L − ell + α − β_x − β_y). *Output L_i* writes the list. The *dissection* terms are those of the 7-dissection.
Below the cost terms the report lists the constraints of the model by name with their slack. A slack above −10⁻⁶ is
zero up to rounding (the tolerance of `verify`). A filter whose center count α is 0 is marked inactive: it filters
nothing, and its parameters do not matter.

## Extending the library

**A new filter family**: one class in `jointtrees/families.py` or in your own file. It declares its variables with
their ranges, gives the prices from them, and a random start. Its variables keep their names everywhere (in
`result.parameters`, `fixed=` and the witness files), and the optimizer, `verify`, the report and the witness files
then work without any other change (`tests/test_extension.py` shows a complete example).

```python
@dataclass(frozen=True)
class MyFilter(Filter):
    """What it is, with the paper location."""
    name = 'my-filter'                             # its name in witness files
    tilt = 'opposite'                              # how the second list's prices relate: 'opposite', 'same', 'own'
    u: object = 'free'                             # the center size: 'free' or a number (the base class checks it)

    def variables(self, ctx):                      # [Variable(path in the filter's dict, low, high, search sides)]
        return [Variable(('slope',), 0.0, 10.0, ('high',))]

    def prices(self, ctx, p):                      # (z, z_y): {digit: price} for both lists, on ctx.digits
        a = {j: p['slope'] * abs(ctx.value(j)) for j in ctx.digits}
        return a, {j: -a[j] for j in a}

    def random_start(self, ctx, rng, shared):      # {path: start value}
        return {('slope',): rng.uniform(0.1, 1.0)}
```

**A new problem**: a subclass of `Problem` (`problems/base.py`) or of `Decoding` in a new file of
`jointtrees/problems/`, imported in `problems/__init__.py` (or in your own file, loaded with `--import` on the
command line). Its constructor takes the problem's parameters (listed in `arguments`, which is what a witness file
stores) and sets `solution` (the type of the solution, {digit: share}), `unit` (bits per label unit), and
`log_solutions` (log₂ of the expected number of solutions per n, used only with `accounting='many-solutions'`). It
overrides what differs among `check`, `rules`, `variables`, `paper_name`, `parts`, `constraints` and `runs`
(`problems/base.py` documents each). `paper_name` gives the paper's name of each of its own variables, e.g. ω₁ for
the window weight of `Decoding`, and the class attribute `meanings` describes them for `parameter_table`.
Partitioning, filters and all joins then work for it. The second implementation of `verify` (`crosscheck.py`)
covers only `SubsetSum` and `LWE`. For a new problem `verify` says "not covered" and checks everything else.

**A new algorithm**: a function that returns a `Tree` (see `jointtrees/algorithms.py`).

**A new strategy**: a script on top of `optimize`, `polish`, `jointtrees.starts` and `SearchSpace`.

**A new kind of join or part** (rare). A new kind of part is a `Part` subclass in `parts.py` whose `share` says what
it contributes at each join, in the terms of [lemma 1]. A new kind of join is what its parts contribute, plus its
own cost terms. The places to change:

| place | what |
|---|---|
| `parts.py` | the `Part` subclass and its `share` |
| `problems/` | the problem's `parts` and `variables` that create it (and `paper_name` for its variables) |
| `build.variables` | joint-type variables (`fractions`) and filter variables of the new part, if it has them |
| `build.filter_of` | which filter sees the part at which join (keep it in step with the filtered parts in `build.variables`) |
| `model.py` | `list_sizes` counts the label bits of the exact parts, and `constraints` the weight of the syndrome part. A join with other costs needs its terms in `evaluate` |
| `tree.py` | the field of `Join` and its checks (RULES.md, section "Joins") |
| `parameters.py` | `part_place` and `filter_place`: the paper's name of the new part (and `parts.label` for the report) |
| `crosscheck.layout` | reject the new feature, so that the second implementation does not check a tree it does not know |

## How it fits together

```
problem + Tree ──build.check──build.build──▶ Instance        the rules of RULES.md resolved once
parameters (paper names) ──parameters.to_model──▶ the model's dict (pair fractions, label units)
                                  │
                   model.evaluate(Instance, model dict) ──▶ Evaluation
                   [lemma 1] per part and join → list sizes → repetitions → permutations
                   → join costs (filters.py, families.py) → constraints     (one function per paper step)
                                  │
     report.py (text)   verify.py (independent re-computation)   optimizer.py (SciPy, over the model's variables)
     witness.py (files: parameters.from_model gives the paper names back)
```

1. **One evaluation for every problem.** `model.evaluate` has no case for subset sum, LWE or decoding. What differs
   is in the problem classes: their rules, their parts, the runs of the whole tree and their own constraints.
2. **Parts.** The solution and its label are cut into parts, and every quantity of a join is the sum over the parts
   of a local contribution (`parts.py`). Partitioning, the window and the syndrome part all use this.
3. **Equations where the paper has them.** `model.py` is a pipeline of named steps in the order of the paper, each
   with its paper location in brackets (`references.py` holds the equation numbers, and `RULES.md` writes the
   equations out).
4. **The paper's names at the edge.** Users, files and `fixed=` see the paper's names (`parameters.py`). The model
   and the optimizer work on their own dict, whose free numbers keep the sums of every joint type right.
5. **Constraints come from one function** (`model.constraints`), and the optimizer, the report and `verify` read it.
6. **Standard library for everything that proves a number.** Only `optimizer.py` imports numpy and scipy.

## Files

| file | content |
|---|---|
| `jointtrees/tree.py` | `Tree` and `Join`: the algorithm |
| `jointtrees/families.py` | the filter families and their base class |
| `jointtrees/problems/` | one class per problem |
| `jointtrees/algorithms.py` | the classical algorithms as Trees |
| `jointtrees/parameters.py` | the parameters under the paper's names: conversion, checks, `parameter_table` |
| `jointtrees/starts.py` | starts for one tree from the optimum of another |
| `jointtrees/build.py` | input checks, the rules, the variables of a tree |
| `jointtrees/parts.py` | the kinds of parts, what each contributes at a join, partitioning |
| `jointtrees/jointtypes.py`, `filters.py`, `entropy.py`, `lee.py` | joint types and [lemma 1], the closed form of the filter exponents, entropies, Lee weights |
| `jointtrees/model.py` | the evaluation step by step, and `constraints`, which creates every constraint |
| `jointtrees/optimizer.py` | `optimize`, `polish`, `SearchSpace` (numpy, scipy) |
| `jointtrees/verify.py`, `crosscheck.py` | independent checks of a witness |
| `jointtrees/report.py`, `witness.py`, `__main__.py` | report, witness files and their JSON form, command line |
| `jointtrees/references.py` | the only place with the paper's equation numbers |
| `RULES.md` | what the library derives by itself: rules, joins, filters, partitioning, the model in equations, parameters, constraints, bounds |
| `examples/` | subset sum, LWE, decoding, Lee, ternary large weight, trade-offs, partitioning, parameters |
| `tests/` | `test_library.py` (standard library), `test_optimizer.py`, `test_extension.py`. `tests/witnesses/`: one witness per problem and feature |

Comparison-only computations of the paper (Hirose's center count, classical Stern, three-set filters) are not in
the library.
