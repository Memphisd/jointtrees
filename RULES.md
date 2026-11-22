# What the library derives by itself

A config is one problem (`SubsetSum`, `LWE`, `Decoding`, `RestrictedDecoding`, `LeeDecoding`, …) and one `Tree`
(the algorithm), written as its joins. Everything on this page is derived from them. Each rule is stated at the
class or field it belongs to (`jointtrees/problems/`, `jointtrees/tree.py`, `jointtrees/families.py`), applied by
the problem's `rules` or in `jointtrees/build.py`, and printed under "applied rules" at the top of every report.
Paper locations are written as names in brackets, e.g. [saturation]. The table at the end maps them to equation
numbers (`jointtrees/references.py` is the only place that holds the numbers).

| # | Rule | Applies when | Where |
|---|---|---|---|
| 1 | Base merge | always | `build.build`, `model.base_merge_cost` |
| 2 | Window (information set decoding) | decoding problems | `Decoding.parts`, `Decoding.runs` |
| 3 | Digit symmetry | `Decoding` (`Tree.nonzero`) | `Decoding.digit_rules`, `jointtypes.pair_classes_alike`, `families.Filter.price_pair` |
| 4 | Large weight | `Decoding` with ω > (q−1)/q | `Decoding.digit_rules` |
| 5 | Number of solutions | decoding problems | `Decoding.__init__`, `model.evaluate` |
| 6 | Near or far neighbours | a top join with a syndrome filter | `build.filter_of`, `families.PriceFilter` |

## Words

| word | meaning |
|---|---|
| label | the value ϕ(x) of a vector that the solution must hit: a subset sum, the syndrome H x, the product A s |
| label unit | one unit of the label per n: a bit for subset sum and LWE, a syndrome coordinate (log₂ q bits) for decoding |
| ℓ^(i) | label bits matched exactly at join i (`ell[i]` for i < d, `ell_d` for the top join). The ℓ^(i) below the top are free variables of the optimizer |
| part | a piece of the solution or of its label: the tree part (or its parts 1, 2, … with partitioning), the exact label units of a join, the syndrome part (`jointtrees/parts.py`) |
| digits | `Tree.digits` are the digits of the lists, and the problem gives the digits of the solution. `Tree.nonzero` says whether the non-zero digits of a decoding problem are alike or free |
| syndrome part | decoding: the syndrome coordinates that the window leaves (`rest` in the model). The top join checks the weight of the error on them, and matches them by distance if it has a syndrome filter |
| window | the coordinates of the error on which the tree runs: the k information coordinates and the exactly matched ones |
| joint type | J^(i)_ab: the share of the positions where the left child has digit a and the right child digit b (`J[i][a][b]`). The model stores it by its pair fractions: of the positions with parent digit s, the fraction written as a given pair of digits (a, b), a + b = s |
| price | filter parameter: one number per digit (in bits). A digit with a high price is frequent inside the center |
| tilt | how the prices of the second list follow from the first: opposite (b = −a), same (b = a), own (b free) |
| center, offset | the center is the set of positions a filter looks at. Its size u is free (the default) or fixed in the filter (`PriceFilter(u=0.5)`), and the offset of the closed form is solved from it |
| filter family | how the parameters of a filter give its prices (`families.py`): `PriceFilter`, `LeeWeightFilter` |
| saturation | a list cannot have more elements than there are vectors of its type: [saturation] |
| representation budget | representations of the solution in the whole tree minus the constraints the labels put on them, ≥ 0 with the accounting `budget`: [budget] |
| accounting | how missing or surplus representations are paid: `budget`, `repetitions` ([repetitions]) or `many-solutions` ([many solutions]) |
| product center | a filter applied on blocks of positions, one center per block: [filter blocks] (k = ⌈√n⌉ blocks, so the cost has no condition) |
| partitioning | the positions of the tree cut into parts, each with its own joint types and filters: [partitioning] |

## Joins

A tree of depth d has the joins 0, 1, …, d (`Tree(joins=[...])`, bottom to top). Each is a `Join(exact, filter,
syndrome, split, dissection)`:

| fields | paper name | allowed |
|---|---|---|
| `Join()` | exact join | every join |
| `Join(filter=F)` | filtered join | joins 1..d, not split |
| `Join(exact=False, syndrome=F)` | nearest-neighbour join (on the syndrome part) | decoding, top join only |
| `Join(syndrome=F)` (exact and by distance) | mixed join | decoding, top join only |
| `Join(exact=False, filter=F, syndrome=G)` | nearest-neighbour join with a window filter | decoding, top join only |
| `Join(filter=F, syndrome=G)` | mixed join with a window filter | decoding, top join only |
| `Join(split=True)` | split join | joins 1..d, with `syndrome` only at the top (Wave) |
| `Join(dissection=True)` | 7-dissection | join 0, exact, no syndrome filter |
| `Join(exact=False)` without `syndrome`, `split` with `filter`, `filter` at join 0, `syndrome` below the top join | none | rejected with a message |

An extension that needs another kind of join adds a kind of part (`parts.py`, a `Part` subclass with `share`), the
problem's `parts` and `variables` that create it, `build.filter_of` for its filter, and the field in `tree.py`.

## 1. Base merge

Every base list is the merge of two halves, and each half is a plain enumeration. This merge is **join 0**. It is not
counted in the depth: a tree of depth d has the joins 0, 1, …, d, and join d is the top join.

- Exact join 0: the two halves are matched on ℓ₀ label units.
- `Join(dissection=True)` at join 0: the 7-dissection [dissection]. Only with exact labels.
- In a tree of depth 0, join 0 is the top join and may have a syndrome filter.

Joins 1..d are representation joins, except split joins: there the two children live on disjoint halves of the
positions, and there are no representations and no consistency loss.

## 2. Window

The decoding problems (`Decoding`, `TypedDecoding`, `RestrictedDecoding`, `LeeDecoding`) are syndrome decoding of a
code of rate R over F_q (Z_q for the Lee metric).

- The labels are the (1 − R)n syndrome coordinates.
- Every exact join j matches ℓ^(j) label bits of them, ℓ^(j)/log₂ q coordinates (optimized, ℓ^(d) is `ell_d`). The
  tree runs on the **window** of κ = R + ℓ coordinates, ℓ = Σ_j ℓ^(j)/log₂ q.
- The error has weight ω₁·n on the window (optimized, and with `nonzero='free'` or an error of a given type a full
  type p^(d) on the window).
- What the window leaves is the **syndrome part** of the top join (`rest` in the model): (1 − κ)n coordinates with
  weight (ω − ω₁)n. A top join without a syndrome filter only checks the weight on it, and a top join with a syndrome
  filter filters on it.
- A nearest-neighbour join matches no coordinates exactly. The **mixed** top join (`Join(syndrome=F)`) matches
  ℓ^(d) bits exactly (their coordinates belong to the window) and filters on the syndrome part (the top join of the
  decoders of the paper).
- If the lists use only some digits of Z_q (restricted errors), the labels on the syndrome part are still uniform over
  all of Z_q: their joint type is J(a, b) = t(a − b)/q for a, b in Z_q (`jointtypes.field_channel`), and the syndrome
  filter has one price per class {a, −a} of Z_q.

## 3. Digit symmetry

`Decoding` treats all non-zero digits alike unless `Tree(nonzero='free')`. Then

- a joint type treats all non-zero digits alike: a parent 0 is written as 0+0 or i+(−i), a parent v ≠ 0 as 0+v
  or i+(v−i). There are two pair fractions per join.
- every filter has one price, for the digit 0 ("zero against the rest"), whatever its family: `z[0]` (≥ 0 with the
  opposite tilt, since swapping the two lists negates it), and `z_y[0]` for the own tilt. The base class `Filter`
  applies this rule, not the families.
- a joint type is given by the paper's cases (both digits zero, one zero, both non-zero with sum zero, both non-zero
  with non-zero sum), and a witness file writes one entry per case.
- for q = 4, 8, 16, … `mod(q)` is not the addition of the field, but only the numbers of pairs in these classes
  enter, and they are the same. So `mod(q)` stands for F_q. q must be a prime power.

`Tree(nonzero='free')` drops the symmetry: full q×q joint types, a free type on the window, one price per digit. It
needs a prime q, and the permutations count the type of the window. For subset sum, LWE and an error of a given type
(`TypedDecoding` and its subclasses), joint types and prices are always per digit. `LeeDecoding` gives types, joint types and the prices of the
syndrome filter per class {a, −a}, and a window filter one price per digit: the mirror image of a good window filter
is another good filter, but their average is not.

## 4. Large weight

If ω > (q − 1)/q, zero is not the most frequent digit of the error. The tree then works on e − (1,…,1). Over F₂
this is the complement, and ω and 1 − ω give the same exponent. As in the paper, the types p^(i) from the shifted
join up are those of e − (1,…,1) (`result.parameters`, the evaluation, the report and the files show them so).
Technically the highest representation join forms its parent as x + y + (1,…,1). The label of the all-one vector is
known, so nothing else changes. Over F_q with q ≥ 3 this needs `Tree(nonzero='free')`.

## 5. Number of solutions

The expected number of solutions is N = (number of vectors of weight ωn) / q^{(1−R)n}, and log₂ N per n is printed.

- `budget` (default) and `repetitions`: the tree looks for one solution, and the runs of the tree are the
  permutations.
- `accounting='many-solutions'` ([many solutions]): the representations need no condition (a deficit lowers the top
  list), every level including the top is capped by the number of distinct vectors, and the number of runs follows
  from the size L_d of the top list: runs = max(0, −L_d − log Q + min(0, log N)) (log Q: the final weight check on
  the syndrome part).

## 6. Near or far neighbours

A syndrome filter with the own tilt (`PriceFilter(tilt='own')`, the default of the named decoders) has its own
prices for each of the two lists and a free center. Two lists whose labels must be close come out with equal prices,
two lists whose labels must differ (large weight) with opposite preferences. `PriceFilter(tilt='same')` and
`LeeWeightFilter` fix the same tilt.

## Filters and centers

A filter is a filter family (`families.py`). Its exponents α, β_x, β_y come from one closed form (`filters.py`) for
every family, and `verify` re-derives them independently.

| family | parameters per filtered part | prices |
|---|---|---|
| `PriceFilter(tilt='opposite')` | `z`: one price per digit (class) | z_i free, second list −z (the paper's filter of the joins) |
| `PriceFilter(tilt='same')` | `z`: one price per digit (class) | z_i free, second list z |
| `PriceFilter(tilt='own')` | `z`, `z_y`: two prices per digit (class) | z_i, z^y_i free (the general filter, used by the syndrome filters) |
| `LeeWeightFilter()` | `sigma` ≥ 0 | z_i = z^y_i = −σ·wt_L(i) (the Lee syndrome filter, [lee metric]) |

Every family has the argument `u`: `'free'` (the default, where the center size u is one more variable) or a number
fixed by the user. A price is given for the non-zero digits (the digit 0 has price 0: only differences matter) and,
with a digit symmetry, for one digit of each class. A new family is one class (docs/GUIDE.md, "Extending the library",
shows the recipe).

## Partitioning

`Tree(parts=P)` cuts the positions the tree enumerates (subset sum and LWE: all of them, decoding: the window) into
P parts of sizes ε_1, …, ε_P (ε_P = 1 − Σ others), fixed once at random and the same for every list [partitioning].

- Every part has on the top level the type the problem gives the tree part (the solution, or the window type), so
  for one of polynomially many random partitions the planted solution has it. For decoding the window stays one
  piece of the permutations.
- Per part: the joint type of every representation join (`parts[k].J[i]`, or the level types `parts[k].p[i]` with
  maxent joint types) and the parameters of the join's filter (`parts[k].filter[i]`), same family.
- Shared: the digits, the label bits ℓ^(i), the top type p^(d), the syndrome part and the syndrome filter.
- D_i, r_i, c_i and α, β_x, β_y of a join are sums over the parts (a center is one product center per part). The
  list sizes, saturation, budget, repetitions, costs and memory follow from the sums. With equal parameters in every
  part the exponent equals the unpartitioned one (a test checks this).
- With the 7-dissection at join 0, its cost and memory use D_0, the entropy of the level-0 lists summed over the
  parts (each weighted by its size), like every other list size.

## The model in equations

The paper states the following in prose (decoding section), and this is what `model.py` computes. All quantities
are exponents per n. b = log₂ q is the number of bits of a label unit (b = 1 for subset sum and LWE), and d is the
depth.
A tree has the tree part (κ_i: the fraction of n it covers on level i, that is 1 for subset sum and LWE and κ for
decoding, halved by every split join above, and its type p^(i) on level i), the exact label bits ℓ^(0), …, ℓ^(d),
and for decoding the syndrome part (size λ = 1 − κ, error weight ω − ω₁, type t of the error on it). With a syndrome
filter, t is the type of the *difference* of the two labels on the syndrome part.

**Per join i** ([lemma 1], `model.contributions` and `model.per_join`, with the shares of `parts.py`):

    r_i       = κ_i (H(J^(i)) − H(p^(i)))     representation join, 0 at a split join
    c_i       = κ_i I(J^(i))                  consistency loss of the tree part, I(J) = H(p_x) + H(p_y) − H(J)
    c_label_i = ℓ^(i)                         the target condition: the exact label bits
    final_d   = λ (b − H(t)) = −log Q         the weight check on the syndrome part at the top join, apart from c_label_d
    ell_i     = ℓ^(i)                         label bits matched exactly (`ell[i]`, `ell_d`)

**Label bits fixed on a list of level i**, and the constraints K a part puts on the representations of the solution
(a constraint on a list of level i − 1 counts for the 2^{d−i} joins of level i):

    fixed_i = Σ_{j≤i} ℓ^(j)
    K(ℓ^(j)) = ℓ^(j) (2^{d−j} − 1)

**Lists** ([list sizes], `model.list_sizes`):

    L_half = D_0/2,   L_0 = 2 L_half − c_label_0,   L_i = 2 L_{i−1} − c_i − c_label_i,   D_i = κ_i H(p^(i))

The top row L_d is the size of the top list before the final weight check on the syndrome part (final_d = −log Q
above, as in [many solutions]), so that L_d, log Q and log N are the three numbers of the run count below. L_d < 0 means fewer
than one false candidate per run.

**Budget and repetitions** ([budget], [repetitions], `model.repetitions`), R = Σ_i 2^{d−i} r_i:

    budget  = R − Σ_j K(ℓ^(j))                                ≥ 0 with the accounting `budget`
    Δ_i     = max(0, Σ_{j≤i} K(ℓ^(j)) − R)                     with the accounting `repetitions`

**Permutations** ([isd], `Decoding.runs`). The window of κ n coordinates with error weight ω₁ n is taken out of the n
coordinates with error weight ω n:

    runs of the whole tree:  runs = B(1, ω) − B(κ, ω₁) − B(1 − κ, ω − ω₁),         B(a, b) = a·h(b/a)

for a weight spread equally over the non-zero digits. With a free window type it is the same count with the entropies
of the types. With `accounting='many-solutions'` there are no permutation terms, and the runs follow from L_d (rule 5).

**Join costs** ([plain join], [filtered join], `model.base_merge_cost`, `model.join_cost`):

    join 0, exact:        T_0 = max(L_half, 2 L_half − ℓ^(0)) = max(D_0/2, L_0),   memory L_half
    join 0, dissection:   T_0 = max(D_0/2, 2(D_0 − M_0)/3),  M_0 = min(M, D_0/4),   memory M_0
    join i, no filter:    T_i = max(L_{i−1}, 2 L_{i−1} − ℓ^(i))
    join i, filters:      T_i = max(L_{i−1} + α − β_x,  L_{i−1} + α − β_y,  2 L_{i−1} − ℓ^(i) + α − β_x − β_y)

with α, β_x, β_y summed over the filtered parts of the join ([filter exponents]): a center is then the product of
one center per part, so the exponents add. For join 0, L_{−1} is L_half. An exact top join does not filter on
the syndrome part, and the weight check on it is the final check log Q. Every join i < d also pays its output L_i.
`evaluation.T[i]` is T_i, and τ = max_i (T_i + Δ_i) is `evaluation.tau`.

With partitioning, D_i and the per-join r_i, c_i, α, β are sums over the tree parts.

**Time and memory** ([tree cost]):

    time = max_i (T_i + Δ_i) + runs,       memory = max(L_half or M_0, L_0, …, L_{d−1})

The top list is not stored. With `accounting='many-solutions'`, runs = max(0, −L_d − log Q + min(0, log N))
([many solutions]).

## The parameters of a tree

Users and witness files see the parameters under the paper's names (`parameters.py`, docs/GUIDE.md, "The parameters, named
as in the paper"). The model and the optimizer work on their own dict, which `build.variables` creates and
documents. `parameters.to_model` and `parameters.from_model` convert, and the problems name their own variables
(`Problem.paper_name`):

| paper name | the model's dict | conversion |
|---|---|---|
| `ell[i]`, i < d | `ell[i]` | bits = label units × log₂ q |
| `ell_d` (decoding, exact or mixed top join) | `ell_d` | the same |
| `omega_1` | `window.weight` | |
| `p[d]` (decoding, free digits) | `window.type` | per digit (the model holds class totals), with rule 4 the type of e₁ − (1, …, 1) |
| `lee_weight[d]` | `window.mean` | |
| `parts[k].size`, k < P | `parts.tree_k.size` | |
| `J[i]`, `parts[k].J[i]` | `joins[i].fractions[part]` | the model holds, for every parent digit s, the share of its positions written as each class of pairs a + b = s (the first class takes the remainder), and J^(i)_ab = p^(i)_s × share / (pairs in the class) |
| `p[i]`, i < d (`joint_types='maxent'`) | `joins[i+1].child[part]` | the free shares per digit (the digit 0, and the digit 1 for integer digits, follow) |
| `lee_weight[i]`, i < d | `joins[i+1].child[part].mean` | |
| `filter[i]`, `parts[k].filter[i]` | `joins[i].filter[tree]`, `joins[i].filter[tree_k]` | the family's own entries (`z`, `z_y`, `u`, `sigma`) |
| `syndrome_filter` | `joins[d].filter[rest]` | the same (one record: the syndrome filter is at the top join) |

`optimize(..., fixed={'filter[3].u': 0.5})` holds a free value. A name of a value that follows from the free ones
(`'J[2][0][0]'`, `'p[1][0]'`) becomes a constraint, and a group (`{'parts[1]': P.parts[1]}`) holds every free value
in it (docs/GUIDE.md, "Fixing and bounding").

## Constraints

All constraints are created by `model.constraints`, in this order. "model" constraints are stated in the paper, and
"domain" constraints say that a quantity is in its range. The report lists the model constraints by name with their
slack and the domain constraints as one line.

<!-- constraints:begin -->
| constraint (value >= 0) | meaning | group | from |
|---|---|---|---|
| `saturation: L_i <= D_i - fixed_i` | a list of level i, 1 <= i < d, has at most as many elements as there are vectors of its type with its fixed label bits (at level 0 this holds by construction) | model | Eq. (4) |
| `saturation: L_d <= D_d - fixed_d` | the same for the top list, only with `accounting='many-solutions'` | model | Appendix F |
| `representation budget >= 0` | sum_i 2^(d-i) r_i >= sum_parts K_part, only with `accounting='budget'` | model | Eq. (5) |
| `memory: L_half <= M` | the two halves of a base list are stored (not with `dissection`) | model | Eq. (8) |
| `memory: L_i <= M` | every list below the top is stored (i < d) | model | Eq. (8) |
| `7-dissection: D_0/7 <= M` | the 7-dissection needs memory 2^(D_0 n / 7) | model | Eq. (20) |
| `syndrome coordinates: exact <= (1 - R) n` | size of the syndrome part = 1 - R - (sum_j ell[j] + ell_d)/log2 q >= 0 | model | Section 7.1 |
| `error weight: window <= ω` | weight of the syndrome part = ω - ω₁ >= 0 | model | Section 7.1 |
| `error type: share of digit X on the syndrome part >= 0` | free window type (TypedDecoding and its subclasses, or `nonzero='free'`): every digit the error has is left over for the syndrome part | model | Section 7.1 |
| `label bits left for the top join: ell_d >= 0` | ell_d = 1 - sum_{j<d} ell[j] >= 0, the label bits left for the top join (SubsetSum, while LWE gives `ell_d`) | model | Eq. (3) |
| `size of X >= 0` | ell[j] and ell_d (label units), part sizes | domain | range of the quantity |
| `weight of X <= its size` | the syndrome part: its error weight is at most its size | domain | range of the quantity |
| `type of X: share of digit X >= 0` | the type of the window (or of the tree part): 0 <= ω₁ <= κ, or every share of a free window type | domain | range of the quantity |
| `join i X: pair fractions >= 0` | every pair fraction (the share of a parent digit written as a pair of a class) is between 0 and 1, and they sum to 1 by construction | domain | range of the quantity |
| `join i X: level type shares >= 0` | joint_types='maxent': the share of the digit 0 (the remainder of the level type) is >= 0 | domain | range of the quantity |
| `join i: joint type exists (scaling residual = 0)` | joint_types='maxent': a joint type with the two level types exists (the iterative scaling converges). The value is -max(0, residual - 1e-10), exactly 0 inside the domain, so that the optimizer's finite differences do not see the scaling noise | domain | range of the quantity |
| `L_i >= 0` | lists below the top are not empty | domain | range of the quantity |
<!-- constraints:end -->

## Variables and their bounds

The optimizer's variables are: ℓ_0, …, ℓ_{d−1} (and ℓ_d of an exact or mixed top join of a decoding problem), the
window weight or type, the sizes of the parts but the last (partitioning), the pair fractions (all but the first of
every table) or, with `joint_types='maxent'`, the shares of the child types (or the mean Lee weight of a Boltzmann
type), and the parameters of every filter as its family declares them. Their bounds are of two kinds
(`build.variables`):

- implied by the constraints above (sizes, weights and fractions between 0 and their maximum). They never bind
  beyond what the model requires.
- a pure **search box**. If a result lies on such a bound, `optimize` prints
  `WARNING: variable … is at its search bound …`, and `result.warnings` lists it: the optimum may lie outside.

<!-- bounds:begin -->
| variable | search box | note |
|---|---|---|
| ell[j] of LWE (labels unlimited below the top) | 0 .. 1 | the model has no upper bound |
| sigma (`LeeWeightFilter`) | 0 .. 10 | sigma >= 0 by its definition |
| z[0] of an opposite-tilt filter with all non-zero digits alike | 0 .. 40 | swapping the two lists negates it |
| digit prices z, z_y (`PriceFilter`) | -40 .. 40 | at 40 bits a fraction theta is 0 or 1 up to 1e-12 |
| center size u of a filter (u free) | 0.001 .. 1 - 0.001 | the offset of the closed form is solved from it |
<!-- bounds:end -->

## Things to know

- **The ℓ_i are free.** How many label units each join matches is optimized together with everything else, subject
  to saturation and the budget. Classical analyses that fix ℓ_i = r_i (every list holds one representation per
  element) are not reproduced exactly: for the digits {−1, 0, 1} and depth 3 the library gives 0.28915, the classical
  value of Becker, Coron and Joux is 0.291.
- **`Tree.digits` are the digits of the lists**, not of the solution, and they change the result. The default is the
  digits of the solution (decoding: the digits of the error and the sums of two of them, all of F_q for `Decoding`).
- **Pairs of digits whose sum is not a digit of the lists do not occur** (subset sum, LWE).
- **Memory** is the largest stored list. The top list is not stored.
- **The optimizer is a local search** from `starts` start points with a fixed `seed`: same input, same output. Its
  result is an upper bound on the optimum of the model. More `starts`, another `seed` or a `warm` start can improve
  it. `no_surplus=True` adds the search constraint budget ≤ 0, which helps random starts of subset-sum trees. It
  never makes a result invalid.
- **Points outside the domain of the formulas** (a logarithm of a negative number, a division by an empty part)
  count as infeasible during the search. Every result is re-evaluated and satisfies all constraints.
- **`optimize` raises `NoFeasiblePoint`** if no start reaches a feasible point (e.g. the memory bound is too small).
- **Exponents shown rounded are rounded up.**

## Comparisons in the paper's repository

The paper's comparisons with earlier algorithms are scripts in the paper's repository: a single nearest-neighbour
join, Stern's algorithm over F_q, Hirose's center count, filters with three sets of positions, trees whose joins below
the top also filter syndrome coordinates, and the decoders of [BBPSWW23].

## References

<!-- references:begin -->
| name used in the code | paper |
|---|---|
| [lemma 1] | Lemma 1 |
| [largest entropy] | Section 3.1 |
| [dissection] | Eq. (20) |
| [list sizes] | Eq. (3) |
| [saturation] | Eq. (4) |
| [budget] | Eq. (5) |
| [repetitions] | Eq. (6) |
| [plain join] | Eq. (7) |
| [tree cost] | Eq. (8) |
| [center 1/2] | Eq. (19) |
| [admissible] | Eq. (11) |
| [closed form] | Lemma 2 |
| [filter exponents] | Eqs. (10), (14) |
| [filtered join] | Eq. (15) |
| [filter blocks] | Theorem 2 |
| [partitioning] | Section 4.6 |
| [simplifications] | Section 4.4 |
| [main theorem] | Theorem 1 |
| [alphabets table] | Table 2 |
| [isd] | Section 7.1 |
| [neighbour join] | Section 7.2 |
| [decoders table] | Table 6 |
| [many solutions] | Appendix F |
| [lee metric] | Appendix H |
| [lee table] | Table 11 |

| key | work |
|---|---|
| [EZ23] | Esser, Zweydinger: New time-memory trade-offs for subset sum (Eurocrypt 2023): 7-dissection base lists, subtree repetitions |
| [MO15] | May, Ozerov: On computing nearest neighbors with applications to decoding of binary linear codes (Eurocrypt 2015) |
| [BJMM12] | Becker, Joux, May, Meurer: Decoding random binary linear codes in 2^(n/20) (Eurocrypt 2012) |
| [BKW22] | Bariffi, Khathuria, Weger: Information set decoding for Lee-metric codes using restricted balls (arXiv 2205.12903, 2022) |
| [BBPSWW23] | Baldi, Bitzer, Pavoni, Santini, Wachter-Zeh, Weger: Generic decoding of restricted errors (ISIT 2023) |
<!-- references:end -->
