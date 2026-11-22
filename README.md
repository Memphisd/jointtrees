# jointtrees

A Python library for the analysis of representation trees. It accompanies the paper
*One Type Fits All: A General Analysis of Representation Trees for Subset Sum, LWE and Decoding*
(Andre Esser, 2026).

The library follows the notation of the paper. A tree is given by its joins, and each join by the joint type
J^(i) of the two vectors it adds. For a problem and a tree, the library sets up the optimization problem of the
paper, solves it numerically, and stores the result as a witness file that anyone can check without the optimizer.
Every quantity has the name it has in the paper.

## Install

```sh
git clone https://github.com/Memphisd/jointtrees.git && cd jointtrees
python3 -m venv .venv && . .venv/bin/activate
pip install .
```

## Example

```python
from jointtrees import SubsetSum, filtered, optimize

result = optimize(SubsetSum(), filtered(depth=3, digits=(-1, 0, 1)))   # about 15 seconds
result.time                      # 0.284316: time 2^(0.2844 n)
result.parameters.J[2]           # the joint type J^(2) of the join into level 2
result.parameters.filter[3].u    # the center size u of the filter at join 3
result.save('tree.json')         # python3 -m jointtrees verify tree.json
```

## Notation

| paper | library |
|---|---|
| subset sum, LWE, decoding over F_q, restricted errors, Lee metric | `SubsetSum()`, `LWE(key)`, `Decoding(q, rate)`, `RestrictedDecoding(q, rate, E, weight)`, `LeeDecoding(q, rate)` |
| tree with joins 0, …, d | `Tree([Join(), …, Join()])`, or `filtered(depth, digits)` for the filtered tree |
| exact join, filtered join, nearest-neighbour join | `Join()`, `Join(filter=PriceFilter())`, `Join(exact=False, syndrome=PriceFilter(tilt='own'))` |
| digits of the lists, partitioning into P parts | `Tree(…, digits=(-1, 0, 1))`, `Tree(…, parts=P)` |
| ℓ^(i), J^(i), p^(i), ω₁ | `ell[i]`, `J[i][a][b]`, `p[i][a]`, `omega_1` |
| filter at join i: center size u, prices z | `filter[i].u`, `filter[i].z[a]` |
| time, memory, L^(i), r^(i), c^(i) | `evaluation.time`, `.memory`, `.L[i]`, `.r[i]`, `.c[i]` |

## Checking a result

```sh
python3 -m jointtrees verify tree.json     # recompute the exponents from the stored parameters
python3 -m jointtrees evaluate tree.json   # every quantity of the tree, level by level
```

## More

- `docs/GUIDE.md`: the full reference (all problems, joins and filters, parameters, optimizer, witness format,
  extending the library)
- `RULES.md`: what the library derives from a problem and a tree
- `examples/`: subset sum, LWE, decoding, Lee metric, partitioning, trade-offs
- https://github.com/Memphisd/one-type-fits-all: the witnesses and scripts behind every number of the paper

## License

MIT
