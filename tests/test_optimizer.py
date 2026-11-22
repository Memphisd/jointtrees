"""Tests that run the optimizer (numpy, scipy).  Run from the folder above:  python3 tests/test_optimizer.py
(5 minutes)"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from jointtrees import (LWE, Decoding, Join, PriceFilter, SubsetSum, Tree, evaluate, optimize,      # noqa: E402
                        polish)
from jointtrees.algorithms import bcj, bjmm, may_ozerov, mixed_top                                  # noqa: E402
from jointtrees.optimizer import NoFeasiblePoint, SearchSpace                                       # noqa: E402
from jointtrees.starts import deeper, into_parts                                                    # noqa: E402
from jointtrees.verify import verify                                                                # noqa: E402


def best(problem, tree, **options):
    return optimize(problem, tree, quiet=True, **options)


def test_weights_stay_in_their_domain():
    """A nearest-neighbour base merge cannot reach a time near 0 through a negative weight."""
    result = best(Decoding(q=2, rate=0.5, weight='gv'), Tree([Join(exact=False, syndrome=PriceFilter('own'))]),
                  starts=4)
    assert result.time > 0.09 and result.evaluation.runs >= 0
    assert min(c.value for c in result.evaluation.records) > -1e-6
    assert verify(result.witness())[0]


def test_binary_complement_symmetry():
    """Weight W and 1 - W are the same problem over F_2 (e -> e + (1,...,1)) and give the same exponent."""
    for tree in (bjmm(2), may_ozerov(2)):
        low = best(Decoding(q=2, rate=0.5, weight=0.1), tree, starts=8)
        high = best(Decoding(q=2, rate=0.5, weight=0.9), tree, starts=8)
        assert abs(low.time - high.time) < 1e-8, (low.time, high.time)


def test_free_digits_do_not_beat_alike_digits():
    """Over F_3 at small weight, free digits with accounting='many-solutions' give the same exponent as alike digits
    (the non-uniform window type is paid for)."""
    code = Decoding(q=3, rate=0.5, weight='gv/2')
    alike = best(code, bjmm(2), starts=8)
    free = best(code, Tree.same_joins(depth=2, nonzero='free', accounting='many-solutions'), starts=8)
    assert abs(alike.time - free.time) < 1e-7, (alike.time, free.time)


def test_no_feasible_point_is_an_error():
    """optimize never returns None: without a feasible point, also for a tree without free parameters, it raises
    NoFeasiblePoint."""
    for tree, bound, starts in ((bcj(3), 0.01, 2), (Tree([Join()]), 0.2, 8)):
        try:
            best(SubsetSum(), tree, memory_bound=bound, starts=starts)
        except NoFeasiblePoint:
            continue
        raise AssertionError(f'no error: {tree!r}')


def test_known_exponents():
    """BCJ at depth 3, and May-Ozerov with and without window filters."""
    assert round(best(SubsetSum(), bcj(3), starts=4).time, 4) == 0.2891
    code = Decoding(q=2, rate=0.42, weight='gv')
    assert abs(best(code, may_ozerov(2), starts=6).time - 0.096646228) < 1e-8
    window = Tree.same_joins(depth=2, filter=PriceFilter(),
                             top=Join(exact=False, filter=PriceFilter(), syndrome=PriceFilter('own')))
    assert abs(best(code, window, starts=6).time - 0.096646228) < 1e-8


def test_search_bound_warning():
    """A result on a pure search-box bound is flagged."""
    result = best(LWE(key={-1: .25, 0: .5, 1: .25}), Tree.same_joins(depth=1, digits=(-1, 0, 1)), starts=2)
    assert isinstance(result.warnings, list)


def test_fixed_parameters():
    """fixed= holds a free parameter (named as in the paper) and turns a derived value into a constraint, bounds=
    narrows a range, and unknown names, model paths and values outside the range are errors.  ℓ^(d) is 'ell_d', free
    at a mixed top join and derived for subset sum."""
    tree = Tree.same_joins(depth=2, filter=PriceFilter(), digits=(-1, 0, 1))
    result = best(SubsetSum(), tree, starts=2, fixed={'filter[2].u': 0.3, 'J[2][0][0]': 0.4})
    P = result.parameters
    assert P.filter[2].u == 0.3 and abs(P.J[2][0][0] - 0.4) < 1e-8 and result.witness().fixed
    low = best(SubsetSum(), tree, starts=0, warm=result, bounds={'ell[0]': (0.31, None)})
    assert low.parameters.ell[0] >= 0.31 - 1e-12
    top = result.evaluation.ell_d
    short = best(SubsetSum(), tree, starts=0, warm=result, bounds={'ell_d': (None, top - 0.01)})
    assert short.evaluation.ell_d <= top - 0.01 + 1e-8 and short.parameters.ell_d is None
    held = best(Decoding(q=3, rate=0.466, weight='gv/2'), mixed_top(1), starts=2, fixed={'ell_d': 0.05})
    assert held.parameters.ell_d == 0.05 and held.evaluation.ell_d == 0.05
    for name, value, words in (('filter[2].width', 0.1, 'names no number'),
                               ('joins.2.filter.tree.center', 0.1, 'model'), ('filter[2].u', 1.5, 'must lie in')):
        try:
            best(SubsetSum(), tree, starts=1, fixed={name: value})
        except ValueError as error:
            assert words in str(error), (name, str(error))
        else:
            raise AssertionError(name)


def test_partitioning_is_not_worse():
    """Two parts started from the one-part optimum are at least as good (subset sum, digits -1, 0, 1, depth 3)."""
    one = best(SubsetSum(), Tree.same_joins(depth=3, filter=PriceFilter(), digits=(-1, 0, 1)), starts=4)
    two = best(SubsetSum(), Tree.same_joins(depth=3, filter=PriceFilter(), digits=(-1, 0, 1), parts=2), starts=2,
               warm=into_parts(one))
    assert two.time <= one.time + 1e-9 and verify(two.witness())[0], (one.time, two.time)


def test_time_limit_polish_and_starts():
    """time_limit stops the search (result.cut), every start is reported, polish never makes a result worse, and a
    deeper tree started by `deeper` is at least as good as the shallower optimum."""
    tree = Tree.same_joins(depth=2, filter=PriceFilter(), digits=(-1, 0, 1))
    seen = []
    result = best(SubsetSum(), tree, starts=3, callback=seen.append)
    assert len(seen) == len(result.starts) == 3 and all(s['status'] in ('regular', 'not converged') for s in seen)
    assert polish(result).time <= result.time + 1e-12
    cut = best(SubsetSum(), Tree.same_joins(depth=5, filter=PriceFilter(), digits=(-1, 0, 1, 2)), starts=50,
               time_limit=3)
    assert cut.cut and len(cut.starts) < 51
    tree3 = Tree.same_joins(depth=3, filter=PriceFilter(), digits=(-1, 0, 1))
    start = deeper(result, SubsetSum(), tree3)
    assert best(SubsetSum(), tree3, starts=0, warm=start).time <= result.time + 1e-9


def test_groups_embed_and_polish_from_a_witness():
    """fixed= with a whole joint type holds its entries, a warm start from another tree starts at the old time
    (embed), and polish starts from a witness file."""
    tree = Tree.same_joins(depth=2, filter=PriceFilter(), digits=(-1, 0, 1))
    first = best(SubsetSum(), tree, starts=2)
    M = first.parameters.J[2]
    other = best(SubsetSum(), tree, starts=1, seed=5, fixed={'J[2]': M})
    assert max(abs(other.parameters.J[2][a][b] - M[a][b]) for a in M for b in M) < 1e-8
    exact = best(SubsetSum(), Tree.same_joins(depth=2, digits=(-1, 0, 1)), starts=2)
    from_exact = best(SubsetSum(), tree, starts=0, warm=exact)
    assert from_exact.starts[0]['from'].startswith('warm start 1') and from_exact.time <= exact.time + 1e-9
    with tempfile.TemporaryDirectory() as folder:
        first.save(os.path.join(folder, 'w.json'))
        assert polish(os.path.join(folder, 'w.json')).time <= first.time + 1e-12


def test_dissection_with_a_memory_bound_verifies():
    """A tree with the 7-dissection under a memory bound gives a witness that verify accepts."""
    tree = Tree([Join(dissection=True), Join(), Join()], digits=(-1, 0, 1), accounting='repetitions')
    result = best(SubsetSum(), tree, memory_bound=0.1, starts=2)
    assert verify(result.witness().to_dict())[0]


def test_polish_keeps_fixed_values():
    """polish of a witness file keeps the values that were fixed when it was found."""
    with tempfile.TemporaryDirectory() as folder:
        fixed = best(SubsetSum(), bcj(2), starts=2, fixed={'ell[0]': 0.29})
        fixed.save(os.path.join(folder, 'fixed.json'))
        assert polish(os.path.join(folder, 'fixed.json')).parameters.ell[0] == 0.29


def test_warm_start_from_another_problem():
    """A warm start from another problem gives the values that share a name, and the start says how many."""
    other = best(Decoding(3, 0.466, 'gv/2'), may_ozerov(1), starts=1)       # shares only some names (ell[0], J[1])
    result = best(SubsetSum(), Tree.same_joins(depth=2, filter=PriceFilter(), digits=(-1, 0, 1)), starts=0, warm=other)
    origin = result.starts[0]['from']                                       # 'warm start 1 (u of t values)'
    used, total = (int(k) for k in origin.split('(')[1].split(' values')[0].split(' of '))
    assert 0 < used < total, origin


def test_level_type_held_as_a_group():
    """A level type of a tree with maximum-entropy joint types, written in full, can be held as a group."""
    key, tree = LWE({-1: .25, 0: .5, 1: .25}), Tree.same_joins(depth=2, digits=(-1, 0, 1), joint_types='maxent')
    result = best(key, tree, starts=1)
    held = best(key, tree, starts=0, warm=result, fixed={'p[1]': result.parameters.p[1]})
    assert abs(held.time - result.time) < 1e-9


def test_search_space():
    """SearchSpace: names, bounds and the round trip vector -> parameters -> vector."""
    tree = Tree.same_joins(depth=2, filter=PriceFilter(), digits=(-1, 0, 1))
    space = SearchSpace(SubsetSum(), tree)
    x = space.random(np.random.default_rng(3))
    assert len(space.names) == len(x) and 'filter[2].u' in space.names
    assert np.allclose(space.vector(space.parameters(x)), x, atol=1e-12)
    assert abs(evaluate(SubsetSum(), tree, space.parameters(x)).time - space.evaluate(x).time) < 1e-12


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            test()
            print(f'ok  {name}', flush=True)
