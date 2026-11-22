"""The extension recipes of docs/GUIDE.md, tested: a new filter family and a new problem, each one class in this file,
work with the model, `verify` and the witness files without any change to the library.
Run from the folder above:  python3 tests/test_extension.py   (standard library only)"""
import copy
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from jointtrees import (Filter, Join, PriceFilter, Record, SubsetSum, Tree, Variable, Witness,      # noqa: E402
                        evaluate)
from jointtrees import model                                                                        # noqa: E402
from jointtrees.build import build, variables                                                       # noqa: E402
from jointtrees.problems import problem_from_dict                                                   # noqa: E402
from jointtrees.verify import verify                                                                # noqa: E402
from jointtrees.witness import claimed_of                                                           # noqa: E402
from neighbour import SLOPE_MAX                                                                     # noqa: E402


@dataclass(frozen=True)
class AbsoluteValueFilter(Filter):
    """A made-up family: the price of digit j is slope * |j| on the first list, the opposite on the second."""
    name = 'absolute-value'
    tilt: str = 'opposite'
    u: object = 'free'

    def variables(self, ctx):
        return [Variable(('slope',), 0.0, SLOPE_MAX, ('high',))]

    def prices(self, ctx, p):
        a = {j: p['slope'] * abs(ctx.value(j)) for j in ctx.digits}
        return a, {j: -a[j] for j in a}

    def random_start(self, ctx, rng, shared):
        return {('slope',): rng.uniform(0.1, 1.0)}


class DenseSubsetSum(SubsetSum):
    """A made-up problem: subset sum whose solution has three quarters ones (only the constructor differs)."""
    arguments = ()

    def __init__(self):
        super().__init__(weight=0.75)


def with_new_family():
    """The LWE witness with the top join's filter replaced by the new family: (problem, tree, parameters)."""
    w = Witness.load(os.path.join(HERE, 'tests', 'witnesses', 'lwe_T0.5_d5_gf.json'))
    joins = list(w.tree.joins)
    joins[5] = Join(filter=AbsoluteValueFilter())
    tree = Tree(joins, digits=w.tree.digits)
    P = copy.deepcopy(w.parameters)
    P.filter[5] = Record(slope=0.2, u=0.45)                 # the family's parameters keep their own names
    return w.problem, tree, P


def test_new_filter_family():
    """A tree with the new family: evaluated, written to a witness, read back, verified independently."""
    problem, tree, P = with_new_family()
    witness = Witness(problem, tree, None, P, claimed_of(evaluate(problem, tree, P)))
    data = json.loads(json.dumps(witness.to_dict()))
    assert data['tree']['joins'][5]['filter']['family'] == 'absolute-value'
    assert data['parameters']['filter']['5']['slope'] == 0.2 and 'z' in data['derived']['filter']['5']
    passed, messages = verify(data)
    assert passed, [m for m in messages if m[0] == 'FAIL']
    assert any('alpha' in text and 'IPF' in text for _, text in messages)      # its exponents were re-derived


def test_new_problem():
    """A new problem class works with every tree, partitioning included, and is written to and read from a file."""
    problem = DenseSubsetSum()
    for tree in (Tree.same_joins(depth=2, digits=(-1, 0, 1)),
                 Tree.same_joins(depth=2, filter=PriceFilter(), digits=(-1, 0, 1), parts=2)):
        inst = build(problem, tree)
        assert inst.problem.solution == {0: 0.25, 1: 0.75}
        assert model.evaluate(inst, variables(inst).template).time > 0
    assert problem_from_dict(json.loads(json.dumps(problem.to_dict()))) == problem


def test_command_line_imports_extensions():
    """A witness that names a family defined outside the package verifies on the command line with --import of the
    file that defines it.  That file imports a module of its own folder (tests/neighbour.py), which --import makes
    importable."""
    problem, tree, P = with_new_family()
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'w.json')
        Witness(problem, tree, None, P, claimed_of(evaluate(problem, tree, P))).save(path)

        def verify_file(*extra):
            return subprocess.run([sys.executable, '-m', 'jointtrees', 'verify', *extra, path], cwd=HERE,
                                  capture_output=True, text=True)
        assert verify_file().returncode != 0
        assert verify_file('--import', os.path.abspath(__file__)).returncode == 0


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            test()
            print(f'ok  {name}', flush=True)
