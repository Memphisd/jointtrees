"""jointtrees: representation trees with joint types and filtered joins.  See README.md and docs/GUIDE.md.

    from jointtrees import SubsetSum, Decoding, Tree, Join, PriceFilter, filtered, optimize
    result = optimize(SubsetSum(), filtered(depth=3, digits=(-1, 0, 1)))
    result.parameters.J[2][-1][1], result.parameters.filter[3].u, result.evaluation.L
"""
from . import build, starts
from .algorithms import filtered
from .families import Filter, LeeWeightFilter, PriceFilter, Variable
from .model import NoFeasiblePoint
from .parameters import Parameters, checked, parameter_names, parameter_table
from .problems import LWE, Decoding, LeeDecoding, Problem, RestrictedDecoding, SubsetSum, TypedDecoding
from .record import Record
from .tree import Join, Tree
from .witness import Witness, from_json, witness_files

__version__ = '0.1'
__all__ = ['Tree', 'Join', 'filtered', 'Filter', 'PriceFilter', 'LeeWeightFilter', 'Variable', 'Parameters', 'Record',
           'NoFeasiblePoint', 'Witness', 'witness_files', 'Problem', 'SubsetSum', 'LWE', 'Decoding', 'TypedDecoding',
           'RestrictedDecoding', 'LeeDecoding', 'starts', 'evaluate', 'parameter_names', 'parameter_table', 'optimize',
           'polish', 'SearchSpace']


def evaluate(problem, tree, parameters, memory_bound=None):
    """The evaluation (model.Evaluation: time, memory, L, D, T, r, c, ...) of `tree` on `problem` at `parameters`
    (paper form: e.g. result.parameters, Witness.load(path).parameters, or a start from jointtrees.starts).  Every
    free parameter of the tree must be given, and nothing else.  A joint type must sum to the type of its level.
    `evaluation.feasible` says whether every constraint of the model holds (`evaluation.violations` lists the
    others)."""
    inst = build.build(problem, tree, memory_bound)
    P = getattr(parameters, 'parameters', parameters)
    if not isinstance(P, Parameters):
        P = from_json(P, inst, top=True)
    return checked(inst, P)[1]


def __getattr__(name):
    """optimize, polish and SearchSpace (optimizer.py): SciPy is imported on their first use, so that the rest of the
    library, verify among it, runs without it."""
    if name in ('optimize', 'polish', 'SearchSpace'):
        from . import optimizer
        return getattr(optimizer, name)
    raise AttributeError(f'module jointtrees has no attribute {name!r}')
