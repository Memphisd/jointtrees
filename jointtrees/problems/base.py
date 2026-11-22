"""The base of every problem.  Standard library only.

A problem class is the problem: what the solution is, what its label is, how many solutions there are, and what the
runs of the whole tree pay.  The algorithm is the Tree.  The model (model.py) has no case for any problem: it asks the
problem, in this order,

    check(tree, digits)          reject Tree options that have no meaning here, with a message that says why
    rules(inst)                  apply the problem's rules to the instance (the quantities the model reads) and
                                 return the lines printed under "applied rules"
    variables(inst, template)    its own free parameters, for the optimizer
    parts(inst, prm)             the solution and its label cut into parts (parts.py)
    constraints(inst, parts)     its own constraints of the model
    runs(inst, parts)            runs of the whole tree

and reads the numbers `unit` (bits per label unit) and `log_solutions` (expected number of solutions) and the digit
addition `add`.

This base class is the case where the tree enumerates the whole vector and every label unit is matched exactly:
subset sum and LWE only set their numbers.  Decoding (decoding.py) overrides the methods with a window, the syndrome
part and permutations.  A new problem subclasses whichever is closer and overrides what differs.  It is registered
under its class name, so that a witness file can name it.
"""
from ..build import UNLIMITED_LABELS
from ..families import Variable
from ..model import Constraint
from ..parts import ExactPart, Parts, TreePart
from ..references import ref


class Problem:
    """The base of every problem (module docstring).  `ell_max` is the number of label bits of the target per n.
    It is 1 for subset sum, whose top join matches what the joins below leave of it, ℓ^(d) = 1 - (ℓ^(0) + ... +
    ℓ^(d-1)), and the constraint ell_d >= 0 keeps that number from going negative.  It is None for LWE, whose top join
    matches the given `ell_d` bits and whose joins below have no bound."""
    arguments = ()              # names of the constructor arguments: what defines the problem (repr, witness files)
    registry = {}               # class name -> class, filled by every subclass (problem_from_dict)
    unit = 1.0                  # bits per label unit
    q = None                    # size of the field of a q-ary problem
    rate = None                 # code rate of a decoding problem
    integers = True             # the digits add as integers: the mean of a type is conserved by a join
    log_solutions = 0.0         # log2 of the expected number of solutions, per n: here one planted solution
    solution = None             # {digit: frequency}: the type of the solution (decoding: of the error e)
    meanings = {}               # {name: words} for the problem's own parameters (paper_name), for parameter_table
    ell_max = 1                 # label bits of the target per n (class docstring)
    ell_d = None                # ℓ^(d): label bits matched by the top join if given (None: what is left of ell_max)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Problem.registry[cls.__name__] = cls

    def add(self, a, b):
        return a + b

    def default_digits(self):
        """The digits of the lists when the Tree gives none: the digits of the solution."""
        return tuple(sorted(self.solution))

    def check(self, tree, digits):
        """Reject Tree options that have no meaning for this problem (a ValueError that says why)."""
        if any(join.syndrome is not None for join in tree.joins):
            raise ValueError('Join(syndrome=...): a join that matches by distance needs a decoding problem (syndrome '
                             'coordinates)')
        if tree.nonzero is not None:
            raise ValueError('nonzero: whether the non-zero digits of an error are alike, only for Decoding')
        missing = sorted(a for a in self.solution if a not in digits)
        if missing:
            raise ValueError(f'Tree.digits: the digits {missing} of the solution are not digits of the lists')
        sums = {self.add(a, b) for a in digits for b in digits}
        if tree.representation_joins and any(v > 0 and a not in sums for a, v in self.solution.items()):
            raise ValueError('Tree.digits: a digit of the solution is not the sum of two digits of the lists')

    def rules(self, inst):
        """Apply the rules of this problem to the instance and return their lines.  Here there is nothing to
        apply: the whole vector is the tree part and the labels are matched exactly."""
        return []

    def variables(self, inst, template):
        """Its own free parameters (build.variables documents the format): the label units ell_0..ell_{d-1}
        matched exactly at the joins below the top.  Start values go into `template`, and the result is a list of
        Variable(path, low, high, search)."""
        if self.ell_max is None:
            return [Variable(('ell', j), 0.0, UNLIMITED_LABELS, ('high',)) for j in range(inst.depth)]
        return [Variable(('ell', j), 0.0, self.ell_max / self.unit) for j in range(inst.depth)]

    def paper_name(self, inst, path):
        """The paper's name of a variable of this problem (see parameters.py): (place, scale) with paper value =
        scale * model value, or None for the variables the tree defines.  Here: ℓ^(i), the label units of join i in
        bits."""
        if path[0] == 'ell':
            return ('ell', path[1]), inst.unit
        return None

    def parts(self, inst, prm):
        """The tree part is the whole solution.  ell_d, the label units of the top join, is given (`ell_d`) or what
        the joins below leave of `ell_max`.  It is the remainder, and its sign is a constraint of the problem."""
        ell = list(prm.get('ell', [0.0] * inst.depth))
        if self.ell_d is not None:
            ell.append(self.ell_d / self.unit)
        else:
            ell.append(self.ell_max / self.unit - sum(ell))
        tree = TreePart('tree', 1.0, {a: float(self.solution.get(a, 0.0)) for a in inst.digits}, 0)
        exact = [ExactPart(f'ell_{j}', x, {0: 1.0}, j, remainder=(j == inst.depth)) for j, x in enumerate(ell)]
        return Parts(tree, exact)

    def constraints(self, inst, parts):
        """The joins below the top leave at least 0 label bits for the top join (unless `ell_d` is given)."""
        if self.ell_d is not None:
            return []
        return [Constraint('label bits left for the top join: ell_d >= 0', parts.exact[-1].size, 'model',
                           ref('list sizes'))]

    def runs(self, inst, parts):
        """log2 of the runs of the whole tree, per n.  Here one run: the tree finds the solution if a representation
        survives."""
        return 0.0

    def to_dict(self):
        """The problem as written in a witness file: its class name and its parameters."""
        return {'name': type(self).__name__, **{k: plain(getattr(self, k)) for k in self.arguments}}

    def replace(self, **changes):
        """The same problem with some arguments changed, e.g. Decoding(q=3, rate=0.4).replace(rate=0.42)."""
        unknown = sorted(set(changes) - set(self.arguments))
        if unknown:
            raise ValueError(f'{type(self).__name__}.replace: {unknown} are not arguments.  The arguments are '
                             f'{list(self.arguments)}')
        return type(self)(**{**{k: getattr(self, k) for k in self.arguments}, **changes})

    def __eq__(self, other):
        return type(self) is type(other) and all(getattr(self, k) == getattr(other, k) for k in self.arguments)

    def __hash__(self):
        return hash((type(self).__name__, repr(self)))

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={getattr(self, k)!r}' for k in self.arguments)})"


def plain(value):
    """An argument of a problem as JSON data: a dict with string keys, a tuple as a list."""
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def problem_from_dict(d):
    """The problem of a witness file (Problem.to_dict)."""
    if d.get('name') not in Problem.registry:
        raise ValueError(f"unknown problem {d.get('name')!r}, known: {', '.join(sorted(Problem.registry))}")
    cls = Problem.registry[d['name']]
    arguments = {k: v for k, v in d.items() if k != 'name'}
    unknown = sorted(set(arguments) - set(cls.arguments))
    if unknown:
        raise ValueError(f"problem {d['name']}: unknown arguments {unknown}.  The arguments are {list(cls.arguments)}")
    return cls(**arguments)
