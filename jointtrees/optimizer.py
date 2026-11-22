"""The optimizer (SciPy): minimizes the time exponent of `model.evaluate` over the free parameters of a tree.

    minimize tau   subject to   every constraint of the evaluation >= 0,   tau >= every cost term + runs

    result = optimize(problem, tree, memory_bound=None, starts=8, seed=1, warm=None, fixed=None, bounds=None,
                      time_limit=None, callback=None, quiet=False, no_surplus=False)
    result = polish(result)                       restart from the optimum until a round gains little
    space = SearchSpace(problem, tree, memory_bound)     the variables as a vector, for searches of one's own

The variables and their bounds come from build.variables, the constraints from model.constraints, and nothing about
a problem is written here.  Each start is first repaired to feasibility (L-BFGS-B on the squared violations) and then
solved by SLSQP.  The starts are warm starts, random points, and perturbed copies of the best point.  The problem is
not convex: the result is an upper bound on the optimum of the model, and a witness that verify.py checks without
SciPy.
"""
import copy
import os
import time as clock
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .build import build, model_variables
from .model import NoFeasiblePoint, evaluate
from .parameters import (Parameters, at_search_bounds, from_model, leaf_paths, lookup, model_place, node, parse,
                         part_place, put, set_node, text_of, to_model)
from .report import report
from .starts import embed
from .witness import Witness, from_json, joint_type_of


class TimeUp(Exception):
    """The time limit of `optimize` has passed (raised inside a SciPy call, caught by `optimize`)."""


@dataclass
class Result:
    """What `optimize` found.  `parameters` holds the free values, `evaluation` every value of the model: a quantity
    that is free, such as J^(i) of a tree with free joint types, is in both under the same name.  In `parameters` a
    quantity per join or level is a dict of its free entries, in `evaluation` a list over i = 0..d.

    time, memory     the exponents (bits per n) of the best point
    parameters       the best point in the paper's notation (parameters.py): result.parameters.ell[0], .J[2][0][1]
    evaluation       every quantity of the model at that point (model.Evaluation)
    warnings         variables at a bound that only limits the search: the optimum may lie beyond it
    starts           one entry per start: where it started, its time (None if it found no feasible point), its
                     status ('regular', 'not converged' (the solver stopped without its convergence test), 'cut'
                     (time limit), 'infeasible'), its seconds
    cut              True if the time limit stopped the search early
    fixed, bounds    what the caller fixed or bounded (stored in the witness, and used again by polish)
    """
    time: float
    memory: float
    parameters: Parameters
    evaluation: object
    instance: object
    warnings: list = field(default_factory=list)
    starts: list = field(default_factory=list)
    cut: bool = False
    fixed: dict = None
    bounds: dict = None

    @property
    def problem(self):
        return self.instance.problem

    @property
    def tree(self):
        return self.instance.tree

    @property
    def memory_bound(self):
        return self.instance.memory_bound

    def witness(self, **notes):
        """The witness of this result (witness.Witness), with `notes` stored in it.  The claimed exponents are those
        of `parameters` as they are now (also after an edit)."""
        return Witness(self.problem, self.tree, self.memory_bound, self.parameters, None, dict(notes),
                       fixed=self.fixed or None, bounds=self.bounds or None)

    def save(self, path, **notes):
        self.witness(**notes).save(path)

    def __str__(self):
        warnings = ''.join(f'\nWARNING: {w}' for w in self.warnings or [])
        return report(self.instance, self.evaluation, legend=True) + warnings


# ------------------------------------------------------------------------------------------------ the search space
class SearchSpace:
    """The free parameters of a tree as a vector x, with their bounds and the constraints of the model.

        space = SearchSpace(problem, tree, memory_bound=None, fixed=None, bounds=None)
        space.names            the name of every variable (parameters.py), e.g. 'filter[3].u'.  A variable that moves
                               a joint type is named 'J[i] (pair shares of digit s)'
        space.low, space.high  the bounds (numpy arrays)
        space.vector(start)    x of a start: parameters (paper form), a Result, a Witness or the path of a witness file,
                               of this tree or another one: what matches is used, the rest starts at its neutral value
        space.parameters(x)    the paper form at x
        space.model(x)         the model's dict at x
        space.evaluate(x)      the Evaluation at x, or None outside the domain of the formulas
        space.constraints(x)   every constraint value (feasible means all >= 0)
        space.random(rng)      a random start (numpy Generator)

    fixed  {name: value}       a free value is held at `value`.  A value that follows from the free ones (an entry of
                               a joint type, a level type, ...) becomes the constraint value == `value`.  A group
                               ({'parts[1]': P.parts[1]}, {'J[3]': P.J[3]}, {'ell': P.ell}) holds every free value in it
    bounds {name: (low, high)} the same with a range (None: no bound on that side)
    """

    def __init__(self, problem, tree, memory_bound=None, fixed=None, bounds=None, no_surplus=False):
        self.inst = inst = build(problem, tree, memory_bound)
        variables = model_variables(inst)
        self.template, self.remainders, self.owners = variables.template, variables.remainders, variables.owners
        self.paths = [slot.path for slot in variables.slots]
        self.places = [model_place(inst, path) for path in self.paths]
        self.low = np.array([slot.low for slot in variables.slots], dtype=float)
        self.high = np.array([slot.high for slot in variables.slots], dtype=float)
        self.no_surplus = no_surplus
        self.derived = []                               # (name, path, low, high) of the bounds on derived values
        self._cache = {}                                # x -> (model's dict, evaluation, memo): the last point only
        for name, value in (bounds or {}).items():
            if not (isinstance(value, (tuple, list)) and len(value) == 2):
                raise ValueError(f'bounds: {name!r} needs (low, high), None for an open side (got {value!r})')
        self.fixed, self.bounds = dict(fixed or {}), {k: tuple(v) for k, v in (bounds or {}).items()}
        for name, value in self.fixed.items():
            if isinstance(value, (dict, list)):                 # a group: a joint type, a filter, a part, ell
                self._hold(name, value)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'fixed: {name!r} needs a number, or the parameters of a group (got {value!r})')
            self._limit(name, value, value, 'fixed')
        for name, (low, high) in self.bounds.items():
            self._limit(name, low, high, 'bounds')

    @property
    def names(self):
        out = []
        for k, (path, place) in enumerate(zip(self.paths, self.places)):
            if place is not None:
                out.append(text_of(place[0]))
            else:                                               # ('joins', i, 'fractions', part, digit, pair)
                out.append(f'{text_of(self.group_of(k))} (pair shares of digit {path[4]}: {path[5]})')
        return out

    def group_of(self, k):
        """The paper place of variable k, and for a pair fraction the place of its joint type."""
        if self.places[k] is not None:
            return self.places[k][0]
        _, i, _, part, _, _ = self.paths[k]
        return part_place(self.inst, part) + ('J', i)

    def _hold(self, name, value):
        """fixed={name: group}: every free variable under `name` (e.g. 'parts[1]', 'J[3]', 'filter[2]', 'ell') is held
        at the value the group gives it."""
        path = parse(name)
        if len(path) >= 2 and path[-2] == 'J':                  # a joint type, also as a matrix or with string keys
            value = joint_type_of(value, self.inst, name)
        elif isinstance(value, (dict, list)):                   # a group as in a witness file (string keys)
            value = from_json(value, self.inst, path)
        start = Parameters()
        put(start, path, copy.deepcopy(value))
        places = {place[0] for place in self.places if place is not None}
        # every value of the paper form may be given: a derived entry of a group (the remainder of a type) follows
        # from the rest
        known = set(places)
        x0 = next(points_inside(self), None)
        if x0 is not None:
            known |= {leaf for group in self.paper(x0) for leaf in leaf_paths(group)}
        unknown = [text_of(leaf) for leaf in leaf_paths(value, path) if leaf not in known and 'J' not in leaf]
        if unknown:
            near = ", ".join(sorted(text_of(p) for p in places if p[:len(path)] == path)[:12]) or "none"
            raise ValueError(f'fixed: {", ".join(unknown[:6])} {"are" if len(unknown) > 1 else "is"} not a free '
                             f'parameter of this tree.  The free parameters in {name}: {near}')
        x = self.vector(start)
        held = [k for k in range(len(self.paths)) if self.group_of(k)[:len(path)] == path]
        J = lookup(start, path) if len(path) >= 2 and path[-2] == 'J' else None
        # a joint type: its entries are held, not only the shares per parent digit, and so the type of its level
        if J is not None:
            digits, add = self.inst.digits, self.inst.problem.add
            if not held:                                        # maximum-entropy joint types: every entry
                for a in digits:
                    for b in digits:
                        if a <= b:
                            self._limit(text_of(path + (a, b)), J[a][b], J[a][b], 'fixed')
                return
            level = path[:-2] + ('p', path[-1])
            for s in digits:
                share = sum(J[a][b] for a in digits for b in digits if add(a, b) == s)
                self._limit(text_of(level + (s,)), share, share, 'fixed')
        if not held:
            raise ValueError(f'fixed: {name!r} holds no free parameter of this tree (parameter_table(problem, tree) '
                             'lists them)')
        for k in held:
            if lookup(start, self.group_of(k)) is not None:     # what the group gives is held, the rest stays free
                self.low[k] = self.high[k] = x[k]

    def _limit(self, name, low, high, what):
        """fixed= or bounds= on one name: a free value gets the narrower range, and a value that follows from the free
        ones becomes a constraint (self.derived)."""
        if isinstance(name, str) and name.startswith(('joins.', 'window.', 'parts.tree')):
            raise ValueError(f'{what}: {name!r} is a path of the model\'s dict.  Write the name of the parameter, e.g. '
                             "'filter[3].u', 'ell[0]', 'omega_1' (parameter_table(problem, tree) lists them)")
        path = parse(name)
        k = next((k for k, place in enumerate(self.places) if place is not None and place[0] == path), None)
        if k is not None:                                       # a free value: its range is narrowed
            factor = self.places[k][1]
            if low is not None and high is not None and low > high:
                raise ValueError(f'{what}: {name}: the lower end {low} is above the upper end {high}')
            lo = self.low[k] if low is None else low / factor
            hi = self.high[k] if high is None else high / factor
            if lo > hi + 1e-15 or lo > self.high[k] + 1e-12 or hi < self.low[k] - 1e-12:
                raise ValueError(f'{what}: {name} must lie in [{self.low[k] * factor:g}, {self.high[k] * factor:g}] '
                                 f'(got {low}..{high})')
            self.low[k], self.high[k] = max(self.low[k], lo), min(self.high[k], hi)
            return
        if low is not None and high is not None and low > high:
            raise ValueError(f'{what}: {name}: the lower end {low} is above the upper end {high}')
        # a value that follows from the free ones: its value at two points inside the domain
        values, names = [], []
        for x in points_inside(self):
            values.append(self.value(x, path))
            names = names or list(dict.fromkeys(text_of(leaf) for group in self.paper(x) for leaf in leaf_paths(group)))
            if len(values) == 2:
                break
        if not values or not isinstance(values[0], (int, float)):
            near = [n for n in names if n.startswith(text_of(path[:1]))] or names
            raise ValueError(f'{what}: {name!r} names no number of the parameters of this tree.  Names nearby: '
                             f'{", ".join(near[:10]) or "none"} (parameter_table(problem, tree) lists all)')
        share = 'p' in path or 'J' in path or path[-1] in ('size', 'u')
        if share and ((low is not None and low > 1) or (high is not None and high < 0)):
            raise ValueError(f'{what}: {name} is a share, between 0 and 1 (got {low}..{high})')
        if len(values) == 2 and values[0] == values[1]:         # the same at both points: the problem or tree fix it
            inside = (low is None or values[0] >= low - 1e-12) and (high is None or values[0] <= high + 1e-12)
            if not inside:
                raise ValueError(f'{what}: {name} is {values[0]!r} for this problem and tree (they fix it), not '
                                 f'{low}..{high}')
        self.derived.append((name, path, low, high))

    # ---- points
    def model(self, x):
        """The model's dict at x: the first pair fraction of every group takes the remainder."""
        params = copy.deepcopy(self.template)
        for path, value in zip(self.paths, x):
            set_node(params, path, float(value))
        for r in self.remainders:
            fractions = node(params, r.path)
            fractions[r.first] = 1.0 - sum(v for k, v in fractions.items() if k != r.first)
        return params

    def evaluate(self, x):
        key = np.asarray(x, dtype=float).tobytes()
        if key not in self._cache:
            self._cache.clear()
            prm = self.model(x)
            # a point outside the domain of the formulas (log of a negative number, division by an empty part)
            # counts as infeasible
            try:
                ev = evaluate(self.inst, prm)
            except (ValueError, ZeroDivisionError, OverflowError):
                ev = None
            self._cache[key] = (prm, ev, {})
        return self._cache[key][1]

    def paper(self, x):
        """(parameters, derived) in the paper form at x (parameters.from_model)."""
        ev = self.evaluate(x)
        prm, _, memo = self._cache[np.asarray(x, dtype=float).tobytes()]
        if ev is None:
            raise ValueError('this point is outside the domain of the formulas')
        if 'P' not in memo:
            memo['P'] = from_model(self.inst, prm, ev)
        return memo['P']

    def parameters(self, x):
        """The free parameters at x, in the paper form."""
        return self.paper(x)[0]

    def value(self, x, path):
        """The value named by `path` at x: a free parameter or a derived value (the joint types of a maximum-entropy
        tree come from the evaluation)."""
        P, D = self.paper(x)
        value = lookup(P, path)
        value = lookup(D, path) if value is None else value
        if value is None and 'J' in path:
            ev = self.evaluate(x)
            if path[0] == 'J':
                value = lookup(Parameters(J=ev.J), path)
            elif path[0] == 'parts':
                value = lookup(Parameters(parts=ev.parts), path)
        return value

    def constraints(self, x):
        """Every constraint value at x (feasible: all >= 0): the model's, the remainders of the pair fractions, the
        bounds on derived values, and budget <= 0 with no_surplus."""
        ev = self.evaluate(x)
        if ev is None:
            return None
        prm = self._cache[np.asarray(x, dtype=float).tobytes()][0]
        values = [c.value for c in ev.records if not c.boxed] + [node(prm, r.path)[r.first] for r in self.remainders]
        if self.derived:
            for _, path, low, high in self.derived:
                value = self.value(x, path)
                values += ([value - low] if low is not None else []) + ([high - value] if high is not None else [])
        return np.array(values + ([-ev.budget] if self.no_surplus else []))

    def vector(self, start):
        """x of a start (None: the template, a feasible chain of the model's start values)."""
        return self.match(start)[0]

    def match(self, start):
        """(x, used, total): x of a start, and how many of the free values of this tree it gives."""
        used = None
        if start is None:
            prm = self.template
        else:
            if isinstance(start, (str, os.PathLike)):
                start = Witness.load(os.fspath(start))
            elif isinstance(start, dict) and 'parameters' in start and 'problem' in start:
                start = Witness.from_dict(start)                # the data of a witness file
            P = start.parameters if isinstance(start, (Result, Witness)) else start
            if not isinstance(P, Parameters) and isinstance(P, dict):
                P = from_json(P, self.inst, top=True)           # parameters as JSON data (string keys)
            used = sum(1 for k in range(len(self.paths)) if lookup(P, self.group_of(k)) is not None)
            if isinstance(start, (Result, Witness)) and start.instance.tree != self.inst.tree and used:
                P = embed(start, self.inst.problem, self.inst.tree, self.inst.memory_bound)     # new features off
            prm = to_model(self.inst, P, strict=False)
        x = np.array([float(node(prm, path)) for path in self.paths])
        return np.clip(x, self.low, self.high), used, len(self.paths)

    def random(self, rng):
        x = random_start(self.paths, self.low, self.high, rng, self.template, self.owners)
        return np.clip(x, self.low, self.high)

    def result(self, x):
        """The Result of the point x (to save it as a witness, or to start `optimize` from it)."""
        evaluation = self.evaluate(x)
        if evaluation is None:
            raise ValueError('this point is outside the domain of the formulas')
        return Result(evaluation.time, evaluation.memory, self.parameters(x), evaluation, self.inst,
                      at_search_bounds(self.inst, self.model(x)), [], False, dict(self.fixed) or None,
                      {k: list(v) for k, v in self.bounds.items()} or None)


def points_inside(space):
    """The points inside the domain of the formulas, where the derived values can be looked up, among the template and
    30 random points (seed 0)."""
    rng = np.random.default_rng(0)
    for x in [space.vector(None)] + [space.random(rng) for _ in range(30)]:
        if space.evaluate(x) is not None:
            yield x


# --------------------------------------------------------------------------------------------------- random starts
def owner_of(path, owners):
    """(base path, family, context) if `path` belongs to a filter, else None."""
    base = tuple(path[:4])
    return (base,) + owners[base] if len(path) > 4 and base in owners else None


def random_start(paths, low, high, rng, template=None, owners=None):
    """Small pair fractions, decreasing exact labels, filters as their families start them, and child types (maxent
    joint types) near their template values, which form a feasible chain."""
    scale = rng.uniform(.2, 1.)
    x, shared, filters = [], {}, {}
    labels = 1.0 if rng.random() < .5 else 10 ** rng.uniform(-1.5, 0)      # tiny exponents need tiny labels
    means = boltzmann_chain(paths, rng, template)
    parts = len([p for p in paths if p[0] == 'parts'])
    for path, lo, hi in zip(paths, low, high):
        name, owner = path[-1], owner_of(path, owners or {})
        if owner is not None:                           # a filter parameter: its family gives the start
            base, flt, ctx = owner
            if base not in filters:
                filters[base] = flt.start(ctx, rng, shared)
            x.append(filters[base].get(tuple(path[4:]), flt.default(tuple(path[4:]))))
        elif 'fractions' in path:
            x.append(rng.uniform(0, .05) * scale)
        elif path[0] in ('ell', 'ell_d'):
            x.append(rng.uniform(lo, lo + .3 * (hi - lo)) * labels)
        elif path[0] == 'parts':                        # partitioning: the size of a part
            x.append(rng.uniform(0.2, 0.8) / max(1, parts))
        elif path[0] == 'window':
            x.append(rng.uniform(lo, min(hi, .3)) if name == 'weight' else rng.uniform(lo, hi))
        elif path[-1] == 'mean':                        # a Boltzmann type: its mean Lee weight, from the chain above
            x.append(means[path])
        elif 'child' in path:                           # a share of a child type (maxent joint types)
            x.append((node(template, path) if template else 0.1) * rng.uniform(0.9, 1.6) + rng.uniform(0.0, 0.02))
        else:
            x.append(rng.uniform(lo, hi))
    ell = [k for k, path in enumerate(paths) if path[0] == 'ell']
    for k, value in zip(ell, sorted((x[k] for k in ell), reverse=True)):
        x[k] = value
    return np.array(x)


def boltzmann_chain(paths, rng, template):
    """Random mean Lee weights for Boltzmann level types, top down: the window at 0.4 .. 1.0 of the template's window
    mean (which is 0.8 of the error's), every child at 0.5 .. 0.85 of its parent (a join adds two digits per position:
    the sum has a larger Lee weight than one summand, but the child must still be at least half the parent)."""
    paths = [path for path in paths if path[-1] == 'mean']
    if not paths:
        return {}
    means, parent = {}, None
    for path in sorted(paths, key=lambda p: (p[0] != 'window', -p[1] if p[0] == 'joins' else 0)):
        if path[0] == 'window':
            parent = (node(template, path) if template else 1.0) * rng.uniform(0.5, 1.25)
        else:
            parent = (parent if parent is not None else 1.0) * rng.uniform(0.5, 0.85)
        means[path] = parent
    return means


# ------------------------------------------------------------------------------------------------------ optimize
def objective(z):
    """The objective of the epigraph form z = (x, tau): tau."""
    return z[-1]


class LocalSearch:
    """The local search of `optimize` from one start, in the epigraph form: minimize tau over z = (x, tau) subject to
    every constraint of the model and tau >= every cost term + runs.  A SciPy call ends at the deadline (None: no
    deadline) with its last iterate."""

    def __init__(self, space, deadline, probe):
        self.space, self.deadline, self.last = space, deadline, None
        self.bounds_z = list(zip(space.low, space.high)) + [(-1., 5.)]
        if space.evaluate(probe) is None:               # the number of constraints and terms, at a point inside
            probe = next(points_inside(space), probe)
        if space.evaluate(probe) is None:
            self.size = 2
        else:
            self.size = len(space.constraints(probe)) + len(space.evaluate(probe).terms)

    def violation(self, x):
        """The sum of the squared violations of the constraints at x (1e6 outside the domain of the formulas)."""
        c = self.space.constraints(x)
        if c is None:
            return 1e6
        return float(np.sum(np.minimum(c, 0.0) ** 2))

    def time(self, x):
        """The time at x (10 outside the domain of the formulas)."""
        e = self.space.evaluate(x)
        return 10.0 if e is None else e.time

    def epigraph(self, z):
        """The constraints of the epigraph form at z = (x, tau), all >= 0 when feasible."""
        e, c = self.space.evaluate(z[:-1]), self.space.constraints(z[:-1])
        if e is None or c is None:
            return -np.ones(self.size)
        return np.r_[c, [z[-1] - t - e.runs for t in e.terms.values()]]

    def inside_box(self, z):
        """The bounds of x as constraints, x - low >= 0 and high - x >= 0 (COBYLA takes no bounds)."""
        return np.r_[z[:-1] - self.space.low, self.space.high - z[:-1]]

    def watch(self, xk, *args):
        """SciPy callback: remember the iterate, stop at the deadline."""
        self.last = np.array(xk, dtype=float)
        if self.deadline is not None and clock.monotonic() > self.deadline:
            raise TimeUp

    def run(self, method, function, x0, **options):
        """A SciPy call that ends at the deadline with its last iterate: (x, status) with status 'regular', 'not
        converged' (the solver stopped without its convergence test) or 'cut'."""
        self.last = None
        try:
            out = minimize(function, x0, method=method, callback=self.watch, **options)
            return out.x, 'regular' if out.success else 'not converged'
        except TimeUp:
            if self.last is None:
                return np.asarray(x0, dtype=float), 'cut'
            return self.last, 'cut'

    def repair(self, x):
        """(y, status): x moved towards feasibility by L-BFGS-B on the squared violations."""
        low, high = self.space.low, self.space.high
        y, status = self.run('L-BFGS-B', self.violation, x, bounds=list(zip(low, high)), options={'maxiter': 500})
        return np.clip(y, low, high), status

    def one_start(self, x0):
        """(found, status) of the search from x0: found = (time, x) of the best feasible point reached, or None.  A
        start that violates a constraint is repaired first.  Where SLSQP leaves the domain (a kinked constraint), the
        derivative-free COBYLA runs from the start again."""
        low, high = self.space.low, self.space.high
        x, status = np.clip(x0, low, high), 'regular'
        if self.violation(x) > 1e-12:
            x, status = self.repair(x)
        found = None
        if self.violation(x) <= 1e-8 and status != 'cut':
            start = (self.time(x), x) if self.violation(x) <= 1e-12 else None
            z, status = self.run('SLSQP', objective, np.r_[x, self.time(x) + 1e-6], bounds=self.bounds_z,
                                 constraints=[{'type': 'ineq', 'fun': self.epigraph}],
                                 options={'maxiter': 1500, 'ftol': 1e-13})
            y = np.clip(z[:-1], low, high)
            if self.violation(y) > 1e-14 and status != 'cut':
                y, _ = self.repair(y)
            if self.violation(y) > 1e-12 and start is not None and status != 'cut':
                z, status = self.run('COBYLA', objective, np.r_[x, self.time(x) + 1e-6],
                                     constraints=[{'type': 'ineq', 'fun': self.epigraph},
                                                  {'type': 'ineq', 'fun': self.inside_box}],
                                     options={'maxiter': 4000, 'rhobeg': 0.02, 'tol': 1e-10})
                y = np.clip(z[:-1], low, high)
                if self.violation(y) > 1e-14:
                    y = self.repair(y)[0]
            if self.violation(y) <= 1e-12 and (start is None or self.time(y) < start[0]):
                found = (self.time(y), y)
            else:
                found = start
        return found, status


def perturbed(x, space, rng):
    """A copy of x moved at random: by 0.006 (standard deviation) in a variable of range at most 1, else by 0.05."""
    steps = np.array([.006 if hi - lo <= 1 else .05 for lo, hi in zip(space.low, space.high)])
    return x + rng.normal(0, 1, len(space.low)) * steps


def warm_starts(space, warm):
    """[(origin, x)] of the warm starts: a Result, a Witness, the path of a witness file or parameters, or a list of
    them."""
    if warm is None:
        warm = []
    elif not isinstance(warm, (list, tuple)):
        warm = [warm]
    out = []
    for k, w in enumerate(w for w in warm if w is not None):
        x, used, total = space.match(w)
        if used == 0:
            raise ValueError(f'warm start {k + 1} gives none of the free parameters of this tree (another problem, '
                             'another number of parts?).  parameter_table(problem, tree) lists them')
        out.append((f'warm start {k + 1} ({used} of {total} values)', x))
    return out


def optimize(problem, tree, memory_bound=None, starts=8, seed=1, warm=None, fixed=None, bounds=None, time_limit=None,
             callback=None, quiet=False, no_surplus=False):
    """The best parameters found for `tree` on `problem` with at most 2^(memory_bound n) memory (None: no bound).

    starts      number of starts after the warm starts (random points and perturbed copies of the best point)
    warm        start points: a Result, a Witness, the path of a witness file, parameters (paper form), or a list of
                them, of this tree or of another one (what matches is used, new features start switched off)
    fixed       {name: value}: e.g. {'filter[3].u': 0.1, 'ell[0]': 0.2, 'J[2][0][0]': 0.9} (the names of
                result.parameters, parameter_table(problem, tree) lists them).  A free value is held, a value that
                follows from the free ones becomes a constraint
    bounds      {name: (low, high)}: the same with a range (None: open on that side)
    time_limit  seconds of wall-clock time for the whole call.  The best point found so far is returned and
                result.cut is True
    callback    called after every start with its entry of result.starts
    quiet       do not print the progress and the report
    no_surplus  search aid: additionally require budget <= 0 (helps random starts of subset-sum trees)
    Returns a Result.  Raises NoFeasiblePoint if no start reached a point that satisfies the constraints (too small a
    memory bound, too few starts, or fixed values that contradict each other).
    """
    space = SearchSpace(problem, tree, memory_bound, fixed, bounds, no_surplus)
    if not len(space.low):                              # nothing to optimize: the one point of the tree
        x = np.zeros(0)
        c = space.constraints(x)
        if c is None or (len(c) and c.min() < -1e-9):
            raise NoFeasiblePoint('the tree has no free parameters, and its one point violates a constraint '
                                  '(e.g. the memory bound)')
        result = space.result(x)
        if not quiet:
            print(result, flush=True)
        return result

    def log(text):
        if not quiet:
            print(text, flush=True)
    deadline = None if time_limit is None else clock.monotonic() + time_limit
    candidates = warm_starts(space, warm)
    if not candidates:                                  # no warm start: the template first, a feasible chain
        candidates, starts = [('template', space.vector(None))], max(starts - 1, 0)
    search = LocalSearch(space, deadline, candidates[0][1])
    rng = np.random.default_rng(seed)
    best, entries, cut = None, [], False                # best: (time, x)
    total = len(candidates) + starts
    for k in range(total):
        if deadline is not None and clock.monotonic() > deadline:
            cut = True
            log(f'  time limit reached: {total - k} of {total} starts not run')
            break
        began = clock.monotonic()
        if k < len(candidates):
            origin, x0 = candidates[k]
        elif best is not None and k % 2:
            origin, x0 = 'perturbed best point', perturbed(best[1], space, rng)
        else:                                           # the least violating of 40 random draws
            origin = 'random point'
            x0 = min((space.random(rng) for _ in range(40)), key=search.violation)
        found, status = search.one_start(x0)
        better = found is not None and (best is None or found[0] < best[0] - 1e-12)
        if better:
            best = found
        cut = cut or status == 'cut'
        if found is None and status != 'cut':
            status = 'infeasible'
        entry = {'start': k + 1, 'from': origin, 'time': None if found is None else found[0], 'status': status,
                 'seconds': round(clock.monotonic() - began, 2)}
        entries.append(entry)
        outcome = 'no feasible point' if found is None else f'{found[0]:.9f}'
        log(f'  start {k + 1:2d} ({origin}): {outcome}' + ('  (new best)' if better else '')
            + ('  [cut by the time limit]' if status == 'cut' else ''))
        if callback is not None:
            callback(entry)
    if best is None:
        raise NoFeasiblePoint(f'no feasible point in {len(entries)} starts' + (' (time limit)' if cut else '')
                              + ': try more starts, a larger memory bound, a warm start from a similar tree, or check '
                              'that the fixed values and bounds can hold together')
    result = space.result(best[1])
    result.starts, result.cut = entries, cut
    log(str(result))
    return result


def polish(result, gain=1e-6, rounds=4, **options):
    """Restart the search from `result` (a Result, a Witness or the path of a witness file, and no other start) until
    a round gains less than `gain`, at most `rounds` times, with the result's fixed values and bounds.  `options` go
    to `optimize` (e.g. time_limit, quiet).  Returns the best Result."""
    options = {'quiet': True, **options}
    if not isinstance(result, Result):                      # a witness: its point as a Result first
        w = Witness.load(os.fspath(result)) if isinstance(result, (str, os.PathLike)) else result
        result = optimize(w.problem, w.tree, w.memory_bound, starts=0, warm=w, fixed=w.fixed,
                          bounds={k: tuple(v) for k, v in (w.bounds or {}).items()} or None, **options)
    best = result
    for _ in range(rounds):
        new = optimize(best.problem, best.tree, best.memory_bound, starts=0, warm=best, fixed=best.fixed,
                       bounds={k: tuple(v) for k, v in (best.bounds or {}).items()} or None, **options)
        improved = new.time < best.time - 1e-12
        if improved:
            new.starts = best.starts + new.starts
        gained = best.time - new.time
        best = new if improved else best
        if gained < gain or new.cut:
            break
    return best
