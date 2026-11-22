"""Tests that need only the standard library.  Run from the folder above:  python3 tests/test_library.py"""
import ast
import copy
import doctest
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tools'))
from jointtrees import (LWE, Decoding, Join, LeeDecoding, LeeWeightFilter, Parameters,      # noqa: E402
                        PriceFilter, RestrictedDecoding, SubsetSum, Tree, TypedDecoding, Witness, algorithms,
                        filters, jointtypes, model, parameter_names, parameter_table, parts)
from jointtrees import evaluate as evaluate_at                                              # noqa: E402
from jointtrees.build import build, context, variables                                      # noqa: E402
from jointtrees.crosscheck import layout, sum_tree                                          # noqa: E402
from jointtrees.entropy import H                                                            # noqa: E402
from jointtrees.lee import boltzmann_type                                                   # noqa: E402
from jointtrees.parameters import from_model, parse, text_of, to_model                      # noqa: E402
from jointtrees.problems import problem_from_dict                                           # noqa: E402
from jointtrees.starts import deeper, embed, into_parts, with_digits                        # noqa: E402
from jointtrees.verify import verify                                                        # noqa: E402
from jointtrees.witness import from_json, null_in, to_json, witness_files                   # noqa: E402
import rules_table                                                                          # noqa: E402

WITNESSES = os.path.join(HERE, 'tests', 'witnesses')       # one witness per problem and per feature


def rejected(make, tree, *words):
    """Building the problem (make() or a problem) and the instance must raise a ValueError with these words.  `tree`
    may be a function, so that errors raised when the Tree itself is made are caught too."""
    try:
        build(make() if callable(make) else make, tree() if callable(tree) else tree)
    except ValueError as error:
        assert all(word in str(error) for word in words), (str(error), words)
        return True
    raise AssertionError(f'accepted: {tree}')


def load(name):
    return Witness.load(os.path.join(WITNESSES, name))


def raw(path):
    with open(path) as handle:
        return json.load(handle)


def refused(data, words):
    """Reading the data of a witness file must raise a ValueError with these words."""
    try:
        Witness.from_dict(data)
    except ValueError as error:
        assert words in str(error), (str(error), words)
        return
    raise AssertionError(f'accepted: {words}')


def lambdas_in(node):
    """The lambdas of an expression that are not arguments of a call."""
    if isinstance(node, ast.Lambda):
        return [node]
    if isinstance(node, ast.Call):
        return []
    return [found for child in ast.iter_child_nodes(node) for found in lambdas_in(child)]


def test_doctests():
    assert all(doctest.testmod(m).failed == 0 for m in (model, filters, jointtypes, parts))


def test_code_style():
    """Every line of the package has at most 120 characters, and no lambda is bound to a name (a def names what it
    computes)."""
    long, bound = [], []
    for root, _, names in os.walk(os.path.join(HERE, 'jointtrees')):
        for name in sorted(n for n in names if n.endswith('.py')):
            path = os.path.join(root, name)
            with open(path, encoding='utf-8') as handle:
                text = handle.read()
            where = os.path.relpath(path, HERE)
            long += [f'{where}:{k}' for k, line in enumerate(text.splitlines(), 1) if len(line) > 120]
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
                    bound += [f'{where}:{found.lineno}' for found in lambdas_in(node.value)]
    assert not long, f'lines over 120 characters: {long}'
    assert not bound, f'lambdas bound to names: {bound}'


def test_witnesses():
    """Every witness reproduces its claimed time to 1e-9, satisfies all constraints, and passes the independent
    checks."""
    files = witness_files(WITNESSES)
    bad = [os.path.relpath(path, HERE) for path in files if not verify(raw(path))[0]]
    assert not bad, bad
    return len(files)


def test_readers_reject_what_they_cannot_read():
    """Unknown tree fields (e.g. the multi-label trees of another study) and files of another format are errors with a
    message, not wrong numbers."""
    for d in ({'joins': [{}], 'colour': 1}, {'depth': 2, 'labels': 3}):
        try:
            Tree.from_dict(d)
        except ValueError as error:
            assert 'unknown' in str(error)
        else:
            raise AssertionError(d)
    data = raw(os.path.join(WITNESSES, 'lwe_T0.5_d5_gf.json'))
    data['format'] = 'jointtrees witness 2'
    try:
        Witness.from_dict(data)
    except ValueError as error:
        assert 'jointtrees witness 3' in str(error)
    else:
        raise AssertionError('format 2 accepted')


def test_embed_another_family_and_tree():
    """A start from another tree (starts.embed): LeeWeightFilter becomes own prices with its prices, and an exact top
    join becomes a mixed top join whose new syndrome filter filters nothing (both at the same time).  A family with
    other parameters lists its prices under "derived"."""
    w = load('q4_half_R0.46_d3_leeweight.json')                   # LeeWeightFilter: sigma, and its prices z (derived)
    own = w.tree.replace(joins=w.tree.joins[:3] + (Join(syndrome=PriceFilter('own')),))
    start = embed(w, w.problem, own)
    assert abs(evaluate_at(w.problem, own, start).time - w.evaluate().time) < 1e-12
    z = w.to_dict()['derived']['syndrome_filter']['z']                # the reference digit 0 is not listed
    assert abs(z['1'] + w.parameters.syndrome_filter.sigma) < 1e-15 and '0' not in z
    w = load('tree_exact_d2_full_M0.040.json')                    # exact top join -> mixed top join
    mixed = w.tree.replace(joins=w.tree.joins[:2] + (Join(syndrome=PriceFilter('own')),))
    start = embed(w, w.problem, mixed, w.memory_bound)
    assert abs(evaluate_at(w.problem, mixed, start, w.memory_bound).time - w.evaluate().time) < 1e-12


def test_crosscheck_skips_what_it_does_not_cover():
    """The second implementation covers subset sum and LWE trees with pair fractions, also on parts and with the
    7-dissection.  A maxent tree is reported as not covered instead of failing."""
    assert layout(Tree.same_joins(depth=2, filter=PriceFilter(), joint_types='maxent').to_dict()) is None
    assert layout(Tree.same_joins(depth=2, filter=PriceFilter(), parts=2).to_dict()) is not None
    assert layout(Tree.same_joins(depth=2, filter=PriceFilter()).to_dict()) is not None
    w = load('subsetsum_pm2_M0.09_d3_two.json')                     # parts and the 7-dissection
    tau = sum_tree(w.to_dict())[0]
    assert abs(tau - w.evaluate().time) < 1e-9 and w.tree.parts == 2 and w.tree.dissection


def test_tree_round_trip():
    """A tree written to a witness file (to_dict) is read back as the same tree."""
    split = Join(split=True)
    trees = [Tree.same_joins(depth=5, filter=PriceFilter(), digits=(-2, -1, 0, 1, 2), parts=2),
             Tree([Join(), Join(), Join(filter=PriceFilter(), syndrome=LeeWeightFilter(u=0.4))]),
             Tree([Join(), split, Join(exact=False, filter=PriceFilter(u=0.5), syndrome=PriceFilter('own'))],
                  nonzero='free', accounting='many-solutions'),
             Tree([Join(), split, Join(), Join(), split, split, split], nonzero='free', accounting='many-solutions'),
             Tree([Join(dissection=True), Join(), Join(filter=PriceFilter(u=0.5)), Join(filter=PriceFilter()),
                   Join(filter=PriceFilter(u=0.3))], accounting='repetitions')]
    for tree in trees:
        assert Tree.from_dict(json.loads(json.dumps(tree.to_dict()))) == tree, tree
    mixed = Tree.same_joins(depth=2, filter=PriceFilter(), top=Join(filter=PriceFilter(), syndrome=PriceFilter('own')))
    assert mixed.mixed and mixed.joins[2].kind == 'mixed join with window filter'


def test_problems_round_trip():
    """A problem written to a witness file (to_dict) is read back as the same problem."""
    for problem in (SubsetSum(), LWE({-1: .25, 0: .5, 1: .25}), LWE({-1: .25, 0: .5, 1: .25}, ell_d=0.8),
                    Decoding(3, 0.5, 'gv/2'),
                    TypedDecoding(3, 0.5, {0: .8, 1: .15, 2: .05}), RestrictedDecoding(157, 0.5, (1, 156), 0.4),
                    LeeDecoding(47, 0.4), LeeDecoding(4, 0.46, 'gv/2')):
        assert repr(problem_from_dict(json.loads(json.dumps(problem.to_dict())))) == repr(problem), problem


def test_join_combinations():
    """The combinations of Join fields that the paper has no join for are rejected with a message."""
    dec = Decoding(q=2, rate=0.5)
    rejected(dec, lambda: Tree([Join(), Join(exact=False)]), 'syndrome filter')
    rejected(dec, lambda: Tree([Join(), Join(split=True, filter=PriceFilter())]), 'split join')
    rejected(dec, lambda: Tree([Join(), Join(syndrome=PriceFilter('own')), Join()]), 'top join only')
    rejected(dec, lambda: Tree([Join(), Join(exact=False, syndrome=PriceFilter('own')), Join()]), 'top join only')
    rejected(dec, lambda: Tree([Join(exact=False, syndrome=PriceFilter('own')), Join()]), 'top join only')
    rejected(dec, lambda: Tree([Join(), Join(split=True, exact=False, syndrome=PriceFilter('own')), Join()]),
             'top join only')
    rejected(dec, lambda: Tree([Join(dissection=True, exact=False, syndrome=PriceFilter('own')), Join()]), 'dissection')
    rejected(dec, lambda: Tree([Join(filter=PriceFilter()), Join()]), 'join 0')
    rejected(SubsetSum(), Tree([Join(), Join(exact=False, syndrome=PriceFilter('own'))]), 'decoding')
    rejected(dec, lambda: Tree([Join(), Join(filter='full')]), 'filter family')
    rejected(dec, lambda: Tree([Join(), Join(filter=PriceFilter(tilt='up'))]), 'tilt')
    rejected(dec, lambda: Tree([Join(), Join(filter=PriceFilter(u=1.5))]), 'center size')
    rejected(dec, lambda: Tree([Join(), Join(split=True, syndrome=PriceFilter('own'))]), 'exact=False')
    rejected(dec, Tree([Join(), Join(exact=False, syndrome=LeeWeightFilter())]), 'alike')
    rejected(SubsetSum(), Tree([Join(), Join(filter=LeeWeightFilter())]), 'Z_q')
    rejected(dec, lambda: Tree.from_dict({'joins': [{}], 'colour': 1}), 'unknown')


def test_digits_alike_rule():
    """With all non-zero digits alike (Decoding over F_q, rule 3) every filter family has one price, on the digit 0:
    with the opposite tilt z[0] >= 0 and z_y = -z, with own prices z[0] and z_y[0]."""
    tree = Tree([Join(), Join(filter=PriceFilter()), Join(exact=False, syndrome=PriceFilter('own'))])
    inst = build(Decoding(q=3, rate=0.5, weight='gv/2'), tree)
    assert [v.path for v in PriceFilter().declared(context(inst, False))] == [('z', '0'), ('u',)]
    a, b = PriceFilter().price_pair(context(inst, False), {'z': {'0': 2.0}, 'u': .5})
    assert a == {0: 2.0, 1: 0.0, 2: 0.0} and b == {0: -2.0, 1: 0.0, 2: 0.0}
    assert [v.path for v in PriceFilter('own').declared(context(inst, True))] == [('z', '0'), ('z_y', '0'), ('u',)]


def test_lee_prices():
    """LeeWeightFilter: price -sigma * Lee weight on both lists.  A syndrome filter over the Lee metric has one price
    per class {a, -a}, a window filter one price per digit."""
    inst = build(LeeDecoding(q=5, rate=0.5),
                 Tree([Join(), Join(filter=PriceFilter()), Join(syndrome=LeeWeightFilter())]))
    ctx = context(inst, True)
    a, b = LeeWeightFilter().price_pair(ctx, {'sigma': 1.0, 'u': .5})
    assert a == {0: 0.0, 1: -1.0, 2: -2.0, 3: -2.0, 4: -1.0} and b == a
    a, _ = PriceFilter().price_pair(context(inst, False), {'z': {'1': 1.5, '2': 3.0}})
    assert a == {0: 0.0, 1: 1.5, 2: 3.0, 3: 0.0, 4: 0.0}
    a, _ = PriceFilter('own').price_pair(ctx, {'z': {'1': 1.5, '2': 3.0}, 'z_y': {}})
    assert a == {0: 0.0, 1: 1.5, 2: 3.0, 3: 3.0, 4: 1.5}


def test_lee_over_any_q():
    """The Lee metric needs only the addition mod q: LeeDecoding works for q = 4 and 49.  An error of a given type
    over F_q needs a prime q."""
    for q in (4, 49):
        build(LeeDecoding(q=q, rate=0.46, weight='gv/2'), Tree.same_joins(depth=2))
    rejected(lambda: TypedDecoding(q=4, rate=0.5, error={0: .8, 1: .1, 3: .1}), Tree.same_joins(depth=2), 'prime')


def test_partitioning_with_equal_parts_changes_nothing():
    """Two parts with the joint types and filters of the unpartitioned tree give its exponent, whatever the sizes of
    the parts: partitioning is a refinement (any problem, here the tree of [main theorem], LWE and decoding trees)."""
    for name in ('subsetsum_pm3_d6_two.json', 'lwe_T0.3_pm3_d5_two.json', 'z2_W0.2_d3_f_mixed.json',
                 'hard_RS_filt_nn.json', 'q4_half_R0.4600_d3_mixed.json', 'q47_R0.35_d2_f_nn.json',
                 'q11_d2_nn_half.json', 'q11_d1_mixed_half.json'):
        w = load(name)
        P, tree = w.parameters, w.tree
        if tree.parts == 2:                                    # use part 1 only
            shared = {k: v for k, v in P.items() if k != 'parts'}
            P = Parameters({**shared, **{k: v for k, v in P.parts[1].items() if k != 'size'}})
            tree = Tree.from_dict({**tree.to_dict(), 'parts': 1})
        single = evaluate_at(w.problem, tree, P, w.memory_bound).time
        two_parts = Tree.from_dict({**tree.to_dict(), 'parts': 2})
        two = evaluate_at(w.problem, two_parts, into_parts(P, sizes=[0.37]), w.memory_bound).time
        assert abs(single - two) < 1e-12, (name, single, two)


def test_depth_zero_and_named_algorithms():
    """Depth 0 (Stern with a nearest-neighbour join on all syndrome coordinates) and every named algorithm build and
    evaluate at their start parameters."""
    a = algorithms
    cases = [(Decoding(q=11, rate=0.5, weight='gv/2'), Tree([Join(exact=False, syndrome=PriceFilter('own'))]))]
    cases += [(SubsetSum(), f(3)) for f in (a.hgj, a.bcj, a.bbss)] + [(SubsetSum(), a.filtered(3, (-1, 0, 1), parts=2))]
    cases += [(LWE({-1: .25, 0: .5, 1: .25}), a.meet_lwe(3))]
    cases += [(Decoding(q=2, rate=0.42), f(2)) for f in (a.bjmm, a.may_ozerov, a.mixed_top)]
    cases += [(Decoding(q=3, rate=0.5, weight=0.894), a.wagner(4, nonzero='free'))]
    for problem, tree in cases:
        inst = build(problem, tree)
        assert math.isfinite(model.evaluate(inst, variables(inst).template).time), (problem, tree)


def test_partitioning_scope():
    """Partitioning applies to every problem, also with the 7-dissection at join 0."""
    for problem in (SubsetSum(), LWE({-1: .25, 0: .5, 1: .25}), Decoding(3, 0.5, 'gv/2'), LeeDecoding(5, 0.5)):
        inst = build(problem, Tree.same_joins(depth=2, filter=PriceFilter(), parts=3))
        template = variables(inst).template
        assert set(template['parts']) == {'tree_1', 'tree_2'} and inst.tree_parts == ['tree_1', 'tree_2', 'tree_3']
    assert build(SubsetSum(), Tree([Join(dissection=True)] + [Join()] * 3, parts=2), 0.2).tree.parts == 2


def test_negative_weight_witness_rejected():
    """An empty syndrome part with a negative error weight, which would give a time near 0, is rejected by the weight
    constraint.  Here the window takes every syndrome coordinate and more weight than the error has."""
    problem, tree = Decoding(q=2, rate=0.5), Tree.same_joins(depth=2)
    inst = build(problem, tree)
    prm = variables(inst).template
    prm['ell_d'] = 1 - problem.rate                              # binary: one bit per syndrome coordinate
    prm['window']['weight'] = problem.omega + 0.1
    passed, messages = verify(Witness(problem, tree, None, from_model(inst, prm)[0]).to_dict())
    assert not passed and any(level == 'FAIL' and 'error weight' in text for level, text in messages)


def test_many_solutions_keep_the_error_type():
    """With accounting='many-solutions' the rest of the error has the proportions of the error type: a restricted error
    has no weight on digits outside E (an equal spread over all non-zero digits would give too low exponents)."""
    problem = RestrictedDecoding(q=5, rate=0.5, E=(1, 4), weight=0.3)
    inst = build(problem, Tree([Join(), Join(split=True), Join(split=True)], accounting='many-solutions'))
    rest = problem.parts(inst, variables(inst).template).rest
    assert rest.type[2] == rest.type[3] == 0 and abs(rest.type[1] - rest.type[4]) < 1e-15


def test_free_digits_and_error_types():
    """A free window type is allowed with every accounting, and an error of a given type as well."""
    f3 = Decoding(q=3, rate=0.5, weight='gv')
    build(f3, Tree.same_joins(depth=2, nonzero='free'))
    build(f3, Tree.same_joins(depth=2, nonzero='free', accounting='many-solutions'))
    typed = TypedDecoding(q=3, rate=0.5, error={0: .8, 1: .15, 2: .05})
    assert abs(build(typed, Tree.same_joins(depth=2)).log_solutions
               - (H({0: .8, 1: .15, 2: .05}) - 0.5 * math.log2(3))) < 1e-12
    rejected(typed, Tree.same_joins(depth=2, nonzero='alike'), 'alike')


def test_large_weight_alike_needs_free_digits():
    """Large weight over F_q, q >= 3, with all non-zero digits alike is rejected (binary is fine)."""
    rejected(Decoding(q=3, rate=0.5, weight=0.9), Tree.same_joins(depth=2), 'large weight')
    assert build(Decoding(q=2, rate=0.5, weight=0.9), Tree.same_joins(depth=2)).shift == {2: 1}


def test_default_digits():
    """Tree.digits=None: the digits of the solution, and for decoding the digits of the error and their sums."""
    assert build(SubsetSum(), Tree.same_joins(depth=2)).digits == (0, 1)
    assert build(Decoding(q=5, rate=0.5), Tree.same_joins(depth=2)).digits == (0, 1, 2, 3, 4)
    assert build(RestrictedDecoding(157, 0.5, (1, 156), 0.4), Tree.same_joins(depth=2)).digits == (0, 1, 2, 155, 156)


def test_input_validation():
    """Every malformed input gets a ValueError that says what is expected."""
    dec, ss = Decoding(q=2, rate=0.5, weight='gv'), SubsetSum()
    rejected(lambda: LWE(key={0: .5, 1: .4}), Tree.same_joins(depth=2), 'sum to 1')
    rejected(lambda: LWE(key={0: .5, 2: .5}), Tree.same_joins(depth=2, digits=(0, 1)), 'not digits of the lists')
    rejected(lambda: LWE(key={-1: .2, 0: .6, 1: .2}, ell_d=0), Tree.same_joins(depth=2), 'ell_d')
    rejected(lambda: Decoding(q=6, rate=0.45), Tree.same_joins(depth=2), 'prime power')
    rejected(lambda: Decoding(q=2, rate=1.5), Tree.same_joins(depth=2), 'rate')
    rejected(dec, Tree.same_joins(depth=2, digits=(0, 1, 2)), 'all digits')
    rejected(dec, lambda: Tree.same_joins(depth=2, nonzero='Free'), 'nonzero')
    rejected(Decoding(q=4, rate=0.45, weight='gv'),
             Tree.same_joins(depth=2, nonzero='free', accounting='many-solutions'), 'prime')
    assert any('F_4' in rule for rule in build(Decoding(q=4, rate=0.45, weight='gv'), Tree.same_joins(depth=2)).rules)
    rejected(ss, Tree.same_joins(depth=2, nonzero='free'), 'nonzero')
    rejected(ss, lambda: Tree.same_joins(depth=2, accounting='both'), 'accounting')
    rejected(ss, lambda: Tree.same_joins(depth=2, accounting='subsample'), 'many-solutions')
    rejected(ss, Tree.same_joins(depth=2, joint_types='maxent', types='boltzmann'), 'LeeDecoding')


def test_constraints_come_from_one_function():
    """Every constraint is a record of model.constraints with a group, and model constraints carry a reference."""
    ev = load('tree_nn_d2_full_M0.040.json').evaluate()
    assert {c.group for c in ev.records} == {'model', 'domain'}
    assert all(c.reference for c in ev.records if c.group == 'model') and len(ev.constraints) == len(ev.records)


def test_rules_page_is_current():
    """RULES.md's generated tables match the code, and every constraint the model produces has a row there."""
    with open(os.path.join(HERE, 'RULES.md')) as handle:
        text = handle.read()
    assert rules_table.render(text) == text, 'run python3 tools/rules_table.py'
    rows = {part.strip() for name, *_ in rules_table.CONSTRAINTS for part in name.split(', ')}
    rows.add('saturation: L_i <= D_i - fixed_i')
    for path in witness_files(WITNESSES):
        names = {rules_table.pattern_of(c.name) for c in Witness.load(path).evaluate().records}
        assert names <= rows, names - rows


def test_paper_form():
    """The parameters of every witness under the paper's names: written in format 3 and read back they give the same
    file and time, and every joint type, type and derived value of the file agrees with the model (Witness.check).
    The file has no null in its parameters and derived values, and the derived block repeats no free value.  A
    quantity per join or level is an object keyed by the index, and the syndrome filter one object."""
    for path in witness_files(WITNESSES):
        w = Witness.load(path)
        data = json.loads(json.dumps(w.to_dict()))
        again = Witness.from_dict(data)
        assert abs(again.evaluate().time - w.evaluate().time) < 1e-12 and not again.check(), path
        assert json.loads(json.dumps(again.to_dict())) == data, path
        assert null_in(data['parameters']) is None and null_in(data['derived']) is None, path
        assert not set(data['parameters']) & set(data['derived']) - {'parts', 'filter', 'syndrome_filter', 'p'}, path
        blocks = [data[key] for key in ('parameters', 'derived')]
        blocks += [part for key in ('parameters', 'derived') for part in data[key].get('parts', {}).values()]
        for block in blocks:
            assert all(isinstance(block[key], dict) for key in ('J', 'p', 'lee_weight', 'filter') if key in block)
            assert not any(key.isdigit() for key in block.get('syndrome_filter', {})), path      # names, no index
    P = load('subsetsum_pm3_d6_two.json').parameters
    J = P.parts[2].J[3]
    assert J[-1][1] == J[1][-1] and abs(sum(sum(row.values()) for row in J.values()) - 1) < 1e-12
    assert len(P.ell) == 6 and P.ell_d is None                  # ℓ^(d) of subset sum follows from the others
    try:
        P.J
    except AttributeError as error:
        assert 'parts[k].J' in str(error)
    else:
        raise AssertionError('P.J with two parts')
    w = load('q11_d1_mixed_half.json')                           # digits alike: the five cases of the paper
    Q = w.parameters
    assert Q.J[1][1][10] == Q.J[1][2][9] and Q.J[1][1][2] == Q.J[1][3][4] and 'parts' not in Q
    ev = w.evaluate()
    assert abs(ev.kappa - 0.494 - (sum(ev.ell) + ev.ell_d) / math.log2(11)) < 1e-12 and ev.omega_1 == Q.omega_1
    assert Q.ell_d == ev.ell_d and len(Q.ell) == len(ev.ell) == 1  # the mixed top join: ℓ^(d) is free
    for text in ('filter[3].u', 'parts[1].J[2][0][1]', 'syndrome_filter.z_y[0]', 'ell[0]', 'ell_d'):
        assert text_of(parse(text)) == text


def test_digit_classes_and_options_round_trip():
    """Model -> paper form -> JSON -> paper form -> model gives the same time at a random interior point, for every kind
    of variable: Lee classes {a, -a} with free and maximum-entropy joint types, Boltzmann types, restricted errors on a
    subset of Z_q, digits alike, rule 4 (large weight), partitioning with maximum-entropy types, many solutions, and
    parts with a free window type."""
    a = algorithms
    lee = LeeDecoding(5, 0.5, 'gv/2')
    cases = [(lee, Tree.same_joins(depth=3)), (lee, Tree.same_joins(depth=3, joint_types='maxent')),
             (lee, Tree([Join()] * 3 + [Join(syndrome=PriceFilter('own'))], joint_types='maxent', types='boltzmann')),
             (RestrictedDecoding(17, 0.5, (1, 16), 0.3), a.mixed_top(2, window=PriceFilter())),
             (Decoding(5, 0.45, 'gv'), a.mixed_top(2)),
             (Decoding(3, 0.5, 0.9),
              Tree([Join(), Join(), Join(split=True)], nonzero='free', accounting='many-solutions')),
             (SubsetSum(), Tree.same_joins(depth=3, digits=(-1, 0, 1), joint_types='maxent', parts=2)),
             (Decoding(3, 0.5, 'gv/2'), Tree.same_joins(depth=3, filter=PriceFilter(), nonzero='free', parts=2)),
             (LWE({-2: .1, -1: .2, 0: .4, 1: .2, 2: .1}),
              Tree.same_joins(depth=3, filter=PriceFilter('own'), digits=(-2, -1, 0, 1, 2)))]
    rng = random.Random(1)
    for problem, tree in cases:
        inst = build(problem, tree)
        found = variables(inst)
        for attempt in range(200):                              # a point with every variable moved, inside the domain
            prm = copy.deepcopy(found.template)
            for slot in found.slots:
                node = prm
                for key in slot.path[:-1]:
                    node = node[key]
                shares = 'fractions' in slot.path or slot.path[0] in ('ell', 'window') or 'child' in slot.path
                node[slot.path[-1]] = slot.low + (slot.high - slot.low) * rng.uniform(0.02, 0.15 if shares else 0.9)
            for r in found.remainders:
                node = prm
                for key in r.path:
                    node = node[key]
                node[r.first] = 1 - sum(v for c, v in node.items() if c != r.first)
            try:
                ev = model.evaluate(inst, prm)
            except ValueError:
                continue
            types = [t for part in ev.by_part.values() for t in part['types']]
            if min(v for t in types for v in t.values()) >= 0:      # every type a type: shares >= 0
                time = ev.time
                break
        else:
            raise AssertionError(f'no point inside the domain: {problem} {tree}')
        P, D = from_model(inst, prm)
        back = from_json(json.loads(json.dumps(to_json(P, inst))), inst, top=True)
        again = model.evaluate(inst, to_model(inst, back)).time
        assert abs(again - time) < 1e-12, (problem, tree, time, again)


def test_witness_file_checks_derived_values():
    """A witness file whose derived values do not follow from its parameters is reported (here a level type), and so
    is a joint type whose sums do not give the level above."""
    data = raw(os.path.join(WITNESSES, 'subsetsum_d4_filtered_M0.3.json'))
    data['derived']['p']['2']['1'] += 0.01
    passed, messages = verify(data)
    assert not passed and any('p[2]' in text for level, text in messages if level == 'FAIL')
    data = raw(os.path.join(WITNESSES, 'subsetsum_d4_filtered_M0.3.json'))
    J = data['parameters']['J']['2']
    J[0][1] += 0.01
    J[1][0] += 0.01
    passed, messages = verify(data)
    assert not passed and any('J[2]' in text for level, text in messages if level == 'FAIL')


def test_witness_paths_are_checked():
    """An edit of the parameters that the model cannot reproduce is refused by evaluate, Witness.evaluate, saving and
    loading, with a message that names it.  The claimed time of a result's witness is that of its parameters."""
    w = load('lwe_T0.5_d5_gf.json')

    def unknown_name(P):
        P.filter[3].center = 0.1

    def derived_value(P):
        P.ell_d = 0.3

    def unbalanced(P):                                          # J[2] no longer sums to the type of its level
        P.J[2][-1][1] += 0.001
        P.J[2][1][-1] += 0.001
    for edit, words in ((unknown_name, 'not a value'), (derived_value, 'follows from the free parameters'),
                        (unbalanced, 'J[2]')):
        bad = Witness(w.problem, w.tree, w.memory_bound, copy.deepcopy(w.parameters))
        edit(bad.parameters)
        for action in (bad.evaluate, bad.to_dict, lambda: evaluate_at(w.problem, w.tree, bad.parameters)):
            try:
                action()
            except ValueError as error:
                assert words in str(error), str(error)
            else:
                raise AssertionError('accepted')
    good = Witness(w.problem, w.tree, None, copy.deepcopy(w.parameters))
    good.parameters.filter[3].u = 0.4                            # a consistent edit: claimed follows it
    with tempfile.TemporaryDirectory() as folder:
        good.save(os.path.join(folder, 'w.json'))
        again = Witness.load(os.path.join(folder, 'w.json'))
    assert again.claimed['time'] == evaluate_at(w.problem, w.tree, good.parameters).time != w.claimed['time']


def test_failed_save_keeps_the_old_file():
    """Saving checks the parameters first: a save that fails leaves the old file as it was."""
    w = load('lwe_T0.5_d5_gf.json')
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'w.json')
        w.save(path)
        size = os.path.getsize(path)
        bad = Witness(w.problem, w.tree, None, copy.deepcopy(w.parameters))
        bad.parameters.J[2][0][0] += 0.01
        try:
            bad.save(path)
        except ValueError:
            pass
        assert os.path.getsize(path) == size and Witness.load(path).evaluate().time == w.evaluate().time


def test_verify_checks_the_claimed_memory():
    """verify compares the claimed memory, not only the claimed time."""
    data = load('lwe_T0.5_d5_gf.json').to_dict()
    data['claimed']['memory'] = 0.01
    assert not verify(data)[0]


def test_notes_and_fixed_values_round_trip():
    """Notes keep names of their own under "notes", next to the format fields, and the fixed values are kept."""
    w = load('lwe_T0.5_d5_gf.json')
    notes = {'tree': 'x', 'claimed': {'time': 0.2}}
    data = Witness(w.problem, w.tree, None, w.parameters, None, notes, fixed={'ell[0]': 0.3}).to_dict()
    again = Witness.from_dict(json.loads(json.dumps(data)))
    assert data['notes'] == notes and again.notes == notes and again.fixed == {'ell[0]': 0.3}
    assert data['tree'] != 'x' and 'claimed_before' not in data['notes']


def test_reader_names_what_is_wrong():
    """The reader of witness files names what is wrong: a misspelled field, a digit that the tree does not have, a
    null, ℓ^(d) in "ell", a key that is not an index, a list where an object keyed by the index belongs, a free value
    under "derived" and a string as a number."""
    data = raw(os.path.join(WITNESSES, 'lwe_T0.5_d5_gf.json'))
    data['memory-bound'] = 0.1
    refused(data, 'memory_bound')
    data = raw(os.path.join(WITNESSES, 'lwe_T0.5_d5_gf.json'))
    digits = data['tree']['digits']
    J = data['parameters']['J']['2']
    rows = {str(a): {str(b): J[x][y] for y, b in enumerate(digits)} for x, a in enumerate(digits)}
    data['parameters']['J']['2'] = rows
    data['parameters']['J']['2']['0']['7'] = 0.01
    refused(data, '7')
    data = raw(os.path.join(WITNESSES, 'q11_d2_nn_half.json'))
    data['parameters']['J']['0'] = None
    refused(data, 'J[0] is null')                                # format 3 has no null
    data = raw(os.path.join(WITNESSES, 'q11_d2_nn_half.json'))
    data['parameters']['ell'].append(0.1)
    refused(data, 'ell_d')                                       # ℓ^(d) is not in "ell"
    data = raw(os.path.join(WITNESSES, 'q11_d2_nn_half.json'))
    data['parameters']['J']['4'] = data['parameters']['J']['2']
    refused(data, 'not an index')
    data = raw(os.path.join(WITNESSES, 'q11_d2_nn_half.json'))
    data['derived']['p'] = list(data['derived']['p'].values())
    refused(data, 'an object keyed by the index')
    data = raw(os.path.join(WITNESSES, 'q11_d1_mixed_half.json'))
    data['derived']['omega_1'] = data['parameters']['omega_1']
    refused(data, 'belongs under "parameters"')
    data = raw(os.path.join(WITNESSES, 'q11_d1_mixed_half.json'))
    data['parameters']['syndrome_filter']['u'] = str(data['parameters']['syndrome_filter']['u'])
    refused(data, 'not a number')


def test_shared_top_type_written_once():
    """The top type p[d], which the parts share, is written once.  With a free window type it is a free parameter,
    under "parameters" and not again as parts[k].p[d] under "derived".  Otherwise it is a derived value at the top
    level of "derived"."""
    problem, tree = Decoding(3, 0.5, 'gv/2'), Tree.same_joins(depth=2, filter=PriceFilter(), nonzero='free', parts=2)
    inst = build(problem, tree)
    P = from_model(inst, variables(inst).template)[0]
    data = json.loads(json.dumps(Witness(problem, tree, None, P).to_dict()))
    assert list(data['parameters']['p']) == ['2']
    assert all(list(part['p']) == ['0', '1'] for part in data['derived']['parts'].values()), data['derived']
    assert Witness.from_dict(data).evaluate().time == model.evaluate(inst, variables(inst).template).time
    derived = load('subsetsum_pm3_d6_two.json').to_dict()['derived']
    assert list(derived['p']) == ['6'] and all('6' not in part['p'] for part in derived['parts'].values())


def test_level_types_at_the_edges():
    """A Boltzmann type above the uniform mean does not overflow.  A maximum-entropy child type on the digits -2, 0, 2
    has one free share, of -2: the digit 0 takes the remainder and the digit 2 balances the mean.  The mean Lee
    weight of a Boltzmann type is searched up to q/2."""
    t = boltzmann_type(15, 41)
    assert abs(sum(t.values()) - 1) < 1e-12 and abs(sum(v * min(a, 41 - a) for a, v in t.items()) - 15) < 1e-9
    inst = build(LWE({-2: .2, 0: .5, 2: .3}), Tree.same_joins(depth=2, digits=(-2, 0, 2), joint_types='maxent'))
    template = variables(inst).template
    assert set(template['joins'][1]['child']['tree']) == {'-2'}          # 2 balances the mean, no digit 1
    assert math.isfinite(model.evaluate(inst, template).time)
    inst = build(LeeDecoding(4, 0.3, 1.8), Tree.same_joins(depth=2, joint_types='maxent', types='boltzmann'))
    assert max(slot.high for slot in variables(inst).slots if slot.path[-1] == 'mean') == 2


def test_command_line_goes_on_after_a_broken_file():
    """verify of a folder reports a file that is not JSON and goes on with the next one."""
    with tempfile.TemporaryDirectory() as folder:
        shutil.copy(os.path.join(WITNESSES, 'lwe_T0.5_d5_gf.json'), os.path.join(folder, 'good.json'))
        with open(os.path.join(folder, 'a_broken.json'), 'w') as handle:
            handle.write('{"format": "jointtrees witness 3", ')
        done = subprocess.run([sys.executable, '-m', 'jointtrees', 'verify', folder], cwd=HERE, capture_output=True,
                              text=True)
        assert done.returncode == 1 and 'not a readable JSON file' in done.stdout, done.stdout + done.stderr
        assert '1 of 2 witnesses verified' in done.stdout, done.stdout


def test_starts():
    """into_parts and with_digits keep the time of the start, and deeper gives parameters of the deeper tree."""
    w = load('lwe_T0.5_d5_gf.json')
    time = w.evaluate().time
    two = Tree.from_dict({**w.tree.to_dict(), 'parts': 2})
    assert abs(evaluate_at(w.problem, two, into_parts(w, sizes=[0.3]), w.memory_bound).time - time) < 1e-12
    wide = Tree.from_dict({**w.tree.to_dict(), 'digits': list(w.tree.digits) + [max(w.tree.digits) + 1]})
    start = with_digits(w, w.problem, wide)
    assert abs(evaluate_at(w.problem, wide, start).time - time) < 1e-12
    joins = [join.to_dict() for join in w.tree.joins]
    deep = Tree.from_dict({**w.tree.to_dict(), 'joins': joins + [joins[1]]})
    start = deeper(w, w.problem, deep)
    assert len(start.ell) == deep.depth and sorted(start.J) == list(range(1, deep.depth + 1))
    assert math.isfinite(evaluate_at(w.problem, deep, start).time)


def test_parameter_table():
    """Every problem and tree has a table of its parameters, and fixed names come from it.  ℓ^(d) is 'ell_d': free at
    an exact or mixed top join of a decoding problem, derived for subset sum, given by LWE and by a nearest-neighbour
    top join.  Every name appears once."""
    a = algorithms
    lee = Tree([Join(), Join(), Join(syndrome=LeeWeightFilter())], joint_types='maxent', types='boltzmann')
    for problem, tree in ((SubsetSum(), a.filtered(3, (-1, 0, 1), parts=2)), (Decoding(2, 0.42), a.mixed_top(2)),
                          (LeeDecoding(4, 0.46, 'gv/2'), lee), (Decoding(3, 0.37, 1.0), a.wagner(3, nonzero='free')),
                          (Decoding(3, 0.4, 0.1), Tree.same_joins(depth=2, nonzero='free'))):
        text = parameter_table(problem, tree)
        assert 'free' in text and 'ell[0]' in text, text
        names = [line.split()[0] for line in text.split('\n')[3:] if line.strip()]
        assert len(names) == len(set(names)), names
    for problem, tree, kind in ((Decoding(2, 0.42), a.mixed_top(2), 'free'), (SubsetSum(), a.bcj(2), 'derived'),
                                (LWE({-1: .25, 0: .5, 1: .25}), a.meet_lwe(2), 'given'),
                                (Decoding(2, 0.42), a.may_ozerov(2), 'given')):
        row = next(line.split() for line in parameter_table(problem, tree).split('\n') if line.startswith('ell_d '))
        assert row[1] == kind and ('ell_d' in parameter_names(problem, tree)) == (kind == 'free'), (tree, row)


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_') and name != 'test_witnesses' and callable(test):
            test()
            print(f'ok  {name}', flush=True)
    print(f'ok  test_witnesses: {test_witnesses()} witnesses verified')
