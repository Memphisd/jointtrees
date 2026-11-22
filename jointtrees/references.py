"""Where the paper states what the code implements.  The only place with the paper's numbers: when the numbering of
the paper changes, change it here (and run tools/rules_table.py, which copies it into RULES.md).  The code and the
documentation name these places in brackets, e.g. [saturation]."""

PAPER = {                                   # the paper's numbers (LaTeX label in the comment)
    'lemma 1':          'Lemma 1',          # lem:model: (a) representations r = H(J) - H(P), (b) loss c = I(X;Y)
    'largest entropy':  'Section 3.1',      # sec:tree: the standard join keeps the joint type of largest entropy
    'dissection':       'Eq. (20)',         # eq:dissection: base lists by the 7-dissection
    'list sizes':       'Eq. (3)',          # eq:lists
    'saturation':       'Eq. (4)',          # eq:sat
    'budget':           'Eq. (5)',          # eq:budget
    'repetitions':      'Eq. (6)',          # eq:reps: subtrees repeated when representations are missing [EZ23]
    'plain join':       'Eq. (7)',          # eq:plainjoin
    'tree cost':        'Eq. (8)',          # eq:treecost
    'center 1/2':       'Eq. (19)',         # eq:oppcutfree with g = 1: opposite tilts give centers of size 1/2
    'admissible':       'Eq. (11)',         # eq:genadmissible: alpha as a minimum over all admissible fractions
    'closed form':      'Lemma 2',          # lem:genclosed: the most likely fractions theta = sigmoid2(c + a_i + b_j)
    'filter exponents': 'Eqs. (10), (14)',  # eq:genbeta, eq:genalpha: alpha, beta for a center of any size
    'filtered join':    'Eq. (15)',         # eq:gencost: the cost of the filtered join
    'filter blocks':    'Theorem 2',        # thm:genjoin (Appendix C): product centers on blocks of positions
    'partitioning':     'Section 4.6',      # sec:severalparts: filters on parts
    'simplifications':  'Section 4.4',      # sec:simplify: same and opposite tilt, prices, the factor g
    'main theorem':     'Theorem 1',        # thm:main: subset sum with the digits -3..3, depth 6, two parts
    'alphabets table':  'Table 2',          # tab:alphabets: subset sum by digit set and depth
    'isd':              'Section 7.1',      # sec:isdtree: window, permutations, top join
    'neighbour join':   'Section 7.2',      # sec:fqnn
    'decoders table':   'Table 6',          # tab:fqisd: the decoders over F_q
    'many solutions':   'Appendix F',       # sec:ternary: top-list cap, number of runs
    'lee metric':       'Appendix H',       # sec:lee: Boltzmann types, the Lee syndrome filter
    'lee table':        'Table 11',         # tab:lee
}

LITERATURE = {
    'EZ23': 'Esser, Zweydinger: New time-memory trade-offs for subset sum (Eurocrypt 2023): 7-dissection base lists, '
            'subtree repetitions',
    'MO15': 'May, Ozerov: On computing nearest neighbors with applications to decoding of binary linear codes '
            '(Eurocrypt 2015)',
    'BJMM12': 'Becker, Joux, May, Meurer: Decoding random binary linear codes in 2^(n/20) (Eurocrypt 2012)',
    'BKW22': 'Bariffi, Khathuria, Weger: Information set decoding for Lee-metric codes using restricted balls '
             '(arXiv 2205.12903, 2022)',
    'BBPSWW23': 'Baldi, Bitzer, Pavoni, Santini, Wachter-Zeh, Weger: Generic decoding of restricted errors (ISIT 2023)',
}


def ref(name):
    return PAPER[name]
