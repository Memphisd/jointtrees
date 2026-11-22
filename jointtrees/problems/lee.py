"""Decoding in the Lee metric.  Standard library only."""
from ..families import Variable
from ..lee import boltzmann_type, lee_gv, lee_weight
from .typed import TypedDecoding


class LeeDecoding(TypedDecoding):
    """Syndrome decoding in the Lee metric over Z_q, any q >= 2 [BKW22]: e has Lee weight δ n, and its type is the
    Boltzmann type of that relative Lee weight (lee.py, the paper's δ, the mean Lee weight per position) [lee metric].

    weight   δ: a number, 'gv' (the Lee Gilbert-Varshamov weight of the rate) or 'gv/2' (half of it).  `weight`
             keeps the argument as given, `delta` is δ as a number, and `omega` (ω, from TypedDecoding) is the share
             of the non-zero digits, which differs from δ.

    The digits a and -a are alike: types, joint types and the prices of the syndrome filter are given per class
    {a, -a}.  A window filter has one price per digit.  One Tree option exists only here:
    types='boltzmann' (the window and every level carry a Boltzmann type, one mean Lee weight each).
    """
    arguments = ('q', 'rate', 'weight')
    ring = True                     # the Lee metric lives on Z_q: any q >= 2 (e.g. Z_4, Z_49)

    def __init__(self, q, rate, weight='gv'):
        if not (weight in ('gv', 'gv/2') or (isinstance(weight, (int, float)) and 0 < weight <= q // 2)):
            raise ValueError(f"weight: the mean Lee weight per position, a number in (0, {q // 2}], 'gv' or 'gv/2'")
        if weight == 'gv':
            T = lee_gv(rate, q)
        elif weight == 'gv/2':
            T = lee_gv(rate, q) / 2
        else:
            T = weight
        super().__init__(q, rate, boltzmann_type(T, q))
        self.weight, self.delta = weight, T                            # δ, the relative Lee weight

    def rules(self, inst):
        """The digit classes {a, -a}, then the decoding rules."""
        q = self.q
        if len(inst.digits) < q:                                    # Boltzmann types on a subset of Z_q
            inst.lee_cutoff = max(lee_weight(a, q) for a in inst.digits)
        inst.class_of = {a: min(a, (-a) % q) for a in inst.digits}
        lines = ['symmetry: the digits a and -a are alike, and types, joint types and syndrome prices are given per '
                 'class {a, -a}; a window filter has one price per digit']
        return lines + super().rules(inst)

    def boltzmann(self, mean, inst):
        """The Boltzmann type with this mean Lee weight, on the digits of the lists."""
        t = boltzmann_type(mean, self.q, inst.lee_cutoff)
        return {a: t.get(a, 0.0) for a in inst.digits}

    def paper_name(self, inst, path):
        """With Boltzmann types the window type is given by its relative Lee weight."""
        if path == ('window', 'mean'):
            return ('lee_weight', inst.depth), 1.0
        return super().paper_name(inst, path)

    def window_variables(self, inst, template):
        """With Boltzmann types the window type is a Boltzmann type, given by its mean Lee weight."""
        if inst.tree.types != 'boltzmann':
            return super().window_variables(inst, template)
        mean = sum(share * lee_weight(a, self.q) for a, share in inst.root_type.items())
        template['window'] = {'mean': 0.8 * mean}
        return [Variable(('window', 'mean'), 0.0, self.q // 2)]

    def window_type(self, inst, prm, size):
        if inst.tree.types != 'boltzmann':
            return super().window_type(inst, prm, size)
        root = self.boltzmann(prm['window']['mean'], inst)
        return root, size * (1 - root[0])
