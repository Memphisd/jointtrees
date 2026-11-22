"""The problems: one class per problem (base.py explains what a problem provides).

    Problem               the base: the tree enumerates the whole vector, every label unit is matched exactly
      SubsetSum           random subset sum
      LWE                 small LWE keys (Meet-LWE)
      Decoding            syndrome decoding over F_q, Hamming metric (window, syndrome part, permutations)
        TypedDecoding     decoding with an error of a given type over Z_q
          RestrictedDecoding   the non-zero entries of the error in a set E
          LeeDecoding          Lee metric
"""
from .base import Problem, problem_from_dict
from .subset_sum import SubsetSum
from .lwe import LWE
from .decoding import Decoding
from .typed import TypedDecoding
from .restricted import RestrictedDecoding
from .lee import LeeDecoding

__all__ = ['Problem', 'problem_from_dict', 'SubsetSum', 'LWE', 'Decoding', 'TypedDecoding', 'RestrictedDecoding',
           'LeeDecoding']
