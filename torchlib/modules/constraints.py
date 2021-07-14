#! -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import torch as th

class Constraint(object):
    def __call__(self):
        raise NotImplementedError('Subclass much implement this method')


class UnitNorm(Constraint):
    """
    UnitNorm constraint.

    Constraints the weights to have column-wise unit norm
    """
    def __init__(self, 
                 frequency=1, 
                 unit='batch',
                 module_filter='*'):

        self.frequency = frequency
        self.unit = unit
        self.module_filter = module_filter

    def __call__(self, module):
        w = module.weight.data
        module.weight.data = w.div(th.norm(w,2,0))


class MaxNorm(Constraint):
    """
    MaxNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have a norm less than or equal to a desired value.

    Any hidden unit vector with a norm less than the max norm
    constaint will not be altered.
    """

    def __init__(self, 
                 value, 
                 axis=0, 
                 frequency=1, 
                 unit='batch',
                 module_filter='*'):
        self.value = float(value)
        self.axis = axis

        self.frequency = frequency
        self.unit = unit
        self.module_filter = module_filter

    def __call__(self, module):
        w = module.weight.data
        module.weight.data = th.renorm(w, 2, self.axis, self.value)


class NonNeg(Constraint):
    """
    Constrains the weights to be non-negative.
    """
    def __init__(self, 
                 frequency=1, 
                 unit='batch',
                 module_filter='*'):
        self.frequency = frequency
        self.unit = unit
        self.module_filter = module_filter

    def __call__(self, module):
        w = module.weight.data
        module.weight.data = w.gt(0).float().mul(w)






