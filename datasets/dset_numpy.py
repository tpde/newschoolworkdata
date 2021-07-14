
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from .dset_base import BaseDataset


class NumpyDataset(BaseDataset):

    def __init__(self, task_type,
                 input, target=None,
                 transform_input=None, 
                 transform_target=None,
                 transform_all=None,
                 dataaug_weight=None,
                 inputs_key = None):
        """
        Argument
            input: it's shape can be like
                [num,48,48,3]
                [ [num,48,48,3], [num,48,48,1] ]
                {'ori1':[num,48,48,3], 'ori2':[num,48,48,1] }
            target: it's shape can be like
                if cls, [num] or [num, 3]
                if seg, [ [num,48,48,1], [num,48,48,1] ]
                    or {'gro1':[num,48,48,3], 'gro2':[num,48,48,1] }
        """
        assert(task_type in ['cls', 'seg'])
        self.task_type = task_type
        
        if isinstance(input, np.ndarray):
            input = [input]
        if isinstance(input, list):
            self.inputs_key = list( range(len(input)) )
        elif isinstance(input, dict):
            self.inputs_key = inputs_key if inputs_key else list(input.keys())
        else:
            raise( TypeError )
        self.inputs = {key:np.array(input[key]) for key in self.inputs_key}
        

        if isinstance(target, np.ndarray):
            target = [target]
        if isinstance(target, list):
            self.targets_key = list( range(len(target)) )
        elif isinstance(target, dict):
            self.targets_key = list(target.keys())
        else:
            self.targets = None
        self.targets = {key:np.array(target[key]) for key in self.targets_key}
        
        self.dataaug_weight = dataaug_weight
        self.set_tsform(transform_input, transform_target, transform_all )




