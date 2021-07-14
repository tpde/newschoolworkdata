
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
from .dset_base import BaseDataset


class FolderDataset(BaseDataset):

    def __init__(self, input, target=None,
                 transform_input=None, 
                 transform_target=None,
                 transform_all=None,
                 dataaug_weight=None):
        """
        description: Dataset class for loading out-of-memory data.

        Arguments
            input:
                if cls: 'data/train'
                        the subfile should be dir
                if seg: enum{'data/train/input',
                            ['data/train/ori1', 'data/train/ori2'],
                            {'ori1':'data/train/ori1','ori2':'data/train/ori2'}}
                        the subfile should be file;
            target:
                if cls: None, the subdir will be label 
                if seg: enum{None,
                            'data/train/gro',
                            ['data/train/gro1', 'data/train/gro2'],
                            {'gro1':'data/train/gro1'}}
                        None means no groundtruth;
        """
        if isinstance(input, str):
            subpath = os.path.join(input, os.listdir(input)[0])
            if os.path.isdir( subpath ):
                self.task_type='cls'
                if os.path.isdir( os.path.join(subpath, os.listdir(subpath)[0]) ):
                    self.inputs_key = os.listdir(subpath)
                    for sdir in os.listdir(input):
                        assert( os.listdir(os.path.join(input,sdir))==self.inputs_key )
                else:
                    self.inputs_key = ['']
                self.inputs = {key:[] for key in self.inputs_key}
                self.targets_key = ['']
                self.targets = {'':[]}
                for subdir in os.listdir(input):
                    assert( int(subdir) in range(len(os.listdir(input))) )
                    subpath = os.path.join(input,subdir)
                    assert( len(set( [len(os.listdir(os.path.join(subpath,ssdir))) for ssdir in self.inputs_key] ))==1)
                    
                    for file in os.listdir( os.path.join(subpath,self.inputs_key[0]) ):
                        for key in self.inputs_key:
                            filepath = os.path.join(subpath,key,file)
                            assert( os.path.exists(filepath) )
                            self.inputs[key].append(filepath)
                        self.targets[''].append(int(subdir))
                
                self.dataaug_weight = dataaug_weight
                
            else:
                input = [input]
        
        if isinstance(input, (tuple,list,dict)):
            self.task_type='seg'
            self.inputs_key=list(input.keys()) if isinstance(input,dict) else range(len(input))
            self.inputs = {key:[] for key in self.inputs_key}
            assert len(set( [len(os.listdir(input[key])) for key in self.inputs_key]))==1
            for file in os.listdir(input[self.inputs_key[0]]):
                for key in self.inputs_key:
                    fpath = os.path.join(input[key],file)
                    assert( os.path.exists(fpath) )
                    self.inputs[key].append(fpath)

            if target==None:
                self.targets=None
            else:
                if isinstance(target, str):
                    target = [target]
                if isinstance(target, (tuple,list,dict)):
                    self.targets_key=list(target.keys()) if isinstance(target,dict) else range(len(target))
                else:
                    raise( 'argument type error: please check the target argument!')
                self.targets = {key:[] for key in self.targets_key}
                
                assert( set([len(os.listdir(target[key])) for key in self.targets_key])==set([len(self.inputs[self.inputs_key[0]])]) )
                for filepath in self.inputs[self.inputs_key[0]]:
                    file = os.path.split(filepath)[1]
                    for key in self.targets_key:
                        fpath = os.path.join(target[key],file)
                        assert(os.path.exists(fpath))
                        self.targets[key].append(fpath)

        self.inputs = {key:np.array(self.inputs[key]) for key in self.inputs_key}
        if self.targets!=None:
            self.targets = {key:np.array(self.targets[key]) for key in self.targets_key}
        assert(self.task_type in ['cls', 'seg'])
        
        self.set_tsform(transform_input, transform_target, transform_all )







