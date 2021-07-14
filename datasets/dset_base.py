
import numpy as np
from random import choice
from ..transforms import Compose



    
class BaseDataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    detail:
        if single input, tsf_i should be Compose() or None, if multi, tsf_i = {key1:Compose(),key2:Compose()}
        if single target,tsf_t should be Compose() or None, if multi, tsf_t = {key1:Compose(),key2:Compose()}
    """
    def set_tsform(self, tsf_i, tsf_t, tsf_all ):
        if tsf_i is None:
            self.tsf_input={key:(lambda x:x) for key in self.inputs_key}
        else:
            if len(self.inputs_key)==1: #just for we don't known key==''
                self.tsf_input={key:tsf_i for key in self.inputs_key}
            else:
                self.tsf_input={key:tsf_i[key] or (lambda y:y) for key in self.inputs_key}
        
        if self.targets!=None:
            if tsf_t is None:
                self.tsf_target={key:(lambda y:y) for key in self.targets_key}
            else:
                if len(self.targets_key)==1:
                    self.tsf_target={key:tsf_t for key in self.targets_key}
                else:
                    self.tsf_target={key:tsf_t[key] or (lambda y:y) for key in self.targets_key}
        
        self.tsf_all = tsf_all or (lambda *x:x if len(x)>1 else x[0])


    def __getitem__(self, index):
        
        if self.task_type=='cls' and self.dataaug_weight!=None:
            cls = choice(self.sample_cls)
            index = self.dict_idxs_cls[cls][self.dict_idxs_cur[cls]]
            self.dict_idxs_cur[cls] = 0 if (self.dict_idxs_cur[cls]+1)>=self.dict_idxs_end[cls] else self.dict_idxs_cur[cls]+1
        
        sample_input = [self.tsf_input[key](self.inputs[key][index]) for key in self.inputs_key]
        
        if self.targets!=None:
            sample_target = [self.tsf_target[key](self.targets[key][index]) for key in self.targets_key]
            sample_input.extend(sample_target)
            sample_data = self.tsf_all(sample_input)
            sample_input = sample_data[:len(self.inputs_key)]
            sample_target= sample_data[len(self.inputs_key):]
            return  np.array(self.inputs[self.inputs_key[0]][index]),   \
                    {key:sample_input[i] for i,key in enumerate(self.inputs_key)}, \
                    {key:sample_target[i] for i,key in enumerate(self.targets_key)}
        else:
            sample_input = self.tsf_all(sample_input)
            return  np.array(self.inputs[self.inputs_key[0]][index]),   \
                    {key:sample_input[i] for i,key in enumerate(self.inputs_key)}

    def __len__(self):
        return self.inputs[self.inputs_key[0]].shape[0]
    

    def add_tsf_input(self, tsf, is_add_front=True):
        if len(self.inputs_key)==1:
            tsf = {key:tsf for key in self.inputs_key}
        if is_add_front:
            self.tsf_input = {key:Compose([tsf[key],self.tsf_input[key]]) for key in self.inputs_key}
        else:
            self.tsf_input = {key:Compose([self.tsf_input[key],tsf[key]]) for key in self.inputs_key}

    def add_tsf_target(self, tsf, is_add_front=True):
        if len(self.targets_key)==1:
            tsf = {key:tsf for key in self.targets_key}
        if is_add_front:
            self.tsf_target = {key:Compose([tsf[key],self.tsf_target[key]]) for key in self.targets_key}
        else:
            self.tsf_target = {key:Compose([self.tsf_target[key],tsf[key]]) for key in self.targets_key}

    def add_tsf_all(self, tsf, is_add_front=True):
        if is_add_front:
            self.tsf_all = Compose([tsf, self.tsf_all])
        else:
            self.tsf_all = Compose([self.tsf_all, tsf])

    
    def load(self, samples_index=None):
        """
        Load all data or a subset of the data into actual memory.
        For instance, if the inputs are paths to image files, then this
        function will actually load those images.
        """
        samples_index = samples_index or range(len(self))
        list_filepath = []         
        dict_list_input = {key:[] for key in self.inputs_key}
        dict_list_target= {key:[] for key in self.targets_key}
        if self.targets==None:
            for index in samples_index:
                filepath, dict_inputs = self.__getitem__(index)
                list_filepath.append(filepath)
                for key in self.inputs_key:
                    dict_list_input[key].append(dict_inputs[key])
            return np.array(list_filepath,axis=0), \
                    {key:np.stack(dict_list_input[key]) for key in self.inputs_key}
        else:
            for index in samples_index:
                filepath, dict_inputs, dict_targets = self.__getitem__(index)
                list_filepath.append(filepath)
                for key in self.inputs_key:
                    dict_list_input[key].append(dict_inputs[key])
                for key in self.targets_key:
                    dict_list_target[key].append(dict_targets[key])
            return np.array(list_filepath),	\
                    {key:np.stack(dict_list_input[key]) for key in self.inputs_key}, \
                    {key:np.stack(dict_list_target[key]) for key in self.targets_key}


    def reset(self, is_shuffle, num_cls=-1):
        self.num_total = len(self)
        assert( self.num_total>0 )
        self.read_idx = 0
        self.read_EOF = False

        list_indexs = list(range(self.num_total))
        if is_shuffle:
            np.random.shuffle(list_indexs)
        self.inputs = {key:self.inputs[key][list_indexs] for key in self.inputs_key}
        if self.targets!=None:
            self.targets = {key:self.targets[key][list_indexs] for key in self.targets_key}

        if self.task_type=='cls' and self.dataaug_weight!=None:
            assert(num_cls!=-1)
            self.dict_idxs_cls = {cls:[] for cls in range(num_cls)}
            for idx,cls in enumerate(self.targets['']):
                self.dict_idxs_cls[cls].append(idx)
            
            self.sample_cls = []              
            for cls,w in enumerate(self.dataaug_weight):
                for j in range(w):
                    self.sample_cls.append(cls)
            self.dict_idxs_cur = {cls:0 for cls in range(num_cls)}
            self.dict_idxs_end = {cls:len(self.dict_idxs_cls[cls]) for cls in range(num_cls)}
        

    def next_batch(self, batch_size):
        idx_end = self.read_idx+batch_size
        if idx_end>self.num_total:
            batch_idxs = list( range(self.read_idx,self.num_total,1) )
            idxs_recur = range(batch_size-len(batch_idxs))
            batch_idxs.extend(idxs_recur)
            self.read_idx = len(idxs_recur)
            self.read_EOF = True
        else:
            batch_idxs = range(self.read_idx, idx_end, 1)
            self.read_idx = idx_end
            self.read_EOF = False
            
        return self.load(batch_idxs)
        
