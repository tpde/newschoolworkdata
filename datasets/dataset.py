import numpy as np
from os import listdir
from os.path import join, isdir


class Dataset():
    def __init__(self, datas, list_tsf, task_type='seg'):
        """
        Arguments
            task_type:
                enum in ['cls', 'seg']
            datas:
                task_type=='cls': 'data/train'
                task_type=='seg': enum{'data/train/',
                            {'ori1':'data/train/ori1','ori2':'data/train/ori2'}}
        """
        self.task_type = task_type
        if self.task_type=='cls':
            rdir = datas
            assert( isinstance(rdir, str) )
            for i,label in enumerate(listdir(rdir)):
                if i==0:
                    if isdir( listdir(join(rdir,label))[0] ):
                        self.list_key = listdir(join(rdir,label))
                    else:
                        self.list_key = ['',]
                    self.dict_data= {key:[] for key in self.list_key}
                    self.dict_data['label'] = []
                
                list_file = listdir(join(rdir,label,self.list_key[0]))
                for file in list_file:
                    for key in self.list_key:
                        self.dict_data[key].append( join(rdir,label,key,file) )
                self.dict_data['label'] += [int(label)]*len(list_file)
                
        else:
            if isinstance(datas, str):
                datas = {key:join(datas,key) for key in listdir(datas)}
            self.list_key = list( datas.keys() )
            self.dict_data= {key:[] for key in self.list_key}
            for file in listdir(datas[self.list_key[0]]):
                for key in self.list_key:
                    self.dict_data[key].append( join(datas[key],file) )
        
        for key in self.list_key:
            self.dict_data[key] = np.array(self.dict_data[key])
                
        for i,tsf in enumerate(list_tsf):
            list_tsf[i][0]= tsf[0] or [key for key in self.list_key if key!='label']
        self.list_tsf = list_tsf
        
        self.num_total = self.dict_data[self.list_key[0]].shape[0]
    
    
    def __len__(self):
        return self.num_total
    

    def reset(self, is_shuffle=False):
        num_total = len(self)
        assert( num_total>0 )
        self.read_idx = 0
        self.read_EOF = False
        
        if self.task_type=='cls':
            assert( is_shuffle == True )
            
        if is_shuffle:
            list_idxs = list(range(num_total))
            np.random.shuffle(list_idxs)
            for key in self.list_key:
                self.dict_data[key] = self.dict_data[key][list_idxs]


    def __getitem__(self, index):
        dict_data = {key:self.dict_data[key][index] for key in self.list_key}
        
        for tsf in self.list_tsf:
            processed = tsf[1]( [dict_data[key] for key in tsf[0]] )
            for i,key in enumerate( tsf[0] ):
                dict_data[key] = processed[i]
        return self.dict_data[self.list_key[0]][index], dict_data
    
    
    def load(self, indexs=None):
        indexs = indexs or range(len(self))
        list_path = []
        dict_batch = {key:[] for key in self.list_key}
        for index in indexs:
            filepath, dict_data = self.__getitem__(index)
            list_path.append(filepath)
            for key in self.list_key:
                dict_batch[key].append( np.array(dict_data[key]) )
                
        for key in self.list_key:
            arr = np.array( dict_batch[key] )
            dict_batch[key] = arr if len(arr.shape)==4 else arr[...,np.newaxis]
        
        return list_path, dict_batch


    def next_batch(self, batch_size):
        idx_end = self.read_idx+batch_size
        if idx_end > self.num_total:
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
