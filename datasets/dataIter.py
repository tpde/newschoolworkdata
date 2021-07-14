from scipy.ndimage import rotate, zoom
from random import choice
import numpy as np
import random


class DataIter(object):
    """ multitask(zr)
    """
    def __init__(self):
        self.dict_train = None
        self.dict_valid = None
        self.dict_test = None
        
        self.trainEOF=True
        self.validEOF=True
        self.testEOF =True
        

    def set_trainset(self, dict_arr, val_split=0, is_shuffle=False):
        self.trainkey = list(dict_arr.keys())
        self.dict_train = dict_arr
        self.validkey = list(dict_arr.keys()) if val_split>0 else None
        self.dict_valid = dict_arr if val_split>0 else None
        
        self.num_total = self.dict_train[ self.trainkey[0] ].shape[0]
        self.num_train = int( self.num_total*(1-val_split) )
        self.num_valid = (self.num_total-self.num_train) if val_split>0 else 0

        list_idx = list(range(self.num_total))
        if is_shuffle:
            np.random.shuffle(list_idx)
        self.list_idx_train = list_idx[:self.num_train]
        self.list_idx_valid = list_idx[self.num_train:]

        self.idx_train_start = 0
        self.idx_valid_start = 0
        
        self.trainEOF=False if self.num_train>0 else True
        self.validEOF=False if self.num_valid>0 else True

    def set_validset(self, dict_arr):
        self.validkey = list(dict_arr.keys())
        self.dict_valid= dict_arr

        self.num_valid = self.dict_valid[ self.validkey[0] ].shape[0]
        self.list_idx_valid = list(range(self.num_valid))
        self.idx_valid_start = 0
        
        self.validEOF=False if self.num_valid>0 else True

    def set_testset(self, dict_arr):
        self.testkey = list(dict_arr.keys())
        self.dict_test = dict_arr

        self.num_test = self.dict_test[ self.testkey[0] ].shape[0]
        list_idx = list(range(self.num_test))
        self.list_idx_test = list_idx
        self.idx_test_start = 0
        
        self.testEOF=False if self.num_test>0 else True
        
        
    def next_train(self, batch_size, is_havewrong=False):
        assert self.dict_train is not None
        dict_batch = dict()
        
        idx_end = self.idx_train_start+batch_size
        if idx_end>self.num_train:
            batch_idxs = self.list_idx_train[self.idx_train_start:]
            idxs_recur = self.list_idx_train[:batch_size-len(batch_idxs)]
            batch_idxs.extend(idxs_recur)
            self.idx_train_start = len(idxs_recur)
            self.trainEOF=True
        else:
            batch_idxs = self.list_idx_train[self.idx_train_start:idx_end]
            self.idx_train_start = idx_end
            self.trainEOF=False
            
        for key in self.trainkey:
            dict_batch[key] = self.dict_train.get(key)[batch_idxs]
            
        if is_havewrong:
            dict_wrong = dict()
            wrong_idxs = random.sample( list(set(self.list_idx_train)-set(batch_idxs)), batch_size)
            for key in self.trainkey:
                dict_wrong[key] = self.dict_train.get(key)[wrong_idxs]
            return dict_batch, dict_wrong
        else:
            return dict_batch


    def next_valid(self, batch_size, is_havewrong=False):
        assert self.dict_valid is not None
        dict_batch = dict()
        
        idx_end = self.idx_valid_start+batch_size
        if idx_end>self.num_valid:
            batch_idxs = self.list_idx_valid[self.idx_valid_start:]
            idxs_recur = self.list_idx_valid[:batch_size-len(batch_idxs)]
            batch_idxs.extend(idxs_recur)
            self.idx_valid_start = len(idxs_recur)
            self.validEOF=True
        else:
            batch_idxs = self.list_idx_valid[self.idx_valid_start:idx_end]
            self.idx_valid_start = idx_end
            self.validEOF=False
            
        for key in self.validkey:
            dict_batch[key] = self.dict_valid[key][batch_idxs]
        
        if is_havewrong:
            dict_wrong = dict()
            wrong_idxs = random.sample( list(set(self.list_idx_valid)-set(batch_idxs)), batch_size)
            for key in self.validkey:
                dict_wrong[key] = self.dict_valid.get(key)[wrong_idxs]
            return dict_batch, dict_wrong
        else:
            return dict_batch

    def next_test(self, batch_size, is_havewrong=False):
        assert self.dict_test is not None
        dict_batch = dict()
        
        idx_end = self.idx_test_start+batch_size
        if idx_end>self.num_test:
            batch_idxs = self.list_idx_test[self.idx_test_start:]
            idxs_recur = self.list_idx_test[:batch_size-len(batch_idxs)]
            batch_idxs.extend(idxs_recur)
            self.idx_test_start = len(idxs_recur)
            self.testEOF=True
        else:
            batch_idxs = self.list_idx_test[self.idx_test_start:idx_end]
            self.idx_test_start = idx_end
            self.testEOF=False
        
        for key in self.testkey:
            dict_batch[key] = self.dict_test[key][batch_idxs]
        
        if is_havewrong:
            dict_wrong = dict()
            wrong_idxs = random.sample( list(set(self.list_idx_test)-set(batch_idxs)), batch_size)
            for key in self.testkey:
                dict_wrong[key] = self.dict_test.get(key)[wrong_idxs]
            return dict_batch, dict_wrong
        else:
            return dict_batch


    def dataaug(self, dict_arrs, dict_arrs_, dict_is_noise, angle=None, is_random=True):
        num, h, w, _= list(dict_arrs.items())[0][1].shape
        for idx in range(num):
            if is_random:
                r = random.random()/5
                ih = np.random.choice(max(int(r*h),1), 1)[0]
                iw = np.random.choice(max(int(r*w),1), 1)[0]
                if choice(range(2)):
                    for key in dict_arrs.keys():
                        dict_arrs_[key][idx] = zoom(dict_arrs.get(key)[idx,:,::-1,:], [r+1,r+1,1], mode='nearest')[ih:ih+h, iw:iw+w, :]
                else:
                    for key in dict_arrs.keys():
                        dict_arrs_[key][idx] = zoom(dict_arrs.get(key)[idx], [r+1,r+1,1], mode='nearest')[ih:ih+h, iw:iw+w, :]
            else:
                for key in dict_arrs.keys():
                    dict_arrs_[key][idx,:] = dict_arrs.get(key)[idx,:]
                
            if angle == None:
                angle = choice(range(360))
            for key in dict_arrs.keys():
                dict_arrs_[key][idx] = rotate(dict_arrs_.get(key)[idx], angle, axes=(0, 1), reshape=False)
        
        for key,is_noise in dict_is_noise.items():
            if is_noise:
                dict_arrs_[key] += np.random.normal(0, 0.2, size=dict_arrs_[key].shape)
            else:
                dict_arrs_[key] = np.round(dict_arrs_[key])
                
        
    def normalize(self, arr_src, arr_dst):
        # arr_dst = np.zeros(arr_src.shape, np.float)
        arr_src = arr_src.astype(np.float)
        for idx in range(arr_src.shape[0]):
            mean=np.mean(arr_src[idx,...][ arr_src[idx,...,0]>0 ], axis=0)
            std =np.std(arr_src[idx,...][ arr_src[idx,...,0]>0 ], axis=0)
            assert len(mean)<=10 and len(std)<=10
            arr_dst[idx,...]=(arr_src[idx,...]-mean)/std

        