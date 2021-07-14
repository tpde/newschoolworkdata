#! -*- coding:utf-8 -*-
import cv2
import numpy as np
from skimage import measure
from ..imglib import read_images
from ..nparray import load_npy, groZeroOne, np2bw, npround, img2array
from ..tools import delkeytail, unique_list, mprintdict
from ..basic import *
from .util import TFPN2param

def get_evalthresh(dict_arr, num_points=300):
    dict_arr = npround(dict_arr, 3)
    dict_count = dict()
    for key,arr in dict_arr.items():
        list_value = np.unique(arr)
        for value in list_value:
            if value in dict_count.keys():
                dict_count[value] += np.sum((arr==value).astype(np.int))
            else:
                dict_count[value] = np.sum((arr==value).astype(np.int))
    # 将 dict_count 按照 value(点的数量） 从大到小排序
    list_pair = sorted(dict_count.items(), key=lambda item:item[1], reverse=True)
    list_thresh = [float(pair[0]) for pair in list_pair]
    return list_thresh[:num_points]

def evaluate( gros_dir, pres_dir, num_points=300, evalmode='area', thresh_overlap=0.199999):
    """
    gros_dir: image directory of groundtruth
    pres_dir: numpy directory of prediction
    """
    assert evalmode in ['area', 'pixel']
    dict_pre = load_npy(pres_dir)
    list_thresh = [thresh/100. for thresh in range(1,101,1)]
    list_thresh.extend(get_evalthresh(dict_pre, num_points))
    list_thresh = unique_list(list_thresh)
    list_thresh = sorted(list_thresh, key=lambda x:x, reverse=False)
    sprint('list_thresh to eval: ',list_thresh)
    list_perthresh = list() # [{'thresh': , 'TP':0, 'FP':0, 'FN':0, 'TN':0}, ...]
    for threshold in list_thresh:
        sprint('threshold: ', threshold)
        dict_pre = load_npy(pres_dir)
        dict_pre = np2bw(dict_pre, threshold)
        
        dict_gro = groZeroOne(read_images(gros_dir))
        
        dict_curthresh = {'thresh':threshold, 'TP':0, 'FP':0, 'TN':0, 'FN':0}
        for key in dict_pre.keys():
            pre = dict_pre.get(key)
            assert key in dict_gro.keys()
            gro = dict_gro.get(key)
            if evalmode=='area':
                dict_TFPN = evalpair_area(pre[:,:,0], gro[:,:,0], thresh_overlap)
            else:
                dict_TFPN = evalpair_pixel(pre[:,:,0], gro[:,:,0])
            
            for key,count in dict_TFPN.items():
                dict_curthresh[key] += count
        dict_param = TFPN2param(dict_curthresh)
        mprintdict(dict_param)
        list_perthresh.append(dict_curthresh)
        
    return list_perthresh



def evalpair_area(pre, gro, overlap_rate):
    TP_array = np.zeros(pre.shape)
    FP_array = np.zeros(pre.shape)
    FN_array = np.zeros(pre.shape)
    TN_array = np.zeros(pre.shape)

    labeled_pre = measure.label(pre, background=0, connectivity=2)
    labeled_gro = measure.label(gro, background=0, connectivity=2)

    TP_array += pre*gro
    
    gro_no = 1-gro
    label = 1
    while label <= labeled_pre.max():
        area = (labeled_pre==label).astype(np.int)
        if np.sum(area*gro)/(np.sum(area)+0.) > overlap_rate:
            TP_array += area*gro_no
        else:
            FP_array += area*gro_no
        label += 1
    
    pre_no = 1-pre
    label = 1
    while label <= labeled_gro.max():
        area = (labeled_gro==label).astype(np.int)
        if np.sum(area*pre)/(np.sum(area)+0.) > overlap_rate:
            TP_array += area*pre_no
        else:
            FN_array += area*pre_no
        label += 1
    
    TN_array = 1-TP_array-FP_array-FN_array
    assert len(np.unique(TP_array))<=2
    assert len(np.unique(TN_array))<=2
    assert len(np.unique(FP_array))<=2
    assert len(np.unique(FN_array))<=2
    dict_array = {'TP':TP_array, 'FP':FP_array, 'FN':FN_array, 'TN':TN_array}
    for key1,array1 in dict_array.items():
        for key2,array2 in dict_array.items():
            if key1 != key2:
                assert np.sum(array1*array2) == 0
    dict_TFPN = dict()
    for key,array in dict_array.items():
        dict_TFPN[key] = np.sum(array)
    return dict_TFPN


def evalpair_pixel(pre, gro):
    gro_no = 1-gro
    pre_no = 1-pre
    
    TP_array = pre*gro
    FP_array = gro_no*pre
    FN_array = gro*pre_no
    TN_array = gro_no*pre_no

    assert len(np.unique(TP_array))<=2
    assert len(np.unique(TN_array))<=2
    assert len(np.unique(FP_array))<=2
    assert len(np.unique(FN_array))<=2
    dict_array = {'TP':TP_array, 'FP':FP_array, 'FN':FN_array, 'TN':TN_array}
    for key1,array1 in dict_array.items():
        for key2,array2 in dict_array.items():
            if key1 != key2:
                assert np.sum(array1*array2) == 0
    dict_TFPN = dict()
    for key,array in dict_array.items():
        dict_TFPN[key] = np.sum(array)
    return dict_TFPN