# -*- coding: utf-8 -*-
import numpy as np
from ..basic import *
from .cutslice import calculate_gridcut


def stack_array( type_array, list_key=None):
    # array: list_dict or dict
    def stack_arr(dict_arr, list_key):
        list_arr = []
        for i in range(len(list_key)):
            arr_imgs = dict_arr[list_key[i]]
            if len(arr_imgs.shape)==2:
                list_arr.append( arr_imgs[np.newaxis,...,np.newaxis] )
            elif len(arr_imgs.shape)==3:
                list_arr.append( arr_imgs[np.newaxis,...] )
            else:
                list_arr.append( arr_imgs )
        return np.concatenate(list_arr, axis=0)

    if isinstance(type_array, list):
        list_key = list(type_array[0].keys()) if list_key==None else list_key
        list_array = []
        for dict_arr in type_array:
            list_array.append(stack_arr(dict_arr, list_key))
        return list_array, list_key
    elif isinstance(type_array, dict):
        list_key = list(type_array.keys()) if list_key==None else list_key
        return stack_arr(type_array, list_key), list_key
    else:
        raise( TypeError )
    

def puzzle_slices(slices, cols):
    # data = np.transpose(data,(0,2,3,1))  #corect format for imshow
    assert slices.shape[0]%cols==0
    if len(slices.shape)==4:
        assert (slices.shape[3]==1 or slices.shape[3]==3)
    else:
        slices = slices.reshape((slices.shape[0],slices.shape[1],slices.shape[2],1))
    rows = int(slices.shape[0]/cols)
    slice_h, slice_w, channel = slices.shape[1:]
    imgArray = np.zeros((slice_h*rows, slice_w*cols, channel))
    for row in range(rows):
        for col in range(cols):
            imgArray[row*slice_h:(row+1)*slice_h,col*slice_w:(col+1)*slice_w] = slices[row*cols+col]
    return imgArray

def puzzel_image(dict_img, slices, stride_h, stride_w, dict_name=None):
    dict_imgArray = dict()
    if len(slices.shape) == 3:
        slices = slices.reshape( (slices.shape[0], slices.shape[1], slices.shape[2], 1))
    slice_h, slice_w, slice_channel = slices.shape[1:]
    slice_one = np.ones( (slice_h, slice_w, slice_channel) )
    count_used = 0
    if dict_name == None:
        dict_name = dict()
        for i in range(len(dict_img.keys())):
            dict_name[i] = i
    for i in range(len(dict_name)):
        img_h, img_w = dict_img[dict_name[i]].shape[:2]
        
        img_newH, img_newW, cutNum_h, cutNum_w, lastpart_validHeight, lastpart_validWidth = calculate_gridcut( 
                    img_h, img_w, slice_h, slice_w, stride_h, stride_w)
        
        slices_array = slices[count_used:count_used+cutNum_h*cutNum_w]
        count_used += cutNum_h*cutNum_w
        imgArray_predict_extend = np.zeros((img_newH, img_newW, slice_channel))
        imgArray_addTimes_extend = np.zeros((img_newH, img_newW, slice_channel))
        
        for row in range(cutNum_h):
            for col in range(cutNum_w):
                slice_index = row*cutNum_w+col
                imgArray_predict_extend[row*stride_h:row*stride_h+slice_h, col*stride_w:col*stride_w+slice_w] += slices_array[slice_index] 
                imgArray_addTimes_extend[row*stride_h:row*stride_h+slice_h, col*stride_w:col*stride_w+slice_w] += slice_one
        imgArray_predict_extend = imgArray_predict_extend/imgArray_addTimes_extend
        
        dict_imgArray[dict_name[i]] = imgArray_predict_extend[:img_h,:img_w]
    return dict_imgArray

def puzzel_label(dict_img, slices_label, slice_h, slice_w, stride_h, stride_w):
    assert slices_label.shape[1]>=2
    num_class = slices_label.shape[1]
    
    dict_imgArray = dict()
    label = np.zeros((len(dict_img),num_class))
    count_used = 0
    slice_one = np.ones( (slice_h, slice_w) )
    for idx in range(len(dict_img)):
        img_h, img_w = dict_img[idx].shape[:2]
        
        img_newH, img_newW, cutNum_h, cutNum_w, lastpart_validHeight, lastpart_validWidth = calculate_gridcut( 
                    img_h, img_w, slice_h, slice_w, stride_h, stride_w)
        
        slices_array = slices_label[count_used:count_used+cutNum_h*cutNum_w]
        count_used += cutNum_h*cutNum_w
        imgArray_predict_extend = np.zeros((img_newH, img_newW, num_class))
        imgArray_addTimes_extend = np.zeros((img_newH, img_newW, num_class))
    
        for row in range(cutNum_h):
            for col in range(cutNum_w):
                slice_index = row*cutNum_w+col
                for idx_class in range(num_class):
                    imgArray_predict_extend[row*stride_h:row*stride_h+slice_h, col*stride_w:col*stride_w+slice_w, idx_class] += slices_array[slice_index, idx_class] 
                    imgArray_addTimes_extend[row*stride_h:row*stride_h+slice_h, col*stride_w:col*stride_w+slice_w, idx_class] += slice_one
        imgArray_predict_extend = imgArray_predict_extend/imgArray_addTimes_extend
        
        dict_imgArray[idx] = imgArray_predict_extend[:img_h,:img_w]
        for idx_class in range(num_class):
            label[idx,idx_class] = np.mean(dict_imgArray[idx][:,:,idx_class])
    return dict_imgArray, label, count_used
