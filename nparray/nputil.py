import numpy as np
import os
from ..basic import readdir, mkdir



def img2array(dict_img):
    dict_array = dict()
    for name, img in dict_img.items():
        array = np.asarray(img).astype(np.uint8)
        if len(array.shape) == 2:
            dict_array[name] = array.reshape( (array.shape[0],array.shape[1],1) )
        else:
            dict_array[name] = array[...,:3]
    return dict_array


def arrs2dict(arrs, listkey=None):
    dict_dst= dict()
    for i in range(arrs.shape[0]):
        if listkey==None:
            dict_dst[i] = arrs[i]
        else:
            dict_dst[listkey[i]] = arrs[i]
    return dict_dst

def save_npy(dict_arr, save_dir):
    mkdir(save_dir)
    for key, arr in dict_arr.items():
        np.save(os.path.join(save_dir,key), arr)

def load_npy( npdir ):
    dict_arr = dict()
    if isinstance(npdir, list):
        for file in npdir:
            dict_arr[os.path.split(file)[1].strip('.npy')] = np.load(file)
    else:
        files = readdir(npdir)
        for file in files:
            dict_arr[os.path.split(file)[1].strip('.npy')] = np.load(os.path.join(npdir,file))

    return dict_arr

    
def npround(dict_arr, roundbit):
    dict_dst = dict()
    for key,arr in dict_arr.items():
        dict_dst[key] = np.round(arr, roundbit)
    return dict_dst


def np2bw(dict_arr, threshold):
    dict_dst = dict()
    for key, arr in dict_arr.items():
        dict_dst[key] = (arr>=threshold).astype(np.int)
    return dict_dst

def divideValue(dict_src, value=255.):
    dict_dst = dict()
    for name, img in dict_src.items():
        arr = np.asarray(img).astype(np.uint8)/value
        dict_dst[name] = arr
    return dict_dst



from skimage import measure

def thresh_area(dict_img, thresh_area):
    def thresh_connect(imarr, thresh_area):
        labelArray = measure.label(imarr, background=0, connectivity=2)
        label = 1
        while (label <= labelArray.max()):
            curarray = (labelArray==label).astype(int)
            if np.sum(curarray)<thresh_area:
                imarr = imarr*(1-curarray)
            label += 1
        return imarr
    
    dict_dst = dict()
    for name, img in dict_img.items():
        imarr = np.array(img)
        if np.max(imarr)>1:
            imarr = imarr/np.max(imarr)
        assert len(np.unique(imarr).tolist())<=2
        dict_dst[name] = thresh_connect(imarr, thresh_area)
    return dict_dst

def twogroOr(dict_gro1, dict_gro2, thresh1=0, thresh2=0):
    dict_dst = {}
    for key in dict_gro1:
        dict_dst[key] = ( (dict_gro1[key]>thresh1) | (dict_gro2[key]>thresh2) ).astype(np.uint8)
    return dict_dst

def twogroAnd(dict_gro1, dict_gro2, thresh1=0, thresh2=0):
    dict_dst = {}
    for key in dict_gro1:
        dict_dst[key] = ( (dict_gro1[key]>thresh1) & (dict_gro2[key]>thresh2) ).astype(np.uint8)
    return dict_dst