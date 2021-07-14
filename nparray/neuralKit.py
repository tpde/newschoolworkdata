#! -*- coding:utf-8 -*-
import numpy as np
import scipy.misc
import random, cv2
from skimage import morphology,filters
from PIL import Image, ImageEnhance
from scipy.ndimage import rotate
from random import choice
from .nputil import arrs2dict


def dataaug( dict_arrs, dict_arrs_, dict_is_noise, noise_min=0.8, noise_max=1.2, angle=None):
    # dict_arrs_ = {'ori':[2,512,512,3], 'gro':[2,512,512,1]}
    for idx in range(list(dict_arrs.items())[0][1].shape[0]):
        if choice(range(2)):
            for key in dict_arrs.keys():
                dict_arrs_[key][idx] = dict_arrs.get(key)[idx,:,::-1,:]
        else:
            for key in dict_arrs.keys():
                dict_arrs_[key][idx] = dict_arrs.get(key)[idx]
        
        if angle == None:
            angle = choice(range(360))
        for key in dict_arrs.keys():
            dict_arrs_[key][idx] = rotate(dict_arrs_.get(key)[idx], angle, axes=(0, 1), reshape=False)
     

    for key,is_noise in dict_is_noise.items():
        if is_noise:
            arr_imgs = dict_arrs_.get(key)
            for idx in range(arr_imgs.shape[0]):
                arr_img = arr_imgs[idx]
                if np.max(arr_img)<=1:
                    arr_img = arr_img*255
                if len(arr_img.shape)==3 and arr_img.shape[2]==1:
                    arr_img = arr_img.reshape(arr_img.shape[:2])
                im=Image.fromarray(arr_img.astype(np.uint8))
                en=ImageEnhance.Color(im)
                im=en.enhance(random.uniform(noise_min,noise_max))
                arr_img=np.asarray(im).astype(np.float)
                dict_arrs_[key][idx]=arr_img.reshape((arr_img.shape[0], arr_img.shape[1], 1)) \
                                                     if len(arr_img.shape)==2 else arr_img
        else:
            dict_arrs_[key] = np.round(dict_arrs_[key])


def normalize( arr_src, arr_dst):
    arr_src = arr_src.astype(np.float)
    for idx in range(arr_src.shape[0]):
        mean=np.mean(arr_src[idx,...], axis=(0,1))
        std=np.std(arr_src[idx,...], axis=(0,1))
        assert len(mean)<=4 and len(std)<=4
        arr_dst[idx,...]=(arr_src[idx,...]-mean)/std

         
def groZeroOne(dict_src, is_shape4=True, is_bw=True):
    dict_dst = dict()
    for name, img in dict_src.items():
        arr = np.array(img).astype(np.uint8)
        if len(arr.shape) == 2 and is_shape4:
            arr = arr.reshape( (arr.shape[0],arr.shape[1],1) )
        if is_bw:
            if np.max(arr) > 1:
                arr = arr/np.max(arr)
            assert len(np.unique(arr))<=2
        else:
            if np.max(arr) > 1:
                arr = arr/255.
        dict_dst[name] = arr
    return dict_dst

def label2hot(label, num_cls):
    label = label.tolist()
    onehot = np.zeros( (len(label), num_cls), dtype=np.float )
    for idx, value in enumerate(label):
        onehot[idx,int(value)] = 1.0
    return onehot

def labeldehot(onehot):
    assert len(onehot.shape)>1
    _, cols = np.nonzero(onehot)
    print(np.array(cols))
    

def gro2hot(src, n_cls):
    assert( src.shape[-1]==1 )
    list_dim = list( src.shape[:-1] )
    assert( all([label in range(n_cls) for label in np.unique(src).tolist()]) )
    dst = np.zeros(list_dim + [n_cls])
    for i,cls in enumerate(range(n_cls)):
        dst[...,i:i+1][src==cls] = 1
    return dst


def thresh_otsu_dict(dict_arr, ratio=-1, dict_mask=None):
    dict_dst = dict()
    for key,arr in dict_arr.items():
        arr = np.array(arr)
        if dict_mask is None:
            thresh = filters.threshold_otsu(arr) if np.max(arr)>0 else arr
        else:
            mask = np.array(dict_mask[key])
            if len(mask.shape)==3:
                mask = mask[...,0]
            thresh = filters.threshold_otsu(arr[mask==1])
        dst = arr >= thresh
        if ratio!=-1:
            dst = morphology.remove_small_objects((dst>0), min_size=(arr.shape[0]*arr.shape[1]/ratio), connectivity=2)
        dict_dst[key] = dst.astype(np.int)
    return dict_dst

def thresh_otsu_arrs(arr_pres):
    if np.max(arr_pres)<=1:
        arr_bw = (arr_pres*255).astype(np.uint8)
    else:
        arr_bw = arr_pres.astype(np.uint8)
    for i,arr in enumerate(arr_bw):
        thresh = filters.threshold_otsu(arr) if np.max(arr_bw)>0 else 1
        arr_bw[i] = (arr >= thresh).astype(np.int)
    return arr_bw

def pixelinmask( gros, pres, masks):
    assert np.unique(gros).tolist()==[0,1]
    assert np.unique(masks).tolist()==[0,1]
    assert np.max(pres)<=1.0 and np.min(pres)>=0.0
    assert pres.shape==gros.shape and pres.shape==masks.shape
    return gros[masks==1].flatten(), pres[masks==1].flatten()

def compare(gros, pres, listkey=None, masks=None):
    gros = arrs2dict(gros, listkey) if not isinstance(gros, dict) else gros
    pres = arrs2dict(pres, listkey) if not isinstance(pres, dict) else pres

    dict_compare = dict()
    # if isinstance(masks, (np.ndarray,dict)):
    if masks:
        masks = arrs2dict(masks, listkey) if not isinstance(masks, dict) else masks
        for key in gros.keys():
            gro = np.array(gros[key])
            pre = np.array(pres[key])
            msk = np.array(masks[key])
            gro = gro[...,0] if len(gro.shape)==3 else gro
            pre = pre[...,0] if len(pre.shape)==3 else pre
            msk = msk[...,0] if len(msk.shape)==3 else msk
            dst=np.zeros((gro.shape[0],gro.shape[1],3))
            dst[(gro==1) & (pre==1) & (msk==1)]=(0,255,0)   #Green (overlapping)
            dst[(gro==1) & (pre!=1) & (msk==1)]=(255,0,0)    #Red (false negative, missing in pred)
            dst[(gro!=1) & (pre==1) & (msk==1)]=(0,0,255)    #Blue (false positive)
            dict_compare[key] = dst
    else:
        for key in gros.keys():
            gro = np.array(gros[key])
            pre = np.array(pres[key])
            gro = gro[...,0] if len(gro.shape)==3 else gro
            pre = pre[...,0] if len(pre.shape)==3 else pre
            dst=np.zeros((gro.shape[0],gro.shape[1],3))
            dst[(gro==1) & (pre==1)]=(0,255,0)   #Green (overlapping)
            dst[(gro==1) & (pre!=1)]=(255,0,0)    #Red (false negative, missing in pred)
            dst[(gro!=1) & (pre==1)]=(0,0,255)    #Blue (false positive)
            dict_compare[key] = dst
    return dict_compare


def save_images(images, size, path):
    """
    >>>images = np.zeros( (2,30,30) )
    >>>for idx, arr in enumerate(image):
    ...    print(idx, arr.shape )
    0 (30, 30)
    1 (30, 30)
    """
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w, :] = image
    return scipy.misc.imsave(path, merge_img)


def create_pairs(label):
    num = label.shape[0]
    if len(label.shape)>1:
        label = labeldehot(label)
    
    dict_idxs = dict()
    for cls in np.unique(label.astype(np.uint8)):
        idxs = np.argwhere(label==cls)
        dict_idxs[cls] = [idx[0] for idx in idxs.tolist()]
    
    dict_synonyms = dict()
    for cls, idxs in dict_idxs.items():
        dict_synonyms[cls] = random.sample(idxs, len(idxs))
    
    dict_antonyms = dict()
    for cls, idxs in dict_idxs.items():
        antonyms = [idx for idx in range(num) if idx not in dict_idxs.get(cls)]
        while(len(antonyms)<len(idxs)):
            antonyms.extend(antonyms)
        dict_antonyms[cls] = random.sample(antonyms, len(idxs))
    
    list_source = list()
    list_synonyms = list()
    list_antonyms = list()
    for cls in dict_idxs.keys():
        list_source.extend(dict_idxs.get(cls))
        list_synonyms.extend(dict_synonyms.get(cls))
        list_antonyms.extend(dict_antonyms.get(cls))

    # return list_source, list_synonyms, list_antonyms
    # arr_synonyms = np.concatenate([arr[np.newaxis,list_source],arr[np.newaxis,list_synonyms]], axis=0)
    # arr_antonyms = np.concatenate([arr[np.newaxis,list_source],arr[np.newaxis,list_antonyms]], axis=0) 
    
    return list_source, list_synonyms, list_antonyms


def softmax(x): 
    exp_x = np.exp(x)
    softmax_x = exp_x / np.tile(np.sum(exp_x, axis=1)[:,np.newaxis], [1,x.shape[1]])
    return softmax_x 
