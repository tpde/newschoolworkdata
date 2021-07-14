#! -*- coding:utf-8 -*-
import os, random
import numpy as np
from os.path import join
from ..imglib import usecv2, read_images, unite_size, fov_imgs, thresh_binary
from ..nparray import groZeroOne, img2array, random_extract, target_extract, grid_extract

def prepare_data(list_dirs, list_isbw):
    list_dict_img = list()
    for imgdir in list_dirs:
        list_dict_img.append( read_images(imgdir) )
        
    for idx, dict_img in enumerate(list_dict_img):
        if list_isbw[idx]:
            dict_img = thresh_binary( dict_img, 127 )
            dict_img = groZeroOne( dict_img )
        else:
            dict_img = img2array(dict_img)
        list_dict_img[idx] = dict_img
    return list_dict_img
    
    
def prepare_images(list_dirs, list_isbw, thresh_fov, img_w, list_hs, list_mode, will_recover=False):
    """
    Args:
        list_dirs:    ['dir/test_ori', 'dir/test_OD', 'dir/test_EX']
        list_isbw:    [False, True, True]
        fov_thresh: -1: not fov, 2.5:fov

    Returns:
        list_dict_img:[ dict_ori, dict_OD, dict_EX]
    """
    list_dict_img = list()
    for imgdir in list_dirs:
        list_dict_img.append( read_images(imgdir) )
    list_dict_img, dict_fov = fov_imgs(list_dict_img, thresh_fov)
    
    for idx, dict_img in enumerate(list_dict_img):
        dict_img, dict_rsiz = unite_size(dict_img, img_w, list_hs, is_binary=list_isbw[idx], is_recover=True)
        if list_isbw[idx]:
            dict_img = groZeroOne( dict_img )
        else:
            dict_img = img2array(dict_img)
            if list_mode[idx]==None:
                pass
            elif list_mode[idx]=='Enhance':
                dict_img = usecv2.imagePreProcess(dict_img)
            elif list_mode[idx]=='greenEnhance':
                dict_img = usecv2.imagePreProcess(usecv2.greenChannel(dict_img))
            else:
                raise('list_mode error!!! the element of list_mode must be in [None, "Enhance", "greenEnhance"]')
        list_dict_img[idx] = dict_img
    if will_recover:
        return list_dict_img, dict_fov, dict_rsiz
    else:
        return list_dict_img


def prepare_patches(list_dirs, list_isbw, extract_mode, num_patches, \
                    patch_h, patch_w, target_idx=-1, stride_h=5, stride_w=5):
    """
    Args:
        list_dirs:    ['dir/test_ori', 'dir/test_OD', 'dir/test_EX']
        list_isbw:    [False, True, True]
        target_idx:    -1: random_extract,    1 or 2: target_extract

    Returns:
        list_dict_img:[ dict_ori, dict_OD, dict_EX]
    """
    list_dict_img = list()
    for idx,imgdir in enumerate(list_dirs):
        dict_img = read_images(imgdir)
        dict_img = groZeroOne(dict_img) if list_isbw[idx] else img2array(dict_img)
        list_dict_img.append( dict_img )
    
    if extract_mode=='random':
        return random_extract(list_dict_img, num_patches, patch_h, patch_w)
        
    elif extract_mode=='target':
        return target_extract(list_dict_img, target_idx, num_patches, patch_h, patch_w)

    elif extract_mode=='grid':
        return [grid_extract(dict_img, patch_h, patch_w, stride_h, stride_w) for dict_img in list_dict_img]

    else:
        raise TypeError


def split_dataset(dict_imgdir, dict_split, save_dir, n_split=-1, is_shuffle=False):
    """
    Args:
        dict_imgdir = {'RGB':'images/RGB','610':'images/610'}
        dict_split = {  'split1':{'test':['001.png','002.png'],'valid':['003.png','004.png']},
                        'split2':{'test':['003.png','004.png'],'valid':['001.png','002.png']}}
        save_dir = 'version'
    """
    dict_dictpath = {}
    for key,imgdir in dict_imgdir.items():
        dict_dictpath[key] = {file:join(imgdir,file) for file in os.listdir(imgdir)}
        
    if n_split!= -1:
        list_files = sorted( list(dict_dictpath.values())[0].keys() )
        if is_shuffle:
            random.shuffle(list_files)
        dict_split = {'split%d'%n:{} for n in range(n_split)}
        for n in range(n_split):
            idx, list_idx = n, []
            while( idx < len(list_files) ):
                list_idx.append( idx )
                idx += n_split
            dict_split['split%d'%n]['test'] = np.array(list_files)[list_idx].tolist()
            
            idx, list_idx = (n+1)%n_split, []
            while( idx < len(list_files) ):
                list_idx.append( idx )
                idx += n_split
            dict_split['split%d'%n]['valid']= np.array(list_files)[list_idx].tolist()
                
    for split,dict_dsetfiles in dict_split.items():
        usedfiles = []
        for dset,list_file in dict_dsetfiles.items():
            usedfiles.extend(list_file)
            for key,dictpath in dict_dictpath.items():
                allimgname = list(dictpath.keys())
                dict_imgpart = {file:dictpath[file] for file in list_file}
                # write_images(dict_imgpart, os.path.join(save_dir,split,dset,key))
                dstdir = join(save_dir, split, dset, key)
                if not os.path.exists(dstdir):
                    os.makedirs(dstdir)
                for file,srcpath in dict_imgpart.items():
                    os.system('cp ' +srcpath+' '+join(dstdir, file))
        for key,dictpath in dict_dictpath.items():
            dict_imgpart = {file:dictpath[file] for file in allimgname if file not in usedfiles}
            for file,srcpath in dict_imgpart.items():
                dstdir = join(save_dir, split, 'train', key)
                if not os.path.exists(dstdir):
                    os.makedirs(dstdir)
                os.system('cp ' +srcpath+' '+join(dstdir, file))
        


