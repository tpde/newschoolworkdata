#! -*- coding:utf-8 -*-
import os, cv2
import numpy as np
from PIL import Image

def heatbatch(arr_imgs, is_frombottom, minV=None, maxV=None, scale=1, yticknum=None, listkey=None):
    dict_dst = dict()
    if type(arr_imgs)==type(np.array([])):
        for i in range(arr_imgs.shape[0]):
            key = i if listkey==None else listkey[i]
            dict_dst[key]=arr2heat(arr_imgs[i], is_frombottom, minV, maxV, scale, yticknum)
    else:
        for key, arr in arr_imgs.items():
            dict_dst[key]=arr2heat(arr, is_frombottom, minV, maxV, scale, yticknum)
    return dict_dst

def arr2heat(arr, is_frombottom, minV=None, maxV=None, scale=1, yticknum=None):
    if len(arr.shape)==3:
        assert( arr.shape[-1]==1 )
        arr = arr[:,:,0]
    assert( len(arr.shape)==2 )
    
    minV = minV or np.min(arr)
    maxV = maxV or np.max(arr)
    bar = np.array( Image.open('./bar.png') )[::-1, 0, :3]
    strideV = (maxV-minV)/(bar.shape[0]-2)
    def get_color( v ):
        if v<minV:
            return 0
        elif v>maxV:
            return 255
        else:
            i = 0
            while( minV + i*strideV < v ):
                i += 1
            return bar[i]
    if is_frombottom:
        arr = arr[::-1, :]
    rows, cols = arr.shape[:2]   
    dst = np.zeros([rows, cols, 3])
    for row in range(rows):
        for col in range(cols):
            dst[row,col,:] = get_color( arr[row,col] )
            
    img = Image.fromarray( dst.astype(np.uint8) )
    if scale!=1:
        img = img.resize( (int(scale*img.size[0]), int(scale*img.size[1])), Image.ANTIALIAS )   # Image.ANTIALIAS, Image.NEARESTNEAREST
    if yticknum:
        bar = Image.open('./bar.png')
        bar = bar.resize( (20, img.size[1]), Image.ANTIALIAS )
        ytick = np.zeros( (img.size[1], 90, 3) )
        ystride = int( img.size[1]/yticknum )
        for i in range(yticknum):
            value = minV + (maxV-minV)/yticknum*i
            ytick = cv2.putText(ytick, '%.3f'%value, (0, int(img.size[1] - i*ystride)),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        ytick = cv2.putText(ytick, '%.3f'%maxV, (0, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        img = np.hstack( [np.array(img), np.array(bar)[...,:3], ytick] )
        img = Image.fromarray( img.astype(np.uint8) )
    return img
    
from ...basic import mkdir
from ...nparray import puzzel
import random

def sampleSlice(list_slices, list_path, num, cols):
    list_example = random.sample(range(list_slices[0].shape[0]), num)
    for idx in range(len(list_slices)):
        arr_sample = list_slices[idx][list_example]
        show_slices(arr_sample, cols, list_path[idx])


def show_slices( slices, cols, filepath):
    assert slices.shape[0]%cols == 0
    if len(os.path.split(filepath)[0])>0:
        mkdir(os.path.split(filepath)[0])
    imgArray = puzzel.puzzle_slices(slices, cols)
    if np.max(imgArray)<=1:
        imgArray = imgArray*255
    if len(imgArray.shape)==3 and imgArray.shape[2]==1:
        imgArray = np.reshape(imgArray, (imgArray.shape[0],imgArray.shape[1]))
    img = Image.fromarray(imgArray.astype(np.uint8))
    img.save( filepath )

def show_img(imgs, save_dir, offset, keylen=3, prefix='', tail='.png'):
    """save imgs to save_dir as Image

    # Arguments
        imgs: can be anny type of these 4:
                dict_img: {index:img}     index must be int
                dict_img: {index:np.array(img)}        index must be int
                list_img: [img]
                list_img: [np.array(img)]

        save_dir: the directory to cantain images
        
        prefix: the prefix of img filename
        
        exception: img_dir is empty

    # Returns

    # Raises
        TypeError: imgs is not dict or list of imgs.
    """
    mkdir(save_dir)
    rightkey = lambda keylen,index: '%02d'%(index) if keylen==2     \
                    else '%03d'%(index) if keylen==3        \
                    else '%04d'%(index) if keylen==4        \
                    else '%05d'%(index)

    if type(imgs) == type(dict()):
        for key,img in imgs.items():
            if type(img)==type(np.array([])):
                if np.max(img)<=1:
                    img = img * 255
                if len(img.shape)==3 and img.shape[2]==1:
                    img = img.reshape((img.shape[0],img.shape[1]))
                img = Image.fromarray(img.astype(np.uint8))
            newkey = prefix + rightkey(keylen,key+offset) + tail
            img.save( os.path.join(save_dir,newkey) )
    elif type(imgs) == type(list()):
        for i in range(len(imgs)):
            img = imgs[i]
            if type(img)==type(np.array([])):
                if np.max(img)<=1:
                        img = img * 255
                if len(img.shape)==3 and img.shape[2]==1:
                    img = img.reshape((img.shape[0],img.shape[1]))
                img = Image.fromarray(img.astype(np.uint8))
            newkey = prefix + rightkey(keylen,i+offset) + tail
            img.save( os.path.join(save_dir,newkey) )
    else:
        raise(TypeError,'show_img(): the imgs must be dict or list of imgs!!!' )

