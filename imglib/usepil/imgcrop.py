#! -*- coding:utf-8 -*-
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import morphology
from scipy.ndimage import rotate
from random import choice
from skimage.measure import label
from ...basic import *

def fov_imgs(list_dict_img, thresh_fov, threshold_pix=0, margin=0):
    if thresh_fov==-1:
        return list_dict_img, None
    
    dict_fov = dict()
    for key,img in list_dict_img[0].items():
        img_h, img_w = img.size
        mask = get_mask(img, thresh_fov, threshold_pix)
        rows, cols = np.where(mask.astype(np.int)==1)

        row_min = np.min(rows)-margin if rows.size!=0 else 0
        row_max = np.max(rows)+margin if rows.size!=0 else img_h
        col_min = np.min(cols)-margin if cols.size!=0 else 0
        col_max = np.max(cols)+margin if cols.size!=0 else img_w
        
        dict_fov[key] = [img_h, img_w, col_min, row_min, col_max, row_max]

        for idx in range(len(list_dict_img)):
            if key not in list_dict_img[idx].keys():
                raise IOError('file miss! list_dict_imgs[{}]: {}'.format(idx, key))
            img_src = list_dict_img[idx].get(key)
            if img_src.size != img.size:
                raise IOError('img size not same!!! filename:%s'%key)
            crop_img = img_src.crop( box = (col_min, row_min, col_max, row_max) )
            list_dict_img[idx][key] = crop_img
    return list_dict_img, dict_fov

def mask_imgs(dict_img, thresh_fov):
    dict_dst = dict()
    for key,img in dict_img.items():
        mask = get_mask(img, thresh_fov)
        dict_dst[key] = Image.fromarray( mask.astype(np.uint8)*255 )
    return dict_dst

def removemargin(dict_img, rect, savepixel=-1):
    # rect = (start_w, start_h, start_w+img_w, start_h+img_h)
    for key,img in dict_img.items():
        if savepixel!=-1:
            img = np.array(img)
            rows, cols = np.where( img==savepixel )
            img = img[np.min(rows):np.max(rows)+1, np.min(cols):np.max(cols)+1]
        else:
            img = img.crop( box = rect )
        dict_img[key] = img
    return dict_img


# def filling_area(dict_img, edgepixel):
#     for key,img in dict_img.items():
#         imarr = np.array(img)
#         rows, cols = np.where( imarr==edgepixel )
#         for idx, row in enumerate(rows):
#             imarr[row][cols):np.max(cols)+1] = edgepixel
#         dict_img[key] = imarr
#     return dict_img

def gray2binary(gray, thresh):
    table  =  []
    for i in range(256):
        if i < thresh:
            table.append(0)
        else:
            table.append(1)
    # convert to binary image by the table 
    im_bw = gray.point(table, '1')
    return im_bw


def get_mask( img, thresh_fov, threshold_pix=0):
    gray = img.convert('L')
#     gray = gray.filter(ImageFilter.GaussianBlur(radius=5))
    if threshold_pix == 0:
        #threshold = ( np.mean(np.asarray(gray)) - np.min(np.asarray(gray)) )/thresh_fov
        threshold = np.mean((np.asarray(gray)).astype(float))/(thresh_fov+0.0)
    else:
        threshold = threshold_pix
    im_bw = gray2binary(gray, threshold)
    # 滤除 面积中小于某个阈值的 连通区域
    im_w, im_h = im_bw.size
    imbool = morphology.remove_small_objects((np.array(im_bw)>0), min_size=(im_w*im_h/8), connectivity=2)
    return imbool


def defov(dict_img, dict_fov):
    for name, img in dict_img.items():
        fov = dict_fov[name]
        dst = Image.new( img.mode, (fov[0], fov[1]))
        dst.paste(img, box=fov[2:] )
        dict_img[name] = dst
    return dict_img


def centercrop(list_dict_img, img_w, img_h):
    list_dict_dst = []
    for dict_img in list_dict_img:
        dict_dst = {}
        for key,img in dict_img.items():
            w, h = img.size
            start_w = int((w-img_w)/2)
            start_h = int((h-img_h)/2)
            crop_img = img.crop( box = (start_w, start_h, start_w+img_w, start_h+img_h) )
            dict_dst[key] = crop_img
        list_dict_dst.append(dict_dst)
    return list_dict_dst


def drawBox(dict_img, angle, m1=0, m2=1):
    dict_box = dict()
    for key,img in dict_img.items():
        assert(len(np.unique(img))<=2)
        assert(np.max(img)==1)
        img = np.round( rotate(img[:,:,0], angle, axes=(0, 1), reshape=False) )
        rows, cols = np.where(img.astype(np.int)==1)
        assert( (rows.size!=0) == (np.max(img)==1) )
        arrbox = np.zeros((img.shape))
        row_min = np.min(rows)-choice(range(m1,m2,1))
        row_max = np.max(rows)+choice(range(m1,m2,1))
        col_min = np.min(cols)-choice(range(m1,m2,1))
        col_max = np.max(cols)+choice(range(m1,m2,1))   
        arrbox[row_min:row_max, col_min:col_max] = 1
            
        dict_box[key] = np.round( rotate(arrbox, -angle, axes=(0,1), reshape=False) )
    return dict_box


def drawbox(dict_gro, m1=0, m2=1):
    dict_box = dict()
    for key,gro in dict_gro.items():
        assert( len(np.unique(gro))<=2 )
        assert( np.max(gro)==1 )
        arr_label = label(gro[:,:,0], background=0, connectivity = 2)
        img_h, img_w = arr_label.shape
        box = np.zeros( (img_h, img_w) )
        for idx in range(1,np.max(arr_label)+1,1):
            rows, cols = np.where( (arr_label==idx)==True )
            row_min = max(np.min(rows)-choice(range(m1,m2,1)), 0)
            row_max = min(np.max(rows)+choice(range(m1,m2,1)), img_h)
            col_min = max(np.min(cols)-choice(range(m1,m2,1)), 0)
            col_max = min(np.max(cols)+choice(range(m1,m2,1)), img_w)
            box[row_min:row_max, col_min:col_max] = 1
        dict_box[key] = box
    return dict_box


def grolabelcrop(list_dict_img, m1=-60, m2=60):
    list_dict_crop = [{} for dict_img in list_dict_img]
    dict_gro = list_dict_img[0]
    for key,gro in dict_gro.items():
        img_h, img_w = gro.size
        arr_gro = np.array(gro)
        assert(len(np.unique(arr_gro))<=2)
        arr_label = label(arr_gro, background=0, connectivity = 2)
        for idx in range(1,np.max(arr_label)+1,1):
            rows, cols = np.where( (arr_label==idx)==True )
            row, col = ( np.min(rows) + np.max(rows) )/2., ( np.min(cols) + np.max(cols) )/2.
            start_h = row-choice(range(m1,m2,1))
            start_w = col-choice(range(m1,m2,1))
            newkey = os.path.splitext(key)[0] + str(idx) + os.path.splitext(key)[1]
            for idx,dict_img in enumerate(list_dict_img):
                if key not in dict_img.keys():
                    raise IOError('file miss! list_dict_imgs[{}]: {}'.format(idx, key))
                img = dict_img.get(key)
                if img.size != gro.size:
                    raise IOError('img size not same!!! filename:%s'%key)
                crop_img = img.crop( box=(start_w, start_h, start_w+img_w, start_h+img_h) )
                list_dict_crop[idx][newkey] = crop_img
    return list_dict_crop
