#! -*- coding:utf-8 -*-
import os
import math
import numpy as np
import cv2 as cv
from PIL import Image
from skimage import morphology, measure
from scipy import ndimage
from data_process import *
from Ailib.io import save2excel
from Ailib.datasets.common import prepare_images
from Ailib.nparray import arrs2dict
from mytrace.mytrace_V2 import BVtraceUseAV

def get_radius(img):
    img = np.array(img)
    if img.ndim == 3:
        high, width, ch = img.shape
        img_t = img[:, :, 0]
    else:
        high, width = img.shape
        img_t = img[:, :]
    ret, img = cv.threshold(img_t, 2, 255, cv.THRESH_BINARY)
    img_canny = cv.Canny(img.astype(np.uint8), 50, 150)
    coordinate = np.nonzero(img_canny)
    radius = int(np.min(np.sqrt((coordinate[0] - int(high / 2 - 1)) ** 2 + (coordinate[1] - int(width / 2 - 1)) ** 2)))
    return radius


def registration(arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,arr_610,arr_rgb):

#     radius = np.min(np.array([get_radius(arr_rgb ), get_radius(arr_610 ), get_radius(arr_570 )]))
    
    high, width, ch = arr_rgb.shape
    radius = min(high, width)//2-2
    mask_gray = np.zeros((high, width))
    mask_gray = cv.circle(mask_gray, (int(width / 2 - 1), int(high / 2 - 1)), radius, (1, 1), -1)
    mask_RGB = np.zeros((high, width, ch))
    mask_RGB = cv.circle(mask_RGB, (int(width / 2 - 1), int(high / 2 - 1)), radius, (1, 1), -1)
    
    arr_rgb = arr_rgb*mask_RGB
    arr_610 = arr_610*mask_gray
    arr_570 = arr_570*mask_gray
    
    arr_vessel = arr_vessel*mask_gray
    arr_vein = arr_vein*mask_gray
    arr_artery = arr_artery*mask_gray
    arr_OD = arr_OD*mask_gray
    
#     point1_h = int(high / 2 - 1) - radius
#     point1_w = int(width / 2 - 1) - radius
#     point2_h = int(high / 2 - 1) + radius
#     point2_w = int(width / 2 - 1) + radius
#     point1 = [point1_h, point1_w]
#     point2 = [point2_h, point2_w]
#     
#     arr_rgb = arr_rgb[ point1[0]:point2[0], point1[1]:point2[1], : ]
#     arr_610 = arr_610[ point1[0]:point2[0], point1[1]:point2[1] ]
#     arr_570 = arr_570[ point1[0]:point2[0], point1[1]:point2[1] ]
#     arr_610 = arr_610[ point1[0]:point2[0], point1[1]:point2[1] ]
#     
#     arr_vessel = arr_vessel[ point1[0]:point2[0], point1[1]:point2[1] ]
#     arr_vein = arr_vein[ point1[0]:point2[0], point1[1]:point2[1] ]
#     arr_artery = arr_artery[ point1[0]:point2[0], point1[1]:point2[1] ]
#     arr_OD = arr_OD[ point1[0]:point2[0], point1[1]:point2[1] ]
    
    return arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,arr_610,arr_rgb

def SO2(arr_vessel,arr_570,arr_610, is_artery=True):
#     _,arr_OD = cv.threshold(arr_OD,20,1,cv.THRESH_BINARY)
#     _,arr_vessel = cv.threshold(arr_vessel,20,1,cv.THRESH_BINARY)
#     arr_OD = arr_OD.astype(np.uint8)
#     arr_vessel = arr_vessel.astype(np.uint8)
#     OD_kernal = np.ones((13,13), np.uint8)
    vessel_kernal_small = np.ones((3,3), np.uint8)
    vessel_kernal_big = np.ones((13,13), np.uint8)
#     dil_OD = cv.dilate(arr_OD,OD_kernal,iterations=1)
#     
#     arr_vessel = arr_vessel*(1-dil_OD)
    vessel_label = measure.label(arr_vessel,connectivity=2)
#     vessel_label = arr_vessel
    labels = np.unique(vessel_label)
    so2_map = np.zeros(arr_vessel.shape, np.uint8)
    list_so2 = []
    for ind in labels[1:]:
        vessel_single = ((vessel_label==ind).astype(np.uint8))
#         img = Image.fromarray( (vessel_single!=0).astype(np.uint8)*255 )
#         img.save('a.png')
        dil_vessel_small = cv.dilate(vessel_single,vessel_kernal_small,iterations=1)
        dil_vessel_big = cv.dilate(vessel_single,vessel_kernal_big,iterations=1)
        vessel_out = dil_vessel_big-dil_vessel_small
#         vessel_skel, distance = morphology.medial_axis(vessel_single, return_distance=True)
#         img_distance = vessel_skel*distance
#         list_dim = img_distance[np.where(img_distance>0)]
#         dim_mean = 2*2.36*np.mean(list_dim)
        dim_mean = 0
        
        in_610 = sorted(arr_610[(arr_610*vessel_single)>0])
#         in_610 = in_610[:int(0.9*len(in_610))]
        in_610 = np.mean( in_610 )
        out_610 = sorted(arr_610[(arr_610*vessel_out)>0])
#         out_610 = out_610[:int(0.9*len(out_610))]
        out_610 = np.mean( out_610 )
        OD_610 = np.log10(out_610/(in_610+1e-3))
        
        in_570 = arr_570[(arr_570*vessel_single)>0]
#         in_570 = in_570[:int(0.9*len(in_570))]
        in_570 = np.mean(in_570  )
        out_570 = np.mean( arr_570[(arr_570*vessel_out)>0] )
        OD_570 = np.log10(out_610/(in_570+1e-3))
        
        ODR_cor = OD_610/(OD_570+1e-6)
        so2 = -225.72*ODR_cor+119
        if is_artery:
            so2 = so2 - 1.52*dim_mean+7.86   #对直径进行校正
            so2 = so2-102.2*np.log10(out_610 / out_570)+20.2
        else:
            so2 = so2+5.456*dim_mean-39.9
            so2 = so2-158.6*np.log10(out_610/out_570)+30.87
        if so2>100:
            so2=100
        elif so2<30:
            so2=30
        so2_map[vessel_single==1] = so2
        list_so2.append(so2)
    so2_map[so2_map>100]=100
    so2_map[so2_map<0]=0
    return so2_map,list_so2
    # img = Image.fromarray( (so2map_vein!=0).astype(np.uint8)*255 )


def plot_heatmap(so2_map,arr_570, save_path, is_add570=False):

    bar = np.array( Image.open('newdb/bar.png') )[...,:3]
    rows, cols = bar.shape[:2]
    table = [[0,0,0], ]
    for i in range(1, 101):
        table.append( bar[rows-1 - int((i-1)*(rows-1)/99), cols//2] )     
    rows, cols = so2_map.shape[:2]   
    dst = np.zeros([rows, cols, 3])
    for row in range(rows):
        for col in range(cols):
            dst[row,col,:] = table[int(so2_map[row,col])]
    dst = Image.fromarray(dst.astype(np.uint8))
    dst.save(save_path)
    return dst
    
def read_images(img_dir, is_gray=False):
    dict_img = dict()
    for file in os.listdir(img_dir):
        try:
            img = Image.open(os.path.join(img_dir,file))
            if is_gray:
                img = img.convert('L')
        except IOError:
            raise( IOError,'file:%s miss!!!'%os.path.join(img_dir,file) )
        dict_img[file] = img
    return dict_img

def cal_so2_vessel(arr_batch_vessel, arr_batch_RGB, arr_batch_610, arr_batch_570):
    list_SO2 = []
    if np.max(arr_batch_vessel)<=1:
        arr_batch_vessel = arr_batch_vessel*255
    for i in range(arr_batch_RGB.shape[0]):
        arr_vessel   = arr_batch_vessel[i,:,:,0]
        arr_610 = arr_batch_610[i,:,:,0]
        arr_570 = arr_batch_570[i,:,:,0]
        
#         arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,\
#         arr_610,arr_RGB = registration(arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,arr_610,arr_RGB)
        arr_570 = make_CLAHE(arr_570)
        arr_610 = make_CLAHE(arr_610)
        
        img_cross = np.zeros(arr_vessel.shape, np.uint8)
        
        arr_vessel = arr_vessel//255
        vessel_skel = (morphology.thin(arr_vessel>0) ).astype(np.uint8)
        img_conv = ndimage.convolve(vessel_skel, np.ones((3,3)), cval=1)
        rows, cols = np.where(img_conv*vessel_skel>3)
        img_cross[rows,cols] = 1
        img_cross_dia = cv2.dilate(img_cross,np.ones((10,10), np.uint8),iterations=1)
        img_vessel_breake = arr_vessel*(1-img_cross_dia)
        img_vessel_breake = img_vessel_breake * 255
        
        img_skel, distance = morphology.medial_axis(img_vessel_breake, return_distance=True)
        distance = img_skel*distance
        img_skel = (distance>=2).astype(np.uint8)
        img_vessel_breake = cv.dilate(img_skel,np.ones((10,10), np.uint8),iterations=1)* arr_vessel
        
        so2_map, list_so2 = SO2(img_vessel_breake, arr_570, arr_610)

        so2_map = so2_map/np.max(so2_map)*255
        list_SO2.append(so2_map)
    return np.array(list_SO2)[...,np.newaxis]

def cal_so2_vessel_AV(arr_batch_vessel, arr_batch_vein, arr_batch_artery,arr_batch_RGB, arr_batch_610, arr_batch_570):
    list_SO2 = []
    if np.max(arr_batch_vessel)<=1:
        arr_batch_vessel = arr_batch_vessel*255
    for i in range(arr_batch_RGB.shape[0]):
        arr_vessel   = arr_batch_vessel[i,:,:,0]
        arr_610 = arr_batch_610[i,:,:,0]
        arr_570 = arr_batch_570[i,:,:,0]
        arr_vein = arr_batch_vein[i,:,:,0]
        arr_artery = arr_batch_artery[i,:,:,0]
        
#         arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,\
#         arr_610,arr_RGB = registration(arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,arr_610,arr_RGB)
        arr_570 = make_CLAHE(arr_570)
        arr_610 = make_CLAHE(arr_610)
        
        arr_artery = arr_vessel*arr_artery
        arr_vein = arr_vessel*arr_vein
        
        so2_map_vein, _ = SO2(arr_vein,arr_570,arr_610, is_artery=False)
        so2_map_artery, _ = SO2(arr_artery,arr_570,arr_610)
        
        so2_map = so2_map_vein+so2_map_artery
        cross = (( so2_map_vein*((so2_map_artery>0).astype(np.uint8)) )>0 ).astype(np.uint8)
        so2_map = so2_map*(1-cross)
        so2_map = so2_map + so2_map*cross*0.5
        
        so2_map = so2_map/np.max(so2_map)*255
        list_SO2.append(so2_map)
    return np.array(list_SO2)[...,np.newaxis]



def cal_so2_traceBVV(arr_batch_vessel, arr_batch_vein, arr_batch_OD, arr_batch_RGB, arr_batch_610, arr_batch_570):
    list_SO2 = []
    if np.max(arr_batch_vessel)<=1:
        arr_batch_vessel = arr_batch_vessel*255
    for i in range(arr_batch_RGB.shape[0]):
        arr_vessel   = arr_batch_vessel[i,:,:,0]
        arr_610 = arr_batch_610[i,:,:,0]
        arr_570 = arr_batch_570[i,:,:,0]
        arr_vein = arr_batch_vein[i,:,:,0]
        arr_OD = arr_batch_OD[i,:,:,0]
        
#         arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,\
#         arr_610,arr_RGB = registration(arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,arr_610,arr_RGB)
        arr_570 = make_CLAHE(arr_570)
        arr_610 = make_CLAHE(arr_610)
        
        label_BV = BVtraceUseAV(arr_vessel>0, arr_vein>0, arr_OD>0, bw4crosslen=10, prefix="")
        
        so2_map, _ = SO2(label_BV,arr_570,arr_610)        
        so2_map = so2_map/np.max(so2_map)*255
        
        list_SO2.append(so2_map)
    return np.array(list_SO2)[...,np.newaxis]

def cal_so2(arr_batch_vessel, arr_batch_RGB, arr_batch_vein, arr_batch_artery, 
            arr_batch_610, arr_batch_570,arr_batch_OD):
    list_SO2 = []
    if np.max(arr_batch_vessel)<=1:
        arr_batch_vessel = arr_batch_vessel*255
    if np.max(arr_batch_vein)<=1:
        arr_batch_vein = arr_batch_vein*255
    if np.max(arr_batch_artery)<=1:
        arr_batch_artery = arr_batch_artery*255
    if np.max(arr_batch_OD)<=1:
        arr_batch_OD = arr_batch_OD*255
    
    for i in range(arr_batch_RGB.shape[0]):
        arr_RGB = arr_batch_RGB[i,:,:,:]
        arr_vessel   = arr_batch_vessel[i,:,:,0]
        arr_vein   = arr_batch_vein[i,:,:,0]
        arr_artery = arr_batch_artery[i,:,:,0]
        arr_610 = arr_batch_610[i,:,:,0]
        arr_570 = arr_batch_570[i,:,:,0]
        arr_OD = arr_batch_OD[i,:,:,0]
        
        arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,arr_610,arr_RGB = registration(arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,arr_610,arr_RGB)
        
        arr_570 = make_CLAHE(arr_570)
        arr_610 = make_CLAHE(arr_610)
        
        # *******************vessel trace**********************
        arr_vessel = arr_vessel>20
        arr_artery = arr_artery>20
        arr_vein = arr_vein>20
        arr_artery, arr_vein = vessel_trace(arr_vessel, arr_vein, arr_artery)
        
        so2_map_vein, list_vein_so2 = SO2(arr_vein,arr_OD,arr_570,arr_610)
        so2_map_artery, list_artery_so2 = SO2(arr_artery,arr_OD,arr_570,arr_610)
        
        so2_map = so2_map_vein+so2_map_artery
        cross = (( so2_map_vein*((so2_map_artery>0).astype(np.uint8)) )>0 ).astype(np.uint8)
        so2_map = so2_map*(1-cross)
        so2_map = so2_map + so2_map*cross*0.5
        
        so2_map = so2_map/np.max(so2_map)*255
        list_SO2.append(so2_map)
    return np.array(list_SO2)[...,np.newaxis]

def cal_so2_TrackNet(arr_batch_vessel, arr_batch_vein,arr_batch_RGB, arr_batch_610, arr_batch_570):
    list_SO2 = []
#     if np.max(arr_batch_vessel)<=1:
#         arr_batch_vessel = arr_batch_vessel*255
    for i in range(arr_batch_RGB.shape[0]):
        arr_vessel   = arr_batch_vessel[i,:,:,0]
        arr_610 = arr_batch_610[i,:,:,0]
        arr_570 = arr_batch_570[i,:,:,0]
        arr_vein = arr_batch_vein[i,:,:,0]
        
#         arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,\
#         arr_610,arr_RGB = registration(arr_vessel, arr_vein,arr_artery,arr_OD,arr_570,arr_610,arr_RGB)
        arr_570 = make_CLAHE(arr_570)
        arr_610 = make_CLAHE(arr_610)
        
        arr_vein = arr_vessel*arr_vein
#         img=Image.fromarray(arr_vein.astype(np.uint8)*255)
#         img.save('vein.png')
        arr_artery = ((arr_vessel- arr_vein)>0).astype(np.uint8)
        k = np.ones((9,9), np.uint8)
        arr_artery = cv.dilate(arr_artery,k,iterations=1)
        arr_artery = arr_vessel*arr_artery
#         img=Image.fromarray(arr_artery.astype(np.uint8)*255)
#         img.save('artery.png')

        so2_map_vein, _ = SO2(arr_vein,arr_570,arr_610)
        so2_map_artery, _ = SO2(arr_artery,arr_570,arr_610)
        
        so2_map = so2_map_vein+so2_map_artery
        cross = (( so2_map_vein*((so2_map_artery>0).astype(np.uint8)) )>0 ).astype(np.uint8)
        so2_map = so2_map*(1-cross)
        so2_map = so2_map + so2_map*cross*0.5
        
        so2_map = so2_map/np.max(so2_map)*255
        list_SO2.append(so2_map)
    return np.array(list_SO2)[...,np.newaxis]
       

from Ailib.nparray import img2array, stack_array, arrs2dict
from Ailib.imglib import write_images, combineimages,array2img
from skimage import morphology,color
if __name__ == '__main__':    
    if True:
        dict_vessel = read_images('temp/vessel/',is_gray=True)
        dict_artery = read_images('temp/artery/',is_gray=True)
        dict_vein = read_images('temp/vein/',is_gray=True)
        dict_OD   = read_images('temp/OD/',is_gray=True)
        dict_570  = read_images('temp/570/',is_gray=True)
        dict_610  = read_images('temp/610/',is_gray=True)
        dict_RGB  = read_images('temp/RGB/')
        
        list_vessel = []
        list_artery = []
        list_vein = []
        list_OD = []
        list_570 = []
        list_610 = []
        list_RGB = []
        list_key = []
        for key in dict_RGB.keys():
            list_vessel.append(np.array(dict_vessel[key])[...,np.newaxis] )
            list_artery.append(np.array(dict_artery[key])[...,np.newaxis])
            list_vein.append(np.array(dict_vein[key])[...,np.newaxis] )
            list_OD.append(np.array(dict_OD[key])[...,np.newaxis] )
            list_570.append(np.array(dict_570[key])[...,np.newaxis] )
            list_610.append(np.array(dict_610[key])[...,np.newaxis] )
            list_RGB.append(np.array(dict_RGB[key]) )
            list_key.append(key)
            
        batch_vessel = np.array(list_vessel)
        batch_artery = np.array(list_artery)
        batch_vein = np.array(list_vein)
        batch_OD = np.array(list_OD)
        batch_570 = np.array(list_570)
        batch_610 = np.array(list_610)
        batch_RGB = np.array(list_RGB)
        
#         arr_SO2 = cal_so2(batch_vessel, batch_RGB, batch_vein, batch_artery, 
#                 batch_610, batch_570, batch_OD)
        arr_SO2 = cal_so2_vessel(batch_vessel, batch_RGB, batch_610, batch_570)
        dict_SO2 = combineimages([array2img(batch_RGB, list_key),
                                 array2img(batch_vessel, list_key),
                                 array2img(arr_SO2, list_key)])
        write_images(dict_SO2, 'temp/SO2/')
    
    if False:
        arr_vessel = np.array( [[255,255, 0,0,0],
                                [255,255, 0,0,0],
                                [255,255, 0,0,255],
                                [255,255, 0,255,255]])
        vessel_label = measure.label(arr_vessel,connectivity=2)
        a = 1
#     dict_so2 = {}
#     for file_name in dict_RGB.keys():
#         arr_vessel = np.array(dict_vessel[file_name])
#         arr_artery = np.array(dict_artery[file_name])
#         arr_vein   = np.array(dict_vein[file_name])
#         arr_OD     = np.array(dict_OD[file_name])
#         arr_570    = np.array(dict_570[file_name])
#         arr_610    = np.array(dict_610[file_name])
#         arr_RGB    = np.array(dict_RGB[file_name])[:,:,:2]
# 
#         arr_vessel,arr_vein, arr_artery, arr_OD, arr_570, arr_610, arr_RGB = registration(arr_vessel, 
#                                                             arr_vein, arr_artery, arr_OD, arr_570, arr_610, arr_RGB)
# 
#         arr_570 = make_CLAHE(arr_570)
#         arr_610 = make_CLAHE(arr_610)
#         
#         so2_map_vein, list_vein_so2 = SO2(arr_vein,arr_OD,arr_570,arr_610, is_artery=True)
#         so2_map_artery, list_artery_so2 = SO2(arr_artery,arr_OD,arr_570,arr_610, is_artery=True)
#         so2_map = so2_map_vein+so2_map_artery
#         cross = (( so2_map_vein*((so2_map_artery>0).astype(np.uint8)) )>0 ).astype(np.uint8)
#         so2_map = so2_map*(1-cross)
#         so2_map = so2_map + so2_map*cross*0.5
#         so2_map = so2_map/np.max(so2_map)*255
#         img = Image.fromarray( so2_map.astype(np.uint8) )
#         img.save('newdb/SO2_/'+file_name)
# #         save_path = 'SO2_img/SO2_CLAHE/'+file_name
# #         img_so2 = plot_heatmap(so2_map,arr_570, save_path, is_add570=False)
#         
#         mean_vein_so2 = np.mean(np.array(list_vein_so2))
#         std_vein_so2 = np.std(np.array(list_vein_so2))
#         mean_artery_so2 = np.mean(np.array(list_artery_so2))
#         std_artery_so2 = np.std(np.array(list_artery_so2))
#         dict_so2[file_name] = {'mean_vein_so2':mean_vein_so2,'std_vein_so2':std_vein_so2,
#                                'mean_artery_so2':mean_artery_so2, 'std_artery_so2':std_artery_so2}

#     dict_img_so2 =  read_images('SO2_img/SO2_CLAHE/')
#     dict_cmp = combineimages( [dict_RGB,dict_img_so2,dict_vein,dict_artery ])
#     write_images(dict_cmp, 'SO2_img/SO2_cmp/', offset=1, keylen=3, prefix='', tail='.png')
#     save2excel(dict_so2, 'SO2.xls')
    
    