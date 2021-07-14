#! -*- coding:utf-8 -*-
import cv2
import numpy as np

def greenChannel(dict_img):
    dict_array = dict()
    for key,imarr in dict_img.items():
        if imarr.shape[2]==3:
            imarr = 0.299*imarr[:,:,0:1]+0.587*imarr[:,:,1:2]+0.114*imarr[:,:,2:3]
        dict_array[key] = imarr
    return dict_array

# def greenChannel(imgsArray):
#     assert (len(imgsArray.shape)==4)
#     assert (imgsArray.shape[3]==3)
#     imgsArray_gray = imgsArray[:,:,:,0]*0.299 + imgsArray[:,:,:,1]*0.587 + imgsArray[:,:,:,2]*0.114
#     imgsArray_gray = np.reshape(imgsArray_gray,(imgsArray.shape[0], imgsArray.shape[1], imgsArray.shape[2], 1))
#     return imgsArray_gray

def imagePreProcess(dict_imgarray):
    dict_dst = normalized_global(dict_imgarray)
    dict_dst = clahe_equalized(dict_imgarray)
    dict_dst = adjust_gamma(dict_imgarray, 1.2)
    return dict_dst

def normalized(dict_imgarray):
    allarray = np.array([])
    for key, imgarray in dict_imgarray.items():
        allarray = np.append(allarray,np.array(imgarray.flat))
    img_max = np.max(allarray)
    
    dict_dst=dict()
    for key, imgarray in dict_imgarray.items():
        imgarray = imgarray/img_max
        dict_dst[key] = imgarray
    return dict_dst


def normalized_global(dict_imgarray):
    allarray = np.array([])
    for key, imgarray in dict_imgarray.items():
        allarray = np.append(allarray,np.array(imgarray.flat))
    img_mean = np.mean(allarray)
    img_std = np.std(allarray)
    
    dict_dst = dict()
    for key, imgarray in dict_imgarray.items():
        imgarray = (imgarray-img_mean)/img_std
        imgarray = ((imgarray - np.min(imgarray)) / (np.max(imgarray)-np.min(imgarray)))*255
        dict_dst[key] = imgarray
    return dict_dst

def normalized_local(dict_imgarray):
    dict_dst = dict()
    for key, imgarray in dict_imgarray.items():
        img_mean = np.mean(imgarray)
        img_std = np.std(imgarray)
        imgarray = (imgarray-img_mean)/img_std
        imgarray = ((imgarray - np.min(imgarray)) / (np.max(imgarray)-np.min(imgarray)))*255
        dict_dst[key] = imgarray
    return dict_dst

def clahe_equalized(dict_imgarray):
    dict_dst = dict()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for key, imgarray in dict_imgarray.items():
        imgarray.flags['WRITEABLE'] = True  
        for channel in range(imgarray.shape[2]):
            imgarray[:,:,channel] = clahe.apply(imgarray[:,:,channel].astype(np.uint8))
        dict_dst[key] = imgarray
    return dict_dst

def adjust_gamma(dict_imgarray, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    dict_dst = dict()
    for key, imgarray in dict_imgarray.items():
        dict_dst[key] = cv2.LUT(imgarray.astype(np.uint8), table)
    return dict_dst


def gaussian_filter(dict_img, ksize=260/30):
    dict_dst = dict()
    for key,img in dict_img.items():
        try:
            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img,(0,0),ksize), -4, 128)
#             mask = np.zeros(img.shape)
#             cv2.circle(mask,(img.shape[1]/2, img.shape[0]/2),  img.shape[1]/2-2, (1,1,1), -1, 8, 0)
#             img = img*mask +128*(1-mask)
            dict_dst[key] = img
        except:
            print(key)
    return dict_dst

def gaussianBlur(dict_img, k=15, sigma=1.5):
    '''
    cv2.GaussianBlur(src,ksize,sigmaX)
    kSize:核大小,sigmaX:高斯核在x轴的标准差,如果为0会根据核的宽和高重新计算
    '''
    dict_dst = {}
    for file,img in dict_img.items():
        blur = cv2.GaussianBlur(img , (k,k) , sigma, sigma)
        dict_dst[file] = blur
    return dict_dst

def erode_dilate(dict_img, is_erode, ksize=5):
    dict_dst = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize))
    for key,img in dict_img.items():
        if is_erode:
            dict_dst[key] = cv2.erode(img, kernel)
        else:
            dict_dst[key] = cv2.dilate(img, kernel)  # 膨胀
    return dict_dst

