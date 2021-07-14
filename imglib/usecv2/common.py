#! -*- coding:utf-8 -*-
import cv2
import os
from ...basic import mkdir

def read_images(img_dir, is_gray=False):
	dict_img = dict()
	for file in os.listdir(img_dir):
		if is_gray:
			dict_img[file] = cv2.imread(os.path.join(img_dir,file), 0)
		else:
			dict_img[file] = cv2.imread(os.path.join(img_dir,file), -1)
	return dict_img

def paste_mask(dict_ori, dict_mask):
	dict_dst = dict()
	for key in dict_ori.keys():
		ori = dict_ori.get(key)
		mask= dict_mask.get(key)
		res = cv2.bitwise_and(ori, ori, mask=mask)       
		dict_dst[key] = res
	return dict_dst

def write_images(dict_img, save_dir):
	mkdir(save_dir)
	for key,img in dict_img.items():
		cv2.imwrite(os.path.join(save_dir,key), img)

def leftrightFlipOver(src):
	dst = src.copy()
	height,width = src.shape[:2] 
	for i in range(height):
		for j in range(width):
			dst[i, width-1-j] =  src[i,j]
	return dst

def updownFlipOver(src): 
	dst = src.copy()
	height,width = src.shape[:2] 
	for i in range(height):
		for j in range(width):
			dst[height-1-i, j] =  src[i,j]
	return dst

def centorFlipOver(src):
	dst = src.copy()
	height,width = src.shape[:2] 
	for i in range(height):
		for j in range(width):
			dst[height-1-i, width-1-j] =  src[i,j]
	return dst



def resize(dict_img, size, thresh=-1):
    # thresh = 70, 75, 80
    for key, img in dict_img.items():
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if thresh>0:
            _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
#             img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
        dict_img[key] = img
    return dict_img
