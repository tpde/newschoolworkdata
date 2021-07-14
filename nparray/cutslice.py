#! -*- coding:utf-8 -*-
import random, math
import numpy as np
from .nputil import img2array
from .neuralKit import groZeroOne
from ..basic import *


def check_count(list_dict_img, patch_h, patch_w, target_idx=-1):
	list_names = list_dict_img[0].keys()
	list_count_points = []
	for name in list_names:
		print( name)
		img_h,img_w = list_dict_img[0].get(name).shape[:2]
		for idx in range(len(list_dict_img))[1:]:
			imarr = list_dict_img[idx].get(name)
			assert img_h == imarr.shape[0]
			assert img_w == imarr.shape[1]
			if target_idx!=-1 and idx==target_idx:
				assert len(np.unique(imarr).tolist())<=2
				list_count_points.append( np.sum(imarr>0) )
		
		if target_idx==-1:
			list_count_points.append((img_h-patch_h+1)*(img_w-patch_w+1))
			
	return list_names, list_count_points


def random_extract(list_dict_img, num_total, patch_h, patch_w):
	"""
	# attention 
		before you call this function, please make sure the key of dict_ori or dict_gro is range(len(dict_ori))
		before you call this function, please make sure:  dict_ori=img2array(dict_ori) or dict_gro=normalize(dict_gro) 
	"""
	list_names, list_count_points = check_count(list_dict_img, patch_h, patch_w, -1)

	list_list_patches = []
	for idx in range(len(list_dict_img)):
		list_list_patches.append([])
	
	for idx,name in enumerate(list_names):
		num_curr = int(num_total*list_count_points[idx]/sum(list_count_points))
		print('random_extract from:{} for {} patches!'.format(name, num_curr) )
	
		count = 0
		while(count<num_curr):
			img_h,img_w = list_dict_img[0][name].shape[:2]
			row = random.randint(0, img_h-patch_h)
			col = random.randint(0, img_w-patch_w)
			for idx_dict, dict_img in enumerate(list_dict_img):
				list_list_patches[idx_dict].append(dict_img[name][row:row+patch_h,col:col+patch_w][np.newaxis,...])
			count += 1
	
	for idx in range(len(list_dict_img)):
		list_list_patches[idx] = np.concatenate(list_list_patches[idx], axis=0)
		print('random_extract:: ',list_list_patches[idx].shape )
			
	return list_list_patches


def target_extract(list_dict_img, target_idx, num_total, patch_h, patch_w):
	"""
	# attention 
		before you call this function, please make sure the key of dict_ori or dict_gro is range(len(dict_ori))
		before you call this function, please make sure:  dict_ori=img2array(dict_ori) or dict_gro=normalize(dict_gro) 
	"""
	list_names, list_count_points = check_count(list_dict_img, patch_h, patch_w, target_idx)
	list_num_sample = [(int(100*math.log(count_points)) if count_points>0 else 0) for count_points in list_count_points ]
	
	list_list_patches = [[] for idx in range(len(list_dict_img))]
	
	for idx,name in enumerate(list_names):
		num_curr = int(num_total*list_num_sample[idx]/sum(list_num_sample))
		print('target_extract from:{} for {} patches!'.format(name, num_curr) )
	
		gro = list_dict_img[target_idx].get(name)
		img_h, img_w = gro.shape[:2]
		
		count = 0
		while(count<num_curr):
			while(True):
				row = random.randint(0, img_h-patch_h)
				col = random.randint(0, img_w-patch_w)
				if np.sum(gro[row:row+patch_h,col:col+patch_w])>0:
					break
			for idx_dict, dict_img in enumerate(list_dict_img):
				list_list_patches[idx_dict].append( dict_img[name][row:row+patch_h,col:col+patch_w][np.newaxis,...] )
			count += 1
	
	for idx in range(len(list_dict_img)):
		list_list_patches[idx] = np.concatenate(list_list_patches[idx], axis=0)
		print('target_extract:: ',list_list_patches[idx].shape )
			
	return list_list_patches


def getRightCenter(img_h, img_w, row_center, col_center, patch_h, patch_w):
	if row_center-patch_h/2<0:
		row_center=patch_h/2
	if row_center+patch_h/2>img_h:
		row_center=img_h-patch_h/2

	if col_center-patch_w/2<0:
		col_center=patch_w/2
	if col_center+patch_w/2>img_w:
		col_center=img_w-patch_w/2

	return int(row_center), int(col_center)


def grid_extract(dict_img, patch_h, patch_w, stride_h=5, stride_w=5):
	"""
	# attention 
		before you call this function, please make sure the key of dict_ori or dict_gro is range(len(dict_ori))
		before you call this function, please make sure:  dict_ori=img2array(dict_ori) or dict_gro=normalize(dict_gro) 
	"""
	# patches_array = np.zeros( (1, patch_h, patch_w, dict_img[0].shape[2]), dtype='uint8')
	dict_patches = dict()
	count = 0
	for key,imgArray in dict_img.items():
		img_h, img_w, channel = imgArray.shape
		img_newHeight, img_newWidth, cutNum_height, cutNum_width, _, _ = calculate_gridcut( 
					img_h, img_w, patch_h, patch_w, stride_h, stride_w)
		count += cutNum_height*cutNum_width
		#584 565 48 48 True 5 5 --> 588 568 109 105 44 45)
		imgArray_new = np.zeros((img_newHeight, img_newWidth, channel), dtype='uint8')
		imgArray_new[0:img_h, 0:img_w, :] =  imgArray.astype(np.uint8)
		patches_tmp = np.zeros( (cutNum_height*cutNum_width, patch_h, patch_w, channel), dtype='uint8' ) # shape = (20*10, 584, 565, 3)
		for row in range(cutNum_height):
			for col in range(cutNum_width):
				patch_index = row*cutNum_width+col
				# print('imgArray_new: imgs:',i, ' row:(', row*stride_h, '->', row*stride_h+patch_h, ') col:(', col*stride_w, '->', col*stride_w+patch_w,')'
				patches_tmp[patch_index, :, :, :] 				\
				= imgArray_new[row*stride_h:row*stride_h+patch_h, col*stride_w:col*stride_w+patch_w].astype(np.uint8)
		# patches_array = np.vstack((patches_array, patches_tmp.astype(np.uint8)))
		dict_patches[key] = patches_tmp.astype(np.uint8)

	#patches_array = patches_array[1:]
	return dict_patches


def calculate_gridcut( img_height, img_width, patch_h, patch_w, stride_h=5, stride_w=5):
	if (img_height-patch_h)%stride_h == 0:
		img_newHeight=img_height
		cutNum_height= int((img_height-patch_h)/stride_h)+1
		lastpart_validHeight= patch_h
	else:
		img_newHeight=(int((img_height-patch_h)/stride_h)+1)*stride_h+patch_h 
		cutNum_height= int((img_height-patch_h)/stride_h)+2
		lastpart_validHeight= img_height-(cutNum_height-1)*stride_h

	if (img_width-patch_w)%stride_w == 0:
		img_newWidth=img_width
		cutNum_width= int((img_width-patch_w)/stride_w)+1
		lastpart_validWidth= patch_w
	else:
		img_newWidth=(int((img_width-patch_w)/stride_w)+1)*stride_w+patch_w
		cutNum_width= int((img_width-patch_w)/stride_w)+2
		lastpart_validWidth= img_width-(cutNum_width-1)*stride_w
# 	print('caculateHowGridCutpatch::',img_height, img_width, patch_h, patch_w, stride_h, stride_w, '-->', img_newHeight, img_newWidth, cutNum_height, cutNum_width, lastpart_validHeight, lastpart_validWidth
	return img_newHeight, img_newWidth, cutNum_height, cutNum_width, lastpart_validHeight, lastpart_validWidth
