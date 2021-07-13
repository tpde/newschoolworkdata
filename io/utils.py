# -*- coding: utf-8 -*-
import os
import h5py
import pickle as pk
import pandas
from ..basic import mkdir
from .txt import readlist

def read_imgs(imgdir):
	assert os.path.isdir(imgdir), '%s is not a valid directory' % imgdir
	
	IMG_EXTENSIONS = [  '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
	                    '.ppm', '.PPM', '.bmp', '.BMP', '.tiff' ]
	
	list_files = []
	for root, _, fnames in sorted(os.walk(imgdir)):
		for fname in fnames:
			if any(fname.endswith(extension) for extension in IMG_EXTENSIONS):
				path = os.path.join(root, fname)
				list_files.append(path)
				
	assert( len(list_files)>0 )
	return list_files

def load_hdf5(filepath):
	with h5py.File(filepath,"r") as f:
		return f["image"][()]


def write_hdf5(arr,filepath):
	if len(os.path.split(filepath)[0])>0:
		mkdir(os.path.split(filepath)[0])
	with h5py.File(filepath,"w") as f:
		f.create_dataset("image", data=arr, dtype=arr.dtype)



def saveToPkl( image_array, label_array, filepath):
	write_file = open( filepath, 'wb')
	pk.dump(image_array, write_file, -1)
	pk.dump(label_array, write_file, -1)
	write_file.close()

def loadFromPkl( filepath ):
	read_file = open(filepath, 'rb')
	image_array = pk.load(read_file)
	label_array = pk.load(read_file)
	read_file.close()
	return image_array, label_array


def readfiles(list_path):
	list_all= []
	colname = ''
	for i,filepath in enumerate(list_path):
		print(i, filepath)
		assert( os.path.exists(filepath) )
		list_line = readlist( filepath )
		if colname!='':
			assert(colname==list_line[0])
		colname = list_line[0]
		# print(len(colname.split('\t')))
		list_all += list_line[1:]
	list_all = sorted(list_all, key=lambda item:item[:len('YYYY-mm-dd HH:MM:SS')] )
	list_all = [colname] + list_all
	return list_all


def extractCol(filepath, list_col, splitc='\t', is_sort=False):
	list_line = readlist( filepath )
	colname = list_line[0].split(splitc) 
	list_col = [colname.index(col) for col in list_col]
	list_data = []
	for line in list_line[1:]:
		list_v = line.split(splitc)
		assert( len(list_v)==len(colname) )
		data = [list_v[col] for col in list_col]
		list_data.append( splitc.join(data) )
	if is_sort:
		list_data = sorted(list_data, key=lambda item:item[:len('YYYY-mm-dd HH:MM:SS')] )
	return [splitc.join([colname[col] for col in list_col])] + list_data


def panda_read(path, isCSV=False):
    if isCSV:
        try:
            da = pandas.read_csv( path, engine='python', encoding='utf-8' )
        except:
            da = pandas.read_csv( path, encoding = 'gb18030' )
    else:
        try:
            da = pandas.read_excel( path, engine='python', encoding='utf-8' )
        except:
            da = pandas.read_excel( path, encoding = 'gb18030' )
    return da
