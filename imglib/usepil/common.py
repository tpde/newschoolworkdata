#! -*- coding:utf-8 -*-
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from ...nparray import puzzel
from ...basic import *

def read_images(imgs, is_skipEX=False):
	"""get imgs dict from img_dir Whether it is gif or not

	# Arguments
		imgs: the directory cantain images or list of img files
		exception: img_dir is empty

	# Returns
		the dict of images from img_dir.

	# Raises
		ValueError: image load error.
	"""
	def getframes(img):
		return img
		mypalette = img.getpalette()
		try:
			while 1:
				img.putpalette(mypalette)
				dst = Image.new(img.mode, img.size)
				dst.paste(img)
				img.seek(img.tell()+1)
		except EOFError:
			pass
		return dst

	dict_img = dict()
	if type(imgs)==type(''):
		img_dir = imgs
		files = readdir(img_dir)
		for file in files:
			img = Image.open(os.path.join(img_dir,file))
			if os.path.splitext(file)[1] == '.gif':
				img = getframes(img)
			dict_img[file] = img

	else:
		for idx in range(len(imgs)):
			file = imgs[idx]
			try:
				img = Image.open(file)
			except IOError:
				if not is_skipEX:
					raise( IOError,'file:%s miss!!!'%file )
				else:
					continue
			if os.path.splitext(file)[1] == '.gif':
				img = getframes(img)
			dict_img[os.path.split(file)[1]] = img

	return dict_img

def write_images(dict_img, save_dir, offset=1, keylen=3, prefix='', tail='', rm=False):
	""" if you want add prefix or tail on filename, please call tools.sortdict.dict_keystandard
	"""
	rightkey = lambda keylen,index: str(index) if keylen==-1	 \
						else '%02d'%(index) if keylen==2	 \
						else '%03d'%(index) if keylen==3		\
						else '%04d'%(index) if keylen==4		\
						else '%05d'%(index)

	if rm:
		rmdir(save_dir)
	mkdir(save_dir)
	for key in dict_img:
		img = dict_img[key]
		if type(key)==type(1):
			newkey = prefix + rightkey(keylen,key+offset) + (tail or '.png')
			if type(img)==type(np.array([])):
				if np.max(img)<=1:
					img = img*255
				if len(img.shape)==3 and img.shape[2]==1:
					img = img.reshape((img.shape[0],img.shape[1]))				
				img = Image.fromarray(img.astype(np.uint8))
			filepath = os.path.join(save_dir,newkey)
			img.save( filepath )
		else:
			if type(img)==type(np.array([])):
				if np.max(img)<=1:
					img = img*255
				if len(img.shape)==3 and img.shape[2]==1:
					img = img.reshape((img.shape[0],img.shape[1]))
				img = Image.fromarray(img.astype(np.uint8))
			
			if tail!='':
				img.save( os.path.join(save_dir,prefix+os.path.splitext(key)[0]+tail) )
			else:
				img.save( os.path.join(save_dir,prefix+key) )

def puzzleImage(imgdir, rows, cols, savedir, direction='right lower', prefix=''):
	"""
		direction: enum ["right lower", "right upper", "lower right", "upper right"]
	"""
	list_path = [join(imgdir,file) for file in sorted(os.listdir(imgdir))]
	assert( len(list_path) % (rows*cols) == 0 )
	mkdir(savedir)
	
	n_split = len(list_path) // (rows*cols)
	
	img0 = Image.open(list_path[0])
	imgW, imgH = img0.size
	for split in range(n_split):
		dst = Image.new( img0.mode, (cols*imgW, rows*imgH) )
		for i,path in enumerate( list_path[split*rows*cols:(split+1)*rows*cols:] ):
			img = Image.open(path)
			if direction=="right lower":
				row, col = i//cols, i%cols
			elif direction=="right upper":
				row, col = i//cols, i%cols
				col = cols-1 - col
			elif direction=="lower right":
				row, col = i%rows, i//rows
			elif direction=="upper right":
				row, col = i%rows, i//rows
				row = rows-1 - row
			else:
				raise( TypeError )
			dst.paste(img, box=(col*imgW, row*imgH, col*imgW+imgW, row*imgH+imgH) )
		dst.save( join(savedir, prefix+'%02d.png'%split) )


def array2img(arr_imgs, listkey=None):
	def getimg(arr):
		if np.max(arr)<=1:
			arr = arr*255
		if len(arr.shape)==3 and arr.shape[2]==1:
			arr = arr.reshape(arr.shape[:2])
		return Image.fromarray(arr.astype(np.uint8))
	
	dict_dst = dict()
	if type(arr_imgs)==type(np.array([])):
		for i in range(arr_imgs.shape[0]):
			if listkey==None:
				dict_dst[i]=getimg(arr_imgs[i])
			else:
				dict_dst[listkey[i]]=getimg(arr_imgs[i])
	else:
		for key, arr in arr_imgs.items():
			dict_dst[key]=getimg(arr)
	return dict_dst


# convert -delay 50 -loop 0 *.png pic.gif

import imageio
def create_gif(list_files, savepath, duration=0.35):
	frames = []
	for file in list_files:
		frames.append(imageio.imread(file))
	imageio.mimsave(savepath, frames, 'GIF', duration=duration)
	
def rotate_img(dict_img, angle):
	for key,img in dict_img.items():
		dict_img[key] = img.rotate(angle, expand=True)
	return dict_img
