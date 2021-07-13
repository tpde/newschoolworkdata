# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.misc
import tensorflow as tf
from PIL import Image
from ..basic import *
# import adjust_pic as ap


IMAGE_SIZE = 64
IMAGE_CHANNEL=3
LABEL_SIZE = 256
LABEL_CHANNEL=1

def inputs(tf_dir, read_fname, batch_size, capacity):
	assert( read_fname in ['cint', 'cimg', 'unc'])
	f_read = cint_readtfrecord if read_fname == 'cint'	\
	else cimg_readtfrecord if read_fname == 'cimg'		\
	else unc_readtfrecord

	if read_fname in ['cint', 'cimg']:
		image, label = f_read(tf_dir)
		batch_image, batch_label = tf.train.shuffle_batch([image, label],
											batch_size = batch_size,
											num_threads = 4,
											capacity = capacity + 3 * batch_size,
											min_after_dequeue = capacity)
		return batch_image, batch_label
	else:
		image = f_read(tf_dir)
		batch_image = tf.train.shuffle_batch([image],
											batch_size = batch_size,
											num_threads = 4,
											capacity = capacity + 3 * batch_size,
											min_after_dequeue = capacity)	

		return batch_image


def myresize(img_data, width, high, method=0):  
	return tf.image.resize_images(img_data,[width, high], method)  

## -------------------- label is int64 (mnist) -----------------------
def cint_write2tfrecord( imgsArray, labels, saveDir, partsNum):
	assert len(imgsArray.shape)==3 or 4
	assert imgsArray.shape[0]%partsNum == 0
	if np.max(imgsArray)<=1:
		imgsArray = (imgsArray*255).astype(np.uint8)
	perpart = imgsArray.shape[0]/partsNum
	for i in range(partsNum):
		writer = tf.python_io.TFRecordWriter( os.path.join(saveDir,'%03d.tfrecords'%(i+1)) )
		for j in range(perpart):
			imgarray = imgsArray[i*perpart+j]
			label = labels[i*perpart+j]

			if len(imgarray.shape)==3 and imgarray.shape[2]==1:
				imgarray = imgarray.reshape(imgarray.shape[:2])
			image = Image.fromarray(np.uint8(imgarray))
			# image = myresize(image, width, high)
			if len(label.shape)==2:
				label = np.argmax(label, 0)
			label = label.astype(np.uint8)
			example = tf.train.Example( features=tf.train.Features( feature={
				'image': tf.train.Feature( bytes_list=tf.train.BytesList(value=[image.tobytes()])),
				# 'label': tf.train.Feature( int64_list=tf.train.Int64List(value=[label])), 	# if type(label) == int 
				'label': tf.train.Feature( bytes_list=tf.train.BytesList(value=[label.tobytes()]))
				}) )
			writer.write( example.SerializeToString() )
		writer.close()

def cint_readtfrecord( tf_dir, name='input'):
	files = os.listdir(tf_dir)
	assert len( files )!=0
	with tf.name_scope(name):
		files = sorted( files, key=(lambda file:int(file[0:file.find('.tfrecords')])) )
		filepaths = [os.path.join(tf_dir, file) for file in files]
		filename_queue =tf.train.string_input_producer(filepaths)
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example( serialized_example, features={
							'image': tf.FixedLenFeature([], tf.string),
							# 'label': tf.FixedLenFeature([], tf.int64),
							'label': tf.FixedLenFeature([], tf.string)
							})
		image = tf.decode_raw( features['image'], tf.uint8 )
		image = tf.reshape( image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL] )
		image = tf.cast(image, tf.float32)
		image = image/tf.reduce_max(image) - 0.5
		
		label = tf.decode_raw( features['label'], tf.uint8 ) # tf.int32
		label = tf.reshape( label, [1])
		return image, label


## ------------ label is image bytes  ----------------
def cimg_write2tfRecord( images, labels, saveDir, partsNum):
	assert images.shape[0]%partsNum == 0
	perpart = images.shape[0]/partsNum
	for i in range(partsNum):
		writer = tf.python_io.TFRecordWriter( os.path.join(saveDir,'%03d.tfrecords'%(i+1)) )
		for j in range(perpart):
			image = images[i*perpart+j]
			label = labels[i*perpart+j]
			example = tf.train.Example( features=tf.train.Features( feature={
				'image': tf.train.Feature( bytes_list=tf.train.BytesList(value=[image.tobytes()])),
				'label': tf.train.Feature( bytes_list=tf.train.BytesList(value=[label.tobytes()]))
				}) )
			writer.write( example.SerializeToString() )
		writer.close()

def cimg_readtfrecord( tf_dir, name='input'):
	files = os.listdir(tf_dir)
	assert len( files )!=0
	with tf.name_scope(name):
		files = sorted( files, key=(lambda filename:int(filename[0:3])) )
		filepaths = [os.path.join(tf_dir, file) for file in files]
		
		filename_queue = tf.train.string_input_producer(filepaths)
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example( serialized_example, features={
							'image': tf.FixedLenFeature([], tf.string),
							'label': tf.FixedLenFeature([], tf.string)
							})
		image = tf.decode_raw( features['image'], tf.uint8 )
		label = tf.decode_raw( features['label'], tf.uint8 )
		# myprint( tf.shape( image ),image.get_shape(),label.get_shape() )
		image = tf.reshape( image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL] )
		image = tf.cast(image, tf.float32) * (1./tf.reduce_max(image)) - 0.5

		label = tf.reshape( label, [LABEL_SIZE, LABEL_SIZE, LABEL_CHANNEL] )
		label = tf.cast(label, tf.float32)
		if tf.reduce_max(label)>=1:
			label = label/tf.reduce_max(label)
		return image, label


# ----------------- uncondition  ---------------
def unc_img2tfrecord( img_dir, save_dir, num_part, size, offset=1):
	'''
	save the imgs to *.tfrecord 
	tailname: '.tfrecords'
	dstSize:  (512, 512)
	'''
	assert len(img_dir)!=0
	mkdir(save_dir)
	
	imgsNum = len(os.listdir(img_dir))
	assert imgsNum%num_part == 0
	perpart = imgsNum/num_part
	for i in range(num_part):
# 		rmdir(os.path.join(save_dir,'%03d.tfrecords'%(i+offset)))
		writer = tf.python_io.TFRecordWriter( os.path.join(save_dir,'%03d.tfrecords'%(i+offset)) )
		for file in os.listdir(img_dir)[ i*perpart : (i+1)*perpart ]:
			image = Image.open( os.path.join(img_dir,file) )
			image = image.resize( size )
			example = tf.train.Example( features=tf.train.Features( feature={
				'width': tf.train.Feature( int64_list=tf.train.Int64List(value=[image.size[0]])),
				'height':tf.train.Feature( int64_list=tf.train.Int64List(value=[image.size[1]])),
				'depth': tf.train.Feature( int64_list=tf.train.Int64List(value=[len(image.getbands())])),
				'image':  tf.train.Feature( bytes_list=tf.train.BytesList(value=[image.tobytes()])),
				}) )
			writer.write( example.SerializeToString() )
		writer.close()

def unc_array2tfrecord( imgsArray, save_dir, num_part, offset=1):
	mkdir(save_dir)
	assert len(imgsArray.shape)==3 or 4
	assert imgsArray.shape[0]%num_part == 0
	if np.max(imgsArray)<=1:
		imgsArray = (imgsArray*255).astype(np.uint8)
	else:
		imgsArray = imgsArray.astype(np.uint8)
	perpart = imgsArray.shape[0]/num_part
	for idx_part in range(num_part):
# 		rmdir(os.path.join(save_dir,'%03d.tfrecords'%(i+offset)))
		writer = tf.python_io.TFRecordWriter( os.path.join(save_dir,'%03d.tfrecords'%(idx_part+offset)) )
		for idx in range(perpart):
			imarr = imgsArray[idx_part*perpart+idx]
			example = tf.train.Example( features=tf.train.Features( feature={
				'image':  tf.train.Feature( bytes_list=tf.train.BytesList(value=[imarr.tobytes()])),
				}) )
			writer.write( example.SerializeToString() )
		writer.close()



def unc_readtfrecord( tf_dir, name='input'):
	files = os.listdir(tf_dir)
	assert len( files )!=0
	with tf.name_scope(name):
		files = sorted( files, key=(lambda filename:int(filename[0:3])) )
		filepaths = [os.path.join(tf_dir, file) for file in files]

		filename_queue = tf.train.string_input_producer(filepaths)
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example( serialized_example, features={
												'image': tf.FixedLenFeature([], tf.string)
												})
		image = tf.decode_raw( features['image'], tf.uint8 )

		image = tf.reshape( image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL] )
		image = tf.cast(image, tf.float32)
		image = image*(1./tf.reduce_max(image)) - 0.5

		return image



# ---------------------------  condition  ---------------------
# def writeTotfRecord_cond( imageDir, labelDir, saveDir, partsNum, dstSize):
# 	'''
# 	dstSize:  (512, 512)
# 	'''
# 	assert len(os.listdir(imageDir))!=0
# 	assert len(os.listdir(labelDir))!=0
# 
# 	imgsNum = len(os.listdir(imageDir))
# 	assert imgsNum%partsNum == 0
# 	perpart = imgsNum/partsNum
# 	for i in range(partsNum):
# 		writer = tf.python_io.TFRecordWriter( os.path.join(saveDir,'%03d.tfrecords'%(i+1)) )
# 		for imgName in os.listdir(imageDir)[ i*perpart : (i+1)*perpart ]:
# 			image = Image.open(imageDir+imgName)
# 			image = image.resize( dstSize )
# 
# 			label = Image.open(labelDir+imgName)
# 			label = label.resize( dstSize )
# 			label_w, label_h = label.size[:2]
# 			example = tf.train.Example( features=tf.train.Features( feature={
# 				'image':  tf.train.Feature( bytes_list=tf.train.BytesList(value=[image.tobytes()])),
# 				'label1':  tf.train.Feature( bytes_list=tf.train.BytesList(value=[label.tobytes()])),
# 				'label2':  tf.train.Feature( bytes_list=tf.train.BytesList(value=[label.resize((label_w/2, label_h/2)).tobytes()])),
# 				'label3':  tf.train.Feature( bytes_list=tf.train.BytesList(value=[label.resize((label_w/4, label_h/4)).tobytes()])),
# 				'label4':  tf.train.Feature( bytes_list=tf.train.BytesList(value=[label.resize((label_w/8, label_h/8)).tobytes()])),
# 				'label5':  tf.train.Feature( bytes_list=tf.train.BytesList(value=[label.resize((label_w/16, label_h/16)).tobytes()])),
# 				'label6':  tf.train.Feature( bytes_list=tf.train.BytesList(value=[label.resize((label_w/32, label_h/32)).tobytes()]))
# 				}) )
# 			writer.write( example.SerializeToString() )
# 		writer.close()
# 
# def readFromTfRecord_cond( tf_dir, name='input'):
# 	files = os.listdir(tf_dir)
# 	assert len( files )!=0
# 	with tf.name_scope(name):
# 		files = sorted( files, key=(lambda filename:int(filename[0:3])) )
# 		filepaths = [os.path.join(tf_dir, file) for file in files]
# 
# 		filename_queue = tf.train.string_input_producer(filepaths)
# 		reader = tf.TFRecordReader()
# 		_, serialized_example = reader.read(filename_queue)
# 		features = tf.parse_single_example( serialized_example, features={
# 							'image': tf.FixedLenFeature([], tf.string),
# 							'label1':tf.FixedLenFeature([], tf.string),
# 							'label2':tf.FixedLenFeature([], tf.string),
# 							'label3':tf.FixedLenFeature([], tf.string),
# 							'label4':tf.FixedLenFeature([], tf.string),
# 							'label5':tf.FixedLenFeature([], tf.string),
# 							'label6':tf.FixedLenFeature([], tf.string)
# 							})
# 		image = tf.decode_raw( features['image'], tf.uint8 )
# 		label1 = tf.decode_raw( features['label1'], tf.uint8 )
# 		label2 = tf.decode_raw( features['label2'], tf.uint8 )
# 		label3 = tf.decode_raw( features['label3'], tf.uint8 )
# 		label4 = tf.decode_raw( features['label4'], tf.uint8 )
# 		label5 = tf.decode_raw( features['label5'], tf.uint8 )
# 		label6 = tf.decode_raw( features['label6'], tf.uint8 )
# 		debug( label1.get_shape() )
# 		debug( label6.get_shape() )
# 		image = tf.reshape( image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL] )
# 		image = tf.cast(image, tf.float32) * (1./tf.cast(tf.reduce_max(image), tf.float32)) - 0.5
# 
# 		if tf.reduce_max(label1)!=1 and tf.reduce_max(label1)!=0:
# 			label1 = tf.reshape( label1, [IMAGE_SIZE, IMAGE_SIZE, 1] )
# 			label2 = tf.reshape( label2, [IMAGE_SIZE/2, IMAGE_SIZE/2, 1] )
# 			label3 = tf.reshape( label3, [IMAGE_SIZE/4, IMAGE_SIZE/4, 1] )
# 			label4 = tf.reshape( label4, [IMAGE_SIZE/8, IMAGE_SIZE/8, 1] )
# 			label5 = tf.reshape( label5, [IMAGE_SIZE/16, IMAGE_SIZE/16, 1] )
# 			label6 = tf.reshape( label6, [IMAGE_SIZE/32, IMAGE_SIZE/32, 1] )
# 			label1 = tf.cast(label1, tf.float32) * (1./tf.cast(tf.reduce_max(label1), tf.float32))
# 			label2 = tf.cast(label2, tf.float32) * (1./tf.cast(tf.reduce_max(label1), tf.float32))
# 			label3 = tf.cast(label3, tf.float32) * (1./tf.cast(tf.reduce_max(label1), tf.float32))
# 			label4 = tf.cast(label4, tf.float32) * (1./tf.cast(tf.reduce_max(label1), tf.float32))
# 			label5 = tf.cast(label5, tf.float32) * (1./tf.cast(tf.reduce_max(label1), tf.float32))
# 			label6 = tf.cast(label6, tf.float32) * (1./tf.cast(tf.reduce_max(label1), tf.float32))
# 
# 		return image, label1, label2, label3, label4, label5, label6
