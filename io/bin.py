# -*- coding: utf-8 -*-
import os
from mylibs import *
import tensorflow as tf

# cifar10数据集的格式，训练样例集和测试样例集分别为50k和10k
# train_samples_per_epoch = 50000
# test_samples_per_epoch = 10000


def readFromBin(data_dir, is_train, fixed_size):
	filenames = os.listdir(data_dir)
	assert len( filenames )!=0
	filepaths = [os.path.join(data_dir, filename) for filename in filenames]
	for filepath in filepaths:
		print(filepath)
		if not tf.gfile.Exists(filepath):
			raise ValueError('fail to find file:'+filepath)

	filename_queue = tf.train.string_input_producer(string_tensor=filepaths)
	bytes_label = 1
	bytes_image = 32*32*3
	bytes_record = bytes_label+bytes_image
	reader = tf.FixedLengthRecordReader(record_bytes=bytes_record)
	key, value_str = reader.read(filename_queue)
	value_bytes = tf.decode_raw(bytes=value_str, out_type=tf.uint8)
	label = tf.slice(input_=value_bytes,begin=[0],size=[bytes_label])
	image_bytes = tf.slice(input_=value_bytes, begin=[bytes_label], size=[bytes_image])
	image_reshape = tf.reshape(image_bytes, [3, 32, 32])
	image_transpose = tf.transpose(image_reshape, perm=[1, 2, 0])
	image = tf.cast(image_transpose, tf.float32)

	tf.image_summary('input_image', tf.reshape(image, [1, 32, 32, 3]))
	image = tf.random_crop(image, size=(fixed_size, fixed_size, 3))
	if is_train:
		image = tf.image.random_brightness(image, max_delta=63)
		image = tf.image.random_flip_left_right(image)
		image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
	image = tf.image.per_image_whitening(image)

	return image, label