import pickle
import tensorflow as tf
import os
import random
import numpy as np
from time import time
labelweights = np.loadtxt('dataset/labelweights.txt')

def Preprocess(batch_data, batch_feature):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    if not batch_feature is None:
        rotated_feature = np.zeros(batch_feature.shape, dtype=np.float32)
    else:
        rotated_feature = None

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]], dtype='float32')
    rotated_data = np.dot(batch_data.reshape((-1, 3)), rotation_matrix)
    if not rotated_feature is None:
        rotated_feature[:,3:6] = np.dot(batch_feature[:,3:6].reshape((-1,3)), rotation_matrix)

    if not batch_feature is None:
        rotated_feature[:,0:3] = rotated_data

    return rotated_data, rotated_feature

def LoadChunkAugment(filename):
	fp = open(filename, 'rb')
	point_set=pickle.load(fp)
	feature_set=pickle.load(fp)
	cloud_pointcloud=pickle.load(fp).astype('float32')
	#cloud_pointcloud = (cloud_pointcloud / 255.0).astype('float32')
	semantic_seg=pickle.load(fp)
	sample_weight=pickle.load(fp)
	s1024=pickle.load(fp)
	s256=pickle.load(fp)
	s64=pickle.load(fp)
	s16=pickle.load(fp)
	g1024=pickle.load(fp)
	g256=pickle.load(fp)
	g64=pickle.load(fp)
	g16=pickle.load(fp)
	t1=pickle.load(fp)
	t2=pickle.load(fp)
	t3=pickle.load(fp)
	t4=pickle.load(fp)
	fp.close()
	sample_weights1 = np.zeros((8192))
	mask = sample_weight > 0
	sample_weight = mask * labelweights[semantic_seg]

	if sample_weight.dtype != np.float32:
		sample_weight = sample_weight.astype('float32')
	#sample_weight *= 1 - (semantic_seg == 0)
	#print(cloud_pointcloud.shape, point_set.shape)
	point_set, feature_set = Preprocess(point_set, feature_set)
	return point_set, feature_set, cloud_pointcloud, semantic_seg, sample_weight, s1024, s256, s64, s16, g1024, g256, g64, g16, t1, t2, t3, t4

def LoadChunk(filename):
	fp = open(filename, 'rb')
	point_set=pickle.load(fp)
	feature_set=pickle.load(fp)
	cloud_pointcloud=pickle.load(fp).astype('float32')
	#cloud_pointcloud = (cloud_pointcloud / 255.0).astype('float32')
	semantic_seg=pickle.load(fp)
	sample_weight=pickle.load(fp)
	s1024=pickle.load(fp)
	s256=pickle.load(fp)
	s64=pickle.load(fp)
	s16=pickle.load(fp)
	g1024=pickle.load(fp)
	g256=pickle.load(fp)
	g64=pickle.load(fp)
	g16=pickle.load(fp)
	t1=pickle.load(fp)
	t2=pickle.load(fp)
	t3=pickle.load(fp)
	t4=pickle.load(fp)
	fp.close()
	if sample_weight.dtype != np.float32:
		sample_weight = sample_weight.astype('float32')
	return point_set, feature_set, cloud_pointcloud, semantic_seg, sample_weight, s1024, s256, s64, s16, g1024, g256, g64, g16, t1, t2, t3, t4

def BuildDataset(batch_size=6, parent_dir='scannet-chunks', category='', chunks=100, augment=1, whole_data=0):
#def BuildDataset(batch_size=6, parent_dir='/oriong4/projects/jingweih/scannet-chunks', category='', chunks=100, augment=1, whole_data=0):
	filenames = os.listdir(parent_dir + '/' + category)
	for i in range(len(filenames)):
		filenames[i] = parent_dir + '/' + category + '/' + filenames[i]
	
	char = '_chunks_'
	if len(filenames[0].split('_chunk_')) == 2:
		char = '_chunk_'

	for f in filenames:
		if len(f.split(char)) < 2:
			print(f)
			exit(0)

	if whole_data == 0:
		records = [(f.split(char)[1], f.split(char)[0]) for f in filenames]
		records.sort()
		filenames = [f[1] + char + f[0] for f in records]
	else:
		records = [(f.split(char)[0], f.split(char)[1]) for f in filenames]
		records.sort()
		filenames = [f[0] + char + f[1] for f in records]

	chunks = len(filenames) // chunks

	if whole_data == 0:
		for i in range(0, len(filenames), chunks):
			temp = filenames[i:i+chunks]
			random.shuffle(temp)
			filenames[i:i+chunks] = temp

	#filenames.sort()
	dataset = tf.data.Dataset.from_tensor_slices(filenames)

	if augment:
		dataset = dataset.map(lambda filename: tf.py_func(LoadChunkAugment, [filename],
			[tf.float32, tf.float32, tf.float32, tf.int32, tf.float32,
			tf.int32, tf.int32, tf.int32, tf.int32,
			tf.int32, tf.int32, tf.int32, tf.int32,
			tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=4)
	else:
		dataset = dataset.map(lambda filename: tf.py_func(LoadChunk, [filename],
			[tf.float32, tf.float32, tf.float32, tf.int32, tf.float32,
			tf.int32, tf.int32, tf.int32, tf.int32,
			tf.int32, tf.int32, tf.int32, tf.int32,
			tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=4)

	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(1)

	if whole_data == 0:
		return dataset, chunks

	return dataset, chunks, filenames
