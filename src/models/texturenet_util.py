""" TextureNet Layers

Author: Jingwei Huang
Date: November 2019
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/interpolation'))
from tf_sampling import gather_point
from tf_grouping import group_point, query_tangent_point_level
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util

def TextureConv(color_cloud,is_training, bn_decay):
    #10x10->8x8->4x4
    #4x4->2x2->1x1
    data_format = 'NHWC'
    bdwidth = [10, 4]
    channels = [3, 4, 8]
    idx = 0
    for w in bdwidth:
        img = tf.reshape(color_cloud, shape=(color_cloud.shape[0], color_cloud.shape[1], w, w, channels[idx]))
        img1 = img[:,:,0:w-2,0:w-2,:] + img[:,:,2:,0:w-2,:] + img[:,:,0:w-2,2:,:] + img[:,:,2:,2:,:]
        img2 = img[:,:,1:w-1,0:w-2,:] + img[:,:,1:w-1,2:,:] + img[:,:,0:w-2,1:w-1,:] + img[:,:,2:,1:w-1,:]
        img3 = img[:,:,1:w-1,1:w-1,:]
        img1 = tf.reshape(img1, shape=(img1.shape[0], img1.shape[1], -1, channels[idx]))
        img2 = tf.reshape(img2, shape=(img2.shape[0], img2.shape[1], -1, channels[idx]))
        img3 = tf.reshape(img3, shape=(img3.shape[0], img3.shape[1], -1, channels[idx]))

        num_out_channel = channels[idx + 1]
        img1 = tf_util.conv2d(img1, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=False, is_training=is_training,
                                    scope='texconv1_%d'%(idx), bn_decay=None,
                                    activation_fn=None,
                                    data_format=data_format) 

        img2 = tf_util.conv2d(img2, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=False, is_training=is_training,
                                    scope='texconv2_%d'%(idx), bn_decay=None,
                                    activation_fn=None,
                                    data_format=data_format) 

        img3 = tf_util.conv2d(img3, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=False, is_training=is_training,
                                    scope='texconv3_%d'%(idx), bn_decay=None,
                                    activation_fn=None,
                                    data_format=data_format)

        signal = img1 + img2 + img3
        s = (img1 + img2) * 0.25 + img2
        with tf.variable_scope('texconv_bn_%d'%(idx)) as sc:
            signal = tf.nn.relu(tf_util.batch_norm_for_conv2d(s, is_training,
                                            bn_decay=bn_decay, scope='bn',
                                            data_format=data_format))
        signal = tf.reshape(signal, shape=(signal.shape[0] * signal.shape[1], w - 2, w - 2, num_out_channel))
        signal = tf_util.max_pool2d(signal,
               [2, 2],
               scope='texconv_pool_'+str(idx))
        color_cloud = tf.reshape(signal, shape=(color_cloud.shape[0], color_cloud.shape[1], signal.shape[1] * signal.shape[2] * signal.shape[3]))
        idx += 1
    return color_cloud

def sample_and_group_level(npoint, radius, nsample, sample, group, xyz, points, tangent, knn=False, use_xyz=True):
    idx = sample
    new_xyz = xyz
    idx1, idx2, idx3, pts_cnt = query_tangent_point_level(radius, nsample, tangent, group)
    
    grouped_xyz_1 = group_point(xyz, idx1, 1) # (batch_size, npoint, nsample, 3)
    grouped_xyz_1 -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization

    grouped_xyz_2 = group_point(xyz, idx2, 1) # (batch_size, npoint, nsample, 3)
    grouped_xyz_2 -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization

    grouped_xyz_3 = group_point(xyz, idx3, 1) # (batch_size, npoint, nsample, 3)
    grouped_xyz_3 -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization

    if points is not None:
        grouped_points_1 = group_point(points, idx1, 0) # (batch_size, npoint, nsample, channel)
        grouped_points_2 = group_point(points, idx2, 0) # (batch_size, npoint, nsample, channel)
        grouped_points_3 = group_point(points, idx3, 0) # (batch_size, npoint, nsample, channel)

        if use_xyz:
            new_points_1 = tf.concat([grouped_xyz_1, grouped_points_1], axis=-1) # (batch_size, npoint, nample, 3+channel)
            new_points_2 = tf.concat([grouped_xyz_2, grouped_points_2], axis=-1) # (batch_size, npoint, nample, 3+channel)
            new_points_3 = tf.concat([grouped_xyz_3, grouped_points_3], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points_1 = grouped_points_1
            new_points_2 = grouped_points_2
            new_points_3 = grouped_points_3
    else:
        new_points_1 = grouped_xyz_1
        new_points_2 = grouped_xyz_2
        new_points_3 = grouped_xyz_3

    return new_xyz, new_points_1, new_points_2, new_points_3, idx1, idx2, idx3, grouped_xyz_1, grouped_xyz_2, grouped_xyz_3


def texture_geodesic_conv(sample, group, conv, tex, xyz, points, npoint, radius, conv_radius, conv_mlp, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False, use_color=0):
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            if use_xyz == True:
                new_xyz, new_p1, new_p2, new_p3, idx1, idx2, idx3, grouped_xyz1, grouped_xyz2, grouped_xyz3 = sample_and_group_level(npoint, radius, nsample//2, sample, group, xyz, points, tex, knn, use_xyz) 
            else:
                new_p1 = group_point(points, idx1, 0) # (batch_size, npoint, nsample, channel)
                new_p2 = group_point(points, idx2, 0) # (batch_size, npoint, nsample, channel)
                new_p3 = group_point(points, idx3, 0) # (batch_size, npoint, nsample, channel)

            # Point Feature Embedding
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])

            new_p1 = tf_util.conv2d(new_p1, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv1_%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
            new_p2 = tf_util.conv2d(new_p2, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv2_%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format) 
            new_p3 = tf_util.conv2d(new_p3, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv3_%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)

            new_points = tf.concat([new_p1, new_p2, new_p3], axis=2)

            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

            # Pooling in Local Regions
            if pooling=='max':
                new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            elif pooling=='avg':
                new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            elif pooling=='weighted_avg':
                with tf.variable_scope('weighted_avg'):
                    dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                    exp_dists = tf.exp(-dists * 5)
                    weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                    new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                    new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
            elif pooling=='max_and_avg':
                max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
                avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
                new_points = tf.concat([avg_points, max_points], axis=-1)

            # [Optional] Further Processing 
            if mlp2 is not None:
                if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
                for i, num_out_channel in enumerate(mlp2):
                    new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1],
                                                bn=bn, is_training=is_training,
                                                scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                                data_format=data_format) 
                if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

            new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
            points = new_points

        downsample_idx = sample
        new_xyz = gather_point(new_xyz, downsample_idx)
        new_points = group_point(new_points, tf.reshape(downsample_idx, (downsample_idx.shape[0], downsample_idx.shape[1], 1)), 0)
        new_points = tf.reshape(new_points, (new_points.shape[0], new_points.shape[1], new_points.shape[3]))

        return new_xyz, new_points
 
def texture_geodesic_tconv(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1
