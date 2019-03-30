import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '/utils'))
import tensorflow as tf
import numpy as np
import tf_util
from texturenet_util import texture_geodesic_conv, texture_geodesic_tconv, TextureConv

def placeholder_inputs(batch_size, num_point, use_color=0):
    points = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    colors = None
    if use_color == 1:
        features = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    elif use_color == 2:
        features = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
        colors = tf.placeholder(tf.float32, shape=(batch_size, num_point, 300))
    else:
        features = None
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))

    num_neighbors = 48
    s1024, s256, s64, s16, g1024, g256, g64, g16 = None, None, None, None, None, None, None, None
    s1024 = tf.placeholder(tf.int32, shape=(batch_size, 1024))
    s256 = tf.placeholder(tf.int32, shape=(batch_size, 256))
    s64 = tf.placeholder(tf.int32, shape=(batch_size, 64))
    s16 = tf.placeholder(tf.int32, shape=(batch_size, 16))

    g1024 = tf.placeholder(tf.int32, shape=(batch_size, 8192, num_neighbors))
    g256 = tf.placeholder(tf.int32, shape=(batch_size, 1024, num_neighbors))
    g64 = tf.placeholder(tf.int32, shape=(batch_size, 256, num_neighbors))
    g16 = tf.placeholder(tf.int32, shape=(batch_size, 64, num_neighbors))

    c1, c2, c3, c4, t1, t2, t3, t4 = None, None, None, None, None, None, None, None
    c1 = tf.placeholder(tf.int32, shape=(batch_size, 8192, num_neighbors))
    c2 = tf.placeholder(tf.int32, shape=(batch_size, 1024, num_neighbors))
    c3 = tf.placeholder(tf.int32, shape=(batch_size, 256, num_neighbors))
    c4 = tf.placeholder(tf.int32, shape=(batch_size, 64, num_neighbors))

    t1 = tf.placeholder(tf.float32, shape=(batch_size, 8192, num_neighbors, 2))
    t2 = tf.placeholder(tf.float32, shape=(batch_size, 1024, num_neighbors, 2))
    t3 = tf.placeholder(tf.float32, shape=(batch_size, 256, num_neighbors, 2))
    t4 = tf.placeholder(tf.float32, shape=(batch_size, 64, num_neighbors, 2))
    return points, features, colors, labels_pl, smpws_pl, s1024, s256, s64, s16, g1024, g256, g64, g16, c1, c2, c3, c4, t1, t2, t3, t4


def get_model(point_cloud, feature_cloud, color_cloud, s1, s2, s3, s4, g1, g2, g3, g4, c1, c2, c3, c4, t1, t2, t3, t4, is_training, num_class, use_color=0, bn_decay=None):
    """ Semantic segmentation TextureNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud
    l0_points = None

    if use_color == 0:
        l0_points = None
    else:
        l0_points = feature_cloud
    if use_color == 2:
        l0_cloud = TextureConv(color_cloud, is_training, bn_decay)
        l0_points = tf.concat([l0_points,l0_cloud],axis=-1)

    # Layer 1
    l1_xyz, l1_points = texture_geodesic_conv(s1, g1, c1, t1, l0_xyz, l0_points, npoint=1024, radius=0.1, conv_radius=0.1, conv_mlp = None, nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_color=use_color)
    l2_xyz, l2_points = texture_geodesic_conv(s2, g2, c2, t2, l1_xyz, l1_points, npoint=256, radius=0.2, conv_radius=0.2, conv_mlp = None, nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2', use_color=use_color)
    l3_xyz, l3_points = texture_geodesic_conv(s3, g3, c3, t3, l2_xyz, l2_points, npoint=64, radius=0.4, conv_radius=0.4, conv_mlp = None, nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3', use_color=use_color)
    l4_xyz, l4_points = texture_geodesic_conv(s4, g4, c4, t4, l3_xyz, l3_points, npoint=16, radius=0.8, conv_radius=0.8, conv_mlp = None, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4', use_color=use_color)

    # Feature Propagation layers
    l3_points = texture_geodesic_tconv(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = texture_geodesic_tconv(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l1_points = texture_geodesic_tconv(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    l0_points = texture_geodesic_tconv(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net


def get_loss(pred, label, smpw):
    """ pred: BxNxC,
        label: BxN, 
        smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

def get_learning_rate(params, batch):
    learning_rate = tf.train.exponential_decay(
                        params.learning_rate,  # Base learning rate.
                        batch * params.batch_size,  # Current index into the dataset.
                        params.decay_step,          # Decay step.
                        params.decay_rate,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(params, batch):
    bn_momentum = tf.train.exponential_decay(
                      params.bn_init_decay,
                      batch*params.batch_size,
                      params.bn_decay_step,
                      params.bn_decay_rate,
                      staircase=True)
    bn_decay = tf.minimum(params.bn_decay_clip, 1 - bn_momentum)
    return bn_decay

def BuildNetwork(params, use_color):
    with tf.device('/gpu:0'):
        points, features, colors, labels, weights,\
        s1, s2, s3, s4,\
        g1, g2, g3, g4,\
        c1, c2, c3, c4,\
        t1, t2, t3, t4 = placeholder_inputs(params.batch_size, params.num_point, use_color=use_color)
        is_training = tf.placeholder(tf.bool, shape=())

        batch = tf.Variable(0)
        bn_decay = get_bn_decay(params, batch)

        prob = get_model(points, features, colors,
            s1, s2, s3, s4,
            g1, g2, g3, g4,
            c1, c2, c3, c4,
            t1, t2, t3, t4,
            is_training, params.num_classes, use_color=use_color, bn_decay=bn_decay)

        loss = get_loss(prob, labels, weights)
        pred = tf.argmax(prob[:,:,1:], 2)
        pred = pred + 1

        learning_rate = get_learning_rate(params, batch)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=batch)
        
        saver = tf.train.Saver()
    
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Init variables
    init = tf.global_variables_initializer()
    sess.run(init)

    ops = {'points': points,'labels': labels,'weights': weights,
           's1': s1,'s2': s2,'s3': s3,'s4': s4,
           'g1': g1,'g2': g2,'g3': g3,'g4': g4,
           'c1': c1,'c2': c2,'c3': c3,'c4': c4,
           't1': t1,'t2': t2,'t3': t3,'t4': t4,
           'pred': pred,'prob': prob,'loss': loss,
           'is_training': is_training,'train_op': train_op,
           'step': batch}

    if use_color > 0:
        ops['features'] = features
    if use_color == 2:
        ops['colors'] = colors

    return sess,saver,ops
