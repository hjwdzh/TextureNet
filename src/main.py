import argparse
import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import models.texturenet as texturenet
from models.texturenet_param import Params
import dataset.dataset as Dataset
from execute import Estimate, Evaluate, EvaluateWhole

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='', help='Directory to save logs and checkpoint models [default: empty]')
parser.add_argument('--use_color', type=int, default=1, help='0=pure geometry, 1=low-res color, 2=high-res color')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--resume', default='', help='Checkpoint to load')
parser.add_argument('--evaluate', default=0, type=int, help='0=train. 1=output results for evaluation set. 2=output results for testing set')
parser.add_argument('--dataset', default='/oriong4/projects/jingweih/scannet-chunks', help='Directory that holds the chunked data')
parser.add_argument('--start_epoch', default=0, type=int, help='staring epoch')
parser.add_argument('--max_epoch', type=int, default=5000000, help='Epoch to run [default: 5000000]')

parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()
if FLAGS.evaluate >= 1:
    FLAGS.batch_size = 1

MAX_EPOCH = FLAGS.max_epoch

params = Params(num_point=8192, num_classes=21, batch_size=FLAGS.batch_size,
    learning_rate=FLAGS.learning_rate,momentum=FLAGS.momentum,
    decay_step=FLAGS.decay_step,decay_rate=FLAGS.decay_rate,
    bn_init_decay=0.5,bn_decay_rate=0.5,bn_decay_step=FLAGS.decay_step,bn_decay_clip=0.99)

LOG_DIR = FLAGS.log_dir
if LOG_DIR!='' and not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
if LOG_DIR!='':
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')

# Shapenet official train/test split
TRAIN_DATASET = None
TEST_DATASET = None
TEST_DATASET_WHOLE_SCENE = None

def log_string(out_str):
    if LOG_DIR != '':
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
    print(out_str)

def train():
    with tf.Graph().as_default():
        train_dataset, train_epoch_iters = Dataset.BuildDataset(batch_size=params.batch_size, parent_dir=FLAGS.dataset, category='scannet-chunks-train', chunks=100, augment=1)
        train_iter = train_dataset.make_initializable_iterator() # create the iterator
        train_el = train_iter.get_next()

        val_dataset, eval_epoch_iters = Dataset.BuildDataset(batch_size=params.batch_size, parent_dir=FLAGS.dataset, category='scannet-chunks-val', chunks=20, augment=1)
        val_iter = val_dataset.make_initializable_iterator() # create the iterator
        val_el = val_iter.get_next()

        if FLAGS.evaluate <= 1:
            val_whole_dataset, eval_whole_epoch_iters, val_filenames = Dataset.BuildDataset(batch_size=params.batch_size, parent_dir=FLAGS.dataset, category='scannet-chunks-val-whole', chunks=1, augment=0, whole_data = 1)
        else:
            val_whole_dataset, eval_whole_epoch_iters, val_filenames = Dataset.BuildDataset(batch_size=params.batch_size, parent_dir=FLAGS.dataset, category='scannet-chunks-test-whole', chunks=1, augment=0, whole_data = 1)
        val_whole_iter = val_whole_dataset.make_initializable_iterator() # create the iterator
        val_whole_el = val_whole_iter.get_next()

        train_epoch_iters = train_epoch_iters // params.batch_size
        eval_epoch_iters = eval_epoch_iters // params.batch_size
        eval_whole_epoch_iters = eval_whole_epoch_iters // params.batch_size

        sess, saver, ops = texturenet.BuildNetwork(params, FLAGS.use_color)
        # Run it
        sess.run(train_iter.initializer)
        sess.run(val_iter.initializer)
        sess.run(val_whole_iter.initializer)

        print('Start training...')
        train_epoch = FLAGS.start_epoch
        eval_epoch = 0
        eval_whole_epoch = 0
        train_acc = 0

        max_mean_class_acc = 0
        max_mean_class_iou = 0

        if FLAGS.resume != '':
            saver.restore(sess, FLAGS.resume)
            print("Restore finished")

        cached_test_data = []
        pred_positions = {}
        pred_labels = {}
        gt_labels = {}
        for train_it in range(FLAGS.start_epoch * train_epoch_iters, MAX_EPOCH):
            if train_it % train_epoch_iters == 0:
                logs = 'Training Epoch %03d, Acc=%.6f'%(train_epoch, train_acc/300.0)
                log_string(logs)
                train_epoch += 1
                train_acc = 0
                if LOG_DIR != '':
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string("Model saved in file: %s" % save_path)

            if train_it % (train_epoch_iters*5) == 0 and (train_it >= 0 or FLAGS.evaluate == 1):
                # evaluation
                if FLAGS.evaluate == 0:
                    logs, mean_acc, mean_iou = Evaluate(sess, ops, val_el, eval_epoch=eval_epoch, eval_epoch_iters=eval_epoch_iters, num_classes=params.num_classes, use_color=FLAGS.use_color)
                    log_string(logs)
                    log_string('mAcc=%.6f, mIoU=%.6f, Best mAcc=%.6f, Best mIoU=%.6f\n'%(mean_acc, mean_iou, max_mean_class_acc, max_mean_class_iou))

                    eval_epoch += 1

                if train_it // (train_epoch_iters) > 200 or train_it % (train_epoch_iters * 20) == 0:
                    logs, mean_acc, mean_iou, pred_positions, pred_labels, gt_labels = EvaluateWhole(sess, ops, val_whole_el,
                        eval_whole_epoch=eval_whole_epoch, eval_whole_epoch_iters=eval_whole_epoch_iters,
                        num_classes=params.num_classes, use_color=FLAGS.use_color, record_result=(FLAGS.evaluate > 0), filenames=val_filenames) 
                    eval_whole_epoch += 1

                    is_best = 0
                    if mean_iou > max_mean_class_iou:
                        max_mean_class_iou = mean_iou
                        is_best = 1
                    if mean_acc > max_mean_class_acc:
                        max_mean_class_acc = mean_acc
                        is_best = 1

                    log_string('mAcc=%.6f, mIoU=%.6f, Best mAcc=%.6f, Best mIoU=%.6f'%(mean_acc, mean_iou, max_mean_class_acc, max_mean_class_iou))

                    if is_best:
                        if LOG_DIR != '':
                            save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_%03d.ckpt"%(train_it//train_epoch_iters)))
                            log_string("Model saved in file: %s" % save_path)

                    if FLAGS.evaluate == 1:
                        if not os.path.exists("../output"):
                            os.mkdir('../output')
                            os.mkdir('../output/pred')
                            os.mkdir('../output/gt')
                            os.mkdir('../output/pos')
                        for scene, pos in pred_positions.items():
                            print(scene)
                            np.savetxt('../output/pos/' + scene + '.txt', pos, fmt='%.6f')
                            np.savetxt('../output/pred/' + scene + '.txt', pred_labels[scene], fmt='%d')
                            np.savetxt('../output/gt/' + scene + '.txt', gt_labels[scene], fmt='%d')
                        exit(0)
                log_string('================================================================')

            train_data = sess.run(train_el)
            loss, acc, pred = Estimate(sess=sess, ops=ops, train_data=train_data, use_color=FLAGS.use_color, confusion=None)
            print('train iter=', train_it, '  loss=%.6f'%loss, '  acc=%.6f'%acc)
            train_acc += acc

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
