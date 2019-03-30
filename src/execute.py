import numpy as np
import tensorflow as tf

def Estimate(sess, ops, train_data, confusion, use_color=0, is_training=True):
    labels = train_data[3]

    feed_dict = {ops['is_training']:is_training,
        ops['points']:train_data[0],
        ops['labels']:train_data[3],
        ops['weights']:train_data[4],
        ops['s1']:train_data[5],
        ops['s2']:train_data[6],
        ops['s3']:train_data[7],
        ops['s4']:train_data[8],
        ops['g1']:train_data[9],
        ops['g2']:train_data[10],
        ops['g3']:train_data[11],
        ops['g4']:train_data[12],
        ops['t1']:train_data[13],
        ops['t2']:train_data[14],
        ops['t3']:train_data[15],
        ops['t4']:train_data[16]
    }
    if use_color == 1:
        color_data = train_data[2][:,:,(5*10+5)*3:(5*10+6)*3]
        feat_data = np.concatenate([train_data[1], color_data], axis=2)
        feed_dict[ops['features']] = feat_data
    if use_color == 2:
        feed_dict[ops['features']] = train_data[1]
        feed_dict[ops['colors']] = train_data[2]
    if is_training:
        step, _, loss, pred = sess.run([ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict
        )
    else:
        step, loss, pred = sess.run([ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict
        )       
    if confusion is not None:
        for b in range(pred.shape[0]):
            evaluate_scan(pred[b], labels[b], train_data[4][b], confusion)
    total_correct = np.count_nonzero((pred == labels) & (labels != 0))
    total_valid = np.count_nonzero(labels)

    acc = (1e-6+total_correct) / (1e-6+total_valid)
    return loss, acc, pred

def evaluate_scan(pred_ids, gt_ids, mask, confusion):
    if not pred_ids.shape == gt_ids.shape:
        print('Wrong shape!')
        exit(0)
    for i in range(gt_ids.shape[0]):
        gt_val = gt_ids[i]
        pred_val = pred_ids[i]
        if gt_val == 0 or mask[i] == 0:
            continue
        confusion[gt_val][pred_val] += 1

def get_iou(label_id, confusion, num_classes):
    tp = np.longlong(confusion[label_id, label_id])
    fn = np.longlong(confusion[label_id, 1:].sum()) - tp
    not_ignored = [l for l in range(1, num_classes) if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())
    denom = (tp + fp + fn)
    if denom == 0:
        return (1, 1, 1, 1)
    return (float(tp + 1e-6) / (denom +1e-6), float(tp + 1e-6)/float(tp+fn+1e-6), tp, denom)

def Evaluate(sess, ops, val_el, eval_epoch=0, eval_epoch_iters=0, num_classes=21, use_color=1):
    confusion = np.zeros((num_classes, num_classes))
    eval_acc = 0
    for eval_it in range(eval_epoch_iters):
        val_data = sess.run(val_el)
        loss, acc, pred = Estimate(sess=sess, ops=ops, train_data=val_data, confusion=confusion, use_color=use_color, is_training=False)
        print('eval iter=', eval_it, '  loss=%.6f'%loss, '  acc=%.6f'%acc)
        eval_acc += acc
        break

    logs = '================================================================\n'
    logs += 'Evaluation Epoch %03d, Acc=%.6f\n'%(eval_epoch, eval_acc/300.0)

    mean_iou = 0
    mean_acc = 0
    for i in range(1, num_classes):
        iou, acc, _, _ = get_iou(i, confusion, num_classes)
        mean_iou += iou
        mean_acc += acc
        logs += 'Class %03d:     IoU=%.6f     Acc=%.6f\n'%(i, iou, acc)

    mean_acc /= 20.0
    mean_iou /= 20.0
    return logs, mean_acc, mean_iou


def EvaluateWhole(sess, ops, val_whole_el, eval_whole_epoch=0, eval_whole_epoch_iters=0, num_classes=21, use_color=1, record_result=0, filenames=None):
    confusion = np.zeros((num_classes, num_classes))
    eval_acc = 0

    pred_positions = {}
    pred_labels = {}
    gt_labels = {}

    for eval_it in range(eval_whole_epoch_iters):
        val_data = sess.run(val_whole_el)
        loss, acc, pred = Estimate(sess=sess, ops=ops, train_data=val_data, confusion=confusion, use_color=use_color, is_training=False)

        if record_result == 1:
            scene = filenames[eval_it].split('/')[-1].split('_chunk')[0]
            if not scene in pred_positions:
                pred_positions[scene] = np.zeros((0, 3))
                pred_labels[scene] = np.zeros((0), dtype='int32')
                gt_labels[scene] = np.zeros((0), dtype='int32')

            pts = []
            gt = []
            preds = []
            
            for i in range(val_data[0].shape[1]):
                if val_data[4][0,i] > 0:
                    pts.append(val_data[0][0,i])
                    gt.append(val_data[3][0,i])
                    preds.append(pred[0,i])

            pts = np.array(pts)
            gt = np.array(gt, dtype='int32')
            preds = np.array(preds, dtype='int32')


            pred_positions[scene] = np.concatenate([pred_positions[scene], pts], axis=0)
            pred_labels[scene] = np.concatenate([pred_labels[scene], preds], axis=0)
            gt_labels[scene] = np.concatenate([gt_labels[scene], gt], axis=0)

            print(pts.shape, pred_positions[scene].shape)

        print('eval_whole iter=', eval_it, '  loss=%.6f'%loss, '  acc=%.6f'%acc)
        eval_acc += acc

    logs = ('================================================================\n')
    logs += 'Evaluation Whole Scene Epoch %03d, Acc=%.6f\n'%(eval_whole_epoch, eval_acc/300.0)

    mean_iou = 0
    mean_acc = 0
    for i in range(1, num_classes):
        iou, acc, _, _ = get_iou(i, confusion, num_classes)
        mean_iou += iou
        mean_acc += acc
        logs += 'Class %03d:     IoU=%.6f     Acc=%.6f\n'%(i, iou, acc)

    mean_acc /= 20.0
    mean_iou /= 20.0

    return logs, mean_acc, mean_iou, pred_positions, pred_labels, gt_labels
