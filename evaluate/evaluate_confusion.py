import os
import numpy as np
scenes = os.listdir('../output/confusion')
scenes.sort()
VALID_CLASS_IDS = [i for i in range(1, 21)]
def get_iou(label_id, confusion):
        if not label_id in VALID_CLASS_IDS:
                return float('nan')
        tp = np.longlong(confusion[label_id, label_id])
        fn = np.longlong(confusion[label_id, 0:].sum()) - tp
        not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
        fp = np.longlong(confusion[not_ignored, label_id].sum())
        denom = (tp + fp + fn)
        if denom == 0:
                return (1, 1, 1, 1)
        return (float(tp) / denom, float(tp)/float(tp+fn+1e-6), tp, denom)

confusions = np.zeros((21, 21))
for s in scenes:
	confusion = np.loadtxt('../output/confusion/' + s)
	confusions += confusion
	average_acc = 0
	total_acc = 0
	for l in range(1, 21):
		iou, acc, _, _ = get_iou(l, confusion)
		if acc > 0:
			average_acc += acc
			total_acc += 1
	print(s, average_acc/total_acc)

mean_iou = 0
mean_acc = 0
for l in range(1, 21):
	iou, acc, _, _ = get_iou(l, confusions)
	print("Class=%d IoU=%.6f Acc=%.6f"%(l, iou, acc))
	mean_iou += iou
	mean_acc += acc

print('Mean Iou=%.6f;   Mean Acc=%.6f'%(mean_iou/20.0, mean_acc/20.0))
