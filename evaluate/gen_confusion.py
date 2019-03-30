import os

scenes = os.listdir('../output/gt')

test_scenes = '/oriong5/jingweih/scannet/data/scans_val'
if not os.path.exists('../output/pred_final'):
	os.mkdir('../output/pred_final')
if not os.path.exists('../output/confusion'):
	os.mkdir('../output/confusion')
fp = open('gen_confusion.sh', 'w')
for s in scenes:
	s4 = s[:-4]
	print(test_scenes + '/' + s4 + '/' + s4 + '.labels.obj')
	if os.path.exists(test_scenes + '/' + s4 + '/' + s4 + '.labels.obj'):
		fp.write('./extract_labels ' + '../output/pos/' + s + ' ../output/pred/' + s + ' ' + test_scenes + '/' + s4 + '/' + s4 + '.labels.obj '+test_scenes+'/' + s4 + '/' + s4 + '.labels.txt ../output/confusion/' + s + ' ../output/pred_final/' + s + '\n')

fp.close()

