import os
from root import root
parent_dir = root
output_dir = root + '/scannet-chunks'
try:
	os.mkdir(output_dir)
except:
	print('exception: mkdir %s'%(output_dir))

dirs = ['scans_train', 'scans_val', 'scans_val', 'scans_test']
splits = ['train', 'val', 'val', 'test']
out_dirs = ['scannet-chunks-train', 'scannet-chunks-val', 'scannet-chunks-val-whole', 'scannet-chunks-test-whole']

chunks = [100, 20, 1, 1]
use_whole = [0, 0, 1, 1]

fp = open('gen_chunks.sh', 'w')
for i in range(0,4):
	path = parent_dir + '/' + dirs[i]
	scenes = os.listdir(path)
	for s in scenes:
		fp.write('python prechunk.py ' + path + ' ' + splits[i] + ' ' + str(chunks[i]) + ' ' + str(use_whole[i]) + ' ' + output_dir + '/' + out_dirs[i] + ' ' + s + '\n')

fp.close()
