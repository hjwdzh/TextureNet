from root import *
import os
val_ids = [l.strip() for l in open('scannet_val_ids.txt')]
directories = os.listdir(root + '/scans')
fp = open('split.sh','w')
fp.write('mkdir %s/scans_val\n'%(root))
for d in directories:
	if d in val_ids:
		fp.write('mv %s/scans/%s %s/scans_val/%s\n'%(root, d, root, d))
fp.write('mv %s/scans %s/scans_train\n'%(root,root))
fp.close()