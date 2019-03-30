import os
from root import parent_dir
dirs = parent_dir
fp = open('gen_objs.sh', 'w')
for d in dirs:
        scenes = os.listdir(d)
        for scene in scenes:
                if scene[0] != 's':
                        continue
                flag = 0
                plyfile = d + '/' + scene + '/' + scene + '_vh_clean_2.labels.ply'
                if not os.path.exists(plyfile):
                	flag = 1
                	plyfile = d + '/' + scene + '/' + scene + '_vh_clean_2.ply'
                objfile = d + '/' + scene + '/' + scene + '.labels.obj'
                labelfile = d + '/' + scene + '/' + scene + '.labels.txt'
                fp.write('python gen_obj.py ' + plyfile + ' ' + objfile + ' ' + labelfile + ' %d\n'%(flag))
fp.close()
