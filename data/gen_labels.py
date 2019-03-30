import os
from root import parent_dir
dirs = parent_dir

fp = open('gen_labels.sh', 'w')
for d in dirs:
        scenes = os.listdir(d)
        for scene in scenes:
                if scene[0] != 's':
                        continue
                objfile = d + '/' + scene + '/' + scene + '.labels.obj'
                txtfile = d + '/' + scene + '/' + scene + '.labels.txt'
                posfile = d + '/' + scene + '/' + scene + '_textiles002.txt'
                labelfile = d + '/' + scene + '/' + scene + '_labels002.txt'
                baryfile = d + '/' + scene + '/' + scene + '_bary002.txt'
                fp.write('./maplabel ' + posfile + ' ' + objfile + ' ' + txtfile + ' ' + labelfile + ' ' + baryfile + '\n')

fp.close()
