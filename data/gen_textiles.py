import os
from root import parent_dir
dirs = parent_dir
fp = open('gen_textiles.sh', 'w')
for d in dirs:
        scenes = os.listdir(d)
        for scene in scenes:
                if scene[0]!='s':
                        continue
                objfile = d + '/' + scene + '/' + scene + '.labels.obj'
                posfile = d + '/' + scene + '/' + scene + '_textiles002.txt'
                framefile = d + '/' + scene + '/' + scene + '_frame002.txt'
                #print(framefile)
                #if not os.path.exists(framefile) or not os.path.exists(posfile):
                fp.write('./parametrize -i ' + objfile + ' -f 1000 -unit 0.02 -o id ' + ' -position ' + posfile + ' -num_levels 5 -rosy 4 -ks 3 -gen_field -gen_textiles -frame ' + framefile + '\n')
fp.close()
