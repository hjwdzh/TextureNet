import os
from root import parent_dir

dirs = parent_dir
fp = open('gen_colors.sh', 'w')
for d in dirs:
        scenes = os.listdir(d)
        for scene in scenes:
                if scene[0] != 's':
                        continue
                folder = d + '/' + scene
                textiles = scene + '_textiles002.txt'
                frames = scene + '_frame002.txt'
                entries = scene + '_entry002.txt'
                ptex = scene + '_color002.txt'
                fp.write('./ptex ' + folder + ' ' + textiles + ' ' + frames + ' ' + ptex  + '\n')

fp.close()