import os
from root import parent_dir

fp = open('gen_sens.sh', 'w')
for d in parent_dir:
  ss = os.listdir(d)
  for s in ss:
    if s[0] == 's':
      fp.write('./sens ' + d + '/' + s + '/' + s + '.sens ' + d + '/' + s + '/image_data\n')
fp.close()
