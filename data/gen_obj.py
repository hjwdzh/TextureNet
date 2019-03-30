import sys
from plyfile import PlyData, PlyElement
import numpy as np
print(sys.argv[1])
plydata = PlyData.read(sys.argv[1])
flag = int(sys.argv[4])
fp = open(sys.argv[2], 'w')
x_array = plydata['vertex']['x']
y_array = plydata['vertex']['y']
z_array = plydata['vertex']['z']
if flag == 0:
	l_array = plydata['vertex']['label']
else:
	l_array = np.array([0 for i in range(x_array.shape[0])])

for i in range(x_array.shape[0]):
        fp.write('v ' + str(x_array[i]) + ' ' + str(y_array[i]) + ' ' + str(z_array[i]) + '\n')

faces = plydata['face']['vertex_indices']
for i in range(faces.shape[0]):
        fp.write('f ' + str(faces[i][0]+1) + ' ' + str(faces[i][1]+1) + ' ' + str(faces[i][2]+1) + '\n')
fp.close()
fp = open(sys.argv[3], 'w')
for i in range(l_array.shape[0]):
        fp.write(str(l_array[i]) + '\n')
fp.close()