import pickle
import os
import sys
import numpy as np
from ctypes import *
from time import time
from plyfile import PlyData, PlyElement

class ScannetDataset():
    def __init__(self, root, npoints=8192, split='train', chunks=100, use_color=0, use_conv=0, use_direct = 0, use_geodesic=0, use_whole = 1, output_dir='', scene=''):
        self.npoints = npoints
        self.dir = root
        self.root = root
        self.split = split
        self.dropout = 0
        self.use_color = use_color
        self.use_conv = use_conv
        self.use_whole = use_whole
        self.chunks = chunks
        self.use_geodesic = use_geodesic
        self.Neighbor = cdll.LoadLibrary("./Neighbors/libNeighbor.so")
        self.Reindex = cdll.LoadLibrary("./libReindex.so")
        self.output_dir = output_dir
        self.use_direct = use_direct
        self.scene = scene
        self.idx = 0
        self.labelweights = np.loadtxt('labelweights.txt')
        '''
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.01+labelweights)
            np.savetxt('labelweights.txt', self.labelweights)
        elif split=='test' or True:
            self.labelweights = np.ones(21)
        '''
    def ExtractChunks(self):
        self.idx += 1
        index = 0

        scene = self.scene
        d = self.root

        textile = d + '/' + scene + '/' + scene + '_textiles002.txt'
        indices = d + '/' + scene + '/' + scene + '_labels002.txt'
        frames_file = d + '/' + scene + '/' + scene + '_frame002.txt'
        models_file = d + '/' + scene + '/' + scene + '_vh_clean_2.ply'
        bary_file = d + '/' + scene + '/' + scene + '_bary002.txt'
        color_file = d + '/' + scene + '/' + scene + '_color002.txt'
        #color_file = '/oriong5/jingweih/densenet/' + scene + '.pkl'
        with open(models_file, 'rb') as f:
            plydata = PlyData.read(f)
        x = plydata.elements[0].data['x']
        y = plydata.elements[0].data['y']
        z = plydata.elements[0].data['z']
        findices = plydata['face'].data['vertex_indices']
        faces0 = np.zeros([findices.shape[0], 3]).astype('int32')
        vertices0 = np.transpose(np.array([x, y, z]).astype('float32'))

        for i in range(faces0.shape[0]):
            faces0[i] = findices[i]

        pts = np.loadtxt(textile).astype('float32')
        labels = np.zeros([pts.shape[0]]).astype('int32')
        #colors = np.zeros([pts.shape[0],3]).astype('float32')
        frame = np.loadtxt(frames_file).astype('float32')

        self.Reindex.ReadArray(c_void_p(labels.ctypes.data), c_char_p(indices.encode('utf-8')), pts.shape[0], 4)

        barycenter = np.zeros([3, pts.shape[0]]).astype('float32')
        baryind = np.zeros([pts.shape[0]]).astype('int32')

        self.Reindex.ReadBaryCentry(c_void_p(barycenter.ctypes.data), c_void_p(baryind.ctypes.data), c_char_p(bary_file.encode('utf-8')))
        barycenter = np.ascontiguousarray(np.transpose(barycenter))
        vertices0 = np.ascontiguousarray(vertices0)
        faces0 = np.ascontiguousarray(faces0)
        faces0_N = np.zeros((faces0.shape[0], 3)).astype('float32')
        faces0_V = np.zeros((faces0.shape[0], 3)).astype('float32')

        self.Reindex.ComputeFacesInfo(c_void_p(faces0_V.ctypes.data), c_void_p(faces0_N.ctypes.data), c_void_p(vertices0.ctypes.data), c_void_p(faces0.ctypes.data), c_int(vertices0.shape[0]), c_int(faces0.shape[0]));
        dict = {1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,14:13,16:14,24:15,28:16,33:17,34:18,36:19,39:20};
        for i in range(labels.shape[0]):
            j = labels[i]
            if j in dict:
                labels[i] = dict[j]
            else:
                labels[i] = 0
        labels = labels.astype('int8')

        E2E = np.zeros([faces0.shape[0] * 3]).astype('int32')

        self.Reindex.InitializeE2E(c_void_p(vertices0.ctypes.data), c_void_p(faces0.ctypes.data), c_int(vertices0.shape[0]), c_int(faces0.shape[0]), c_void_p(E2E.ctypes.data))

        name = scene
        color_set0 = np.zeros((labels.shape[0], 300), dtype='float32')
        self.Reindex.ReadArray(c_void_p(color_set0.ctypes.data), c_char_p(color_file.encode('utf-8')), color_set0.shape[0] * 300, 4)
        if self.use_whole == 0:
            for ii in range(self.chunks):
                target_name = self.output_dir + '/' + name + '_chunks_' + str(ii) + '.pkl'

                point_set = pts
                frame_set = frame
                color_set = color_set0
                vertices = vertices0
                faces = faces0
                e2e = E2E
                barycenter_set = barycenter
                baryind_set = baryind

                semantic_seg = labels.astype(np.int32)

                coordmax = np.max(point_set,axis=0)
                coordmin = np.min(point_set,axis=0)
                smpmin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin)
                smpmin[2] = coordmin[2]
                smpsz = np.minimum(coordmax-smpmin,[1.5,1.5,3.0])
                smpsz[2] = coordmax[2]-coordmin[2]
                isvalid = False
                for i in range(10):
                    curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
                    curmin = curcenter-[0.75,0.75,1.5]
                    curmax = curcenter+[0.75,0.75,1.5]
                    curmin[2] = coordmin[2]
                    curmax[2] = coordmax[2]
                    curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3
                    cur_point_set = point_set[curchoice,:]
                    if self.use_color > 0:
                        cur_color_set = color_set[curchoice,:]
                    cur_semantic_seg = semantic_seg[curchoice]
                    cur_frame_set = frame_set[curchoice,:]
                    cur_barycenter_set = barycenter_set[curchoice,:]
                    cur_baryind_set = baryind_set[curchoice]
                    if len(cur_semantic_seg)==0:
                        continue
                    mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
                    vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
                    vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
                    isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
                    if isvalid:
                        break

                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True).astype('int32')

                point_set = cur_point_set[choice,:]
                if self.use_color > 0:
                    color_set = cur_color_set[choice,:]
                frame_set = cur_frame_set[choice,:]
                semantic_seg = cur_semantic_seg[choice]
                mask = mask[choice]
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask

                if self.use_color == 1:
                    feature_set = np.zeros((color_set.shape[0],9), dtype='float32')
                    feature_set[:,0:3] = point_set
                    feature_set[:,3:6] = frame_set[:,3:6]
                    feature_set[:,6:9] = color_set
                elif self.use_color == 2:
                    feature_set = np.zeros((color_set.shape[0],6), dtype='float32')
                    feature_set[:,0:3] = point_set
                    feature_set[:,3:6] = frame_set[:,3:6]
                    cloud_pointcloud = color_set
                else:
                    feature_set = None

                if self.use_geodesic == 0:
                    return point_set, feature_set, cloud_pointcloud, semantic_seg, sample_weight, None, None, None, None, None, None, None, None, None, None, None, None

                #return point_set, semantic_seg, sample_weight, frame_set
                barycenter_set = cur_barycenter_set[choice,:]
                baryind_set = cur_baryind_set[choice]

                if self.dropout == 1:
                    dropout_ratio = np.random.random()*0.875 # 0-0.875
                    dropout_idx = np.where(np.random.random((point_set.shape[0]))<=dropout_ratio)[0]
                    point_set[dropout_idx,:] = point_set[0,:]
                    semantic_seg[dropout_idx] = semantic_seg[0]
                    frame_set[dropout_idx,:] = frame_set[0,:]
                    if self.use_color == 1:
                        color_set[dropout_idx,:] = color_set[0,:]
                    barycenter_set[dropout_idx,:] = barycenter_set[0,:]
                    baryind_set[dropout_idx] = baryind_set[0]
                    sample_weight[dropout_idx] *= 0

                faces_N = faces0_N
                faces_V = faces0_V

                s1024 = np.zeros([1024],dtype='int32')
                s256 = np.zeros([256],dtype='int32')
                s64 = np.zeros([64],dtype='int32')
                s16 = np.zeros([16],dtype='int32')

                num_neighbors = 48
                g1024 = np.zeros([8192,num_neighbors],dtype='int32')
                g256 = np.zeros([1024, num_neighbors],dtype='int32')
                g64 = np.zeros([256, num_neighbors],dtype='int32')
                g16 = np.zeros([64, num_neighbors],dtype='int32')

                c1 = np.zeros([8192,num_neighbors],dtype='int32')
                c2 = np.zeros([1024,num_neighbors],dtype='int32')
                c3 = np.zeros([256,num_neighbors],dtype='int32')
                c4 = np.zeros([64,num_neighbors],dtype='int32')

                t1 = np.zeros([8192,num_neighbors,2],dtype='float32')
                t2 = np.zeros([1024,num_neighbors,2],dtype='float32')
                t3 = np.zeros([256, num_neighbors,2],dtype='float32')
                t4 = np.zeros([64, num_neighbors,2],dtype='float32')

                self.Neighbor.InitializeMesh(c_void_p(vertices.ctypes.data), c_void_p(faces.ctypes.data), c_void_p(faces_V.ctypes.data), c_void_p(faces_N.ctypes.data), c_void_p(e2e.ctypes.data), c_int(vertices.shape[0]), c_int(faces.shape[0]))

                self.Neighbor.FurthestSampling(c_void_p(point_set.ctypes.data), c_void_p(s1024.ctypes.data), c_int(self.npoints), c_int(1024))
                point_set_1024 = point_set[s1024,:]
                frame_set_1024 = frame_set[s1024,:]
                baryind_set_1024 = baryind_set[s1024]
                self.Neighbor.FurthestSampling(c_void_p(point_set_1024.ctypes.data), c_void_p(s256.ctypes.data), c_int(1024), c_int(256))
                point_set_256 = point_set_1024[s256,:]
                frame_set_256 = frame_set_1024[s256,:]
                baryind_set_256 = baryind_set_1024[s256]
                self.Neighbor.FurthestSampling(c_void_p(point_set_256.ctypes.data), c_void_p(s64.ctypes.data), c_int(256), c_int(64))
                point_set_64 = point_set_256[s64,:]
                frame_set_64 = frame_set_256[s64,:]
                baryind_set_64 = baryind_set_256[s64]
                self.Neighbor.FurthestSampling(c_void_p(point_set_64.ctypes.data), c_void_p(s16.ctypes.data), c_int(64), c_int(16))
                point_set_16 = point_set_64[s16,:]
                frame_set_16 = frame_set_64[s16,:]
                baryind_set_16 = baryind_set_64[s16]

                if self.use_geodesic == 1:
                    self.Neighbor.GetNeighborhood(c_float(0.1), c_void_p(point_set.ctypes.data), c_void_p(frame_set.ctypes.data), c_void_p(baryind_set.ctypes.data), 8192,
                        c_void_p(point_set.ctypes.data), c_void_p(baryind_set.ctypes.data), c_int(baryind_set.shape[0]), c_void_p(g1024.ctypes.data), c_void_p(t1.ctypes.data), num_neighbors)
                    self.Neighbor.GetNeighborhood(c_float(0.2), c_void_p(point_set_1024.ctypes.data), c_void_p(frame_set_1024.ctypes.data), c_void_p(baryind_set_1024.ctypes.data), 1024,
                        c_void_p(point_set_1024.ctypes.data), c_void_p(baryind_set_1024.ctypes.data), c_int(baryind_set_1024.shape[0]), c_void_p(g256.ctypes.data), c_void_p(t2.ctypes.data), num_neighbors)
                    self.Neighbor.GetNeighborhood(c_float(0.4), c_void_p(point_set_256.ctypes.data), c_void_p(frame_set_256.ctypes.data), c_void_p(baryind_set_256.ctypes.data), 256,
                        c_void_p(point_set_256.ctypes.data), c_void_p(baryind_set_256.ctypes.data), c_int(baryind_set_256.shape[0]), c_void_p(g64.ctypes.data), c_void_p(t3.ctypes.data), num_neighbors)
                    self.Neighbor.GetNeighborhood(c_float(0.8), c_void_p(point_set_64.ctypes.data), c_void_p(frame_set_64.ctypes.data), c_void_p(baryind_set_64.ctypes.data), 64,
                        c_void_p(point_set_64.ctypes.data), c_void_p(baryind_set_64.ctypes.data), c_int(baryind_set_64.shape[0]), c_void_p(g16.ctypes.data), c_void_p(t4.ctypes.data), num_neighbors)

                    if self.use_conv == 1:
                        self.Neighbor.GetNeighborhood(c_float(0.05), c_void_p(point_set.ctypes.data), c_void_p(frame_set.ctypes.data), c_void_p(baryind_set.ctypes.data), 8192,
                            c_void_p(point_set.ctypes.data), c_void_p(baryind_set.ctypes.data), c_int(baryind_set.shape[0]), c_void_p(c1.ctypes.data), c_void_p(t1.ctypes.data), num_neighbors)
                        self.Neighbor.GetNeighborhood(c_float(0.2), c_void_p(point_set_1024.ctypes.data), c_void_p(frame_set_1024.ctypes.data), c_void_p(baryind_set_1024.ctypes.data), 1024,
                            c_void_p(point_set_1024.ctypes.data), c_void_p(baryind_set_1024.ctypes.data), c_int(baryind_set_1024.shape[0]), c_void_p(c2.ctypes.data), c_void_p(t2.ctypes.data), num_neighbors)
                        self.Neighbor.GetNeighborhood(c_float(0.4), c_void_p(point_set_256.ctypes.data), c_void_p(frame_set_256.ctypes.data), c_void_p(baryind_set_256.ctypes.data), 256,
                            c_void_p(point_set_256.ctypes.data), c_void_p(baryind_set_256.ctypes.data), c_int(baryind_set_256.shape[0]), c_void_p(c3.ctypes.data), c_void_p(t3.ctypes.data), num_neighbors)
                        self.Neighbor.GetNeighborhood(c_float(0.8), c_void_p(point_set_64.ctypes.data), c_void_p(frame_set_64.ctypes.data), c_void_p(baryind_set_64.ctypes.data), 64,
                            c_void_p(point_set_64.ctypes.data), c_void_p(baryind_set_64.ctypes.data), c_int(baryind_set_64.shape[0]), c_void_p(c4.ctypes.data), c_void_p(t4.ctypes.data), num_neighbors)

                sample_weight = sample_weight.astype('float32')
                fp = open(target_name, 'wb')
                pickle.dump(point_set, fp)
                pickle.dump(feature_set, fp)
                pickle.dump(cloud_pointcloud, fp)
                pickle.dump(semantic_seg, fp)
                pickle.dump(sample_weight, fp)
                pickle.dump(s1024, fp)
                pickle.dump(s256, fp)
                pickle.dump(s64, fp)
                pickle.dump(s16, fp)
                pickle.dump(g1024, fp)
                pickle.dump(g256, fp)
                pickle.dump(g64, fp)
                pickle.dump(g16, fp)
                pickle.dump(t1, fp)
                pickle.dump(t2, fp)
                pickle.dump(t3, fp)
                pickle.dump(t4, fp)
                fp.close()
        else:
            name = scene
            point_set_ini = pts
            semantic_seg_ini = labels.astype(np.int32)
            frame_set_ini = frame

            color_set_ini = color_set0

            vertices = vertices0
            faces = faces0
            e2e = E2E
            barycenter_set_ini = barycenter
            baryind_set_ini = baryind

            coordmax = np.max(point_set_ini,axis=0)
            coordmin = np.min(point_set_ini,axis=0)
            nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
            nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)

            isvalid = False

            faces_N = faces0_N
            faces_V = faces0_V
            self.Neighbor.InitializeMesh(c_void_p(vertices.ctypes.data), c_void_p(faces.ctypes.data), c_void_p(faces_V.ctypes.data), c_void_p(faces_N.ctypes.data), c_void_p(e2e.ctypes.data), c_int(vertices.shape[0]), c_int(faces.shape[0]))

            iters = 0
            for i in range(nsubvolume_x):
                for j in range(nsubvolume_y):
                    curmin = coordmin+[i*1.5,j*1.5,0]
                    curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]

                    curmin[2] = coordmin[2]
                    curmax[2] = coordmax[2]

                    curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3

                    cur_point_set = point_set_ini[curchoice,:]
                    cur_frame_set = frame_set_ini[curchoice,:]
                    if self.use_color > 0:
                        cur_color_set = color_set_ini[curchoice,:]

                    cur_barycenter_set = barycenter_set_ini[curchoice,:]
                    cur_baryind_set = baryind_set_ini[curchoice]

                    cur_semantic_seg = semantic_seg_ini[curchoice]
                    
                    if len(cur_semantic_seg) == 0:
                        continue
                    mask0 = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3

                    vidx = np.ceil((cur_point_set[mask0,:]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
                    vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])

                    choices = np.array([i for i in range(len(cur_semantic_seg))], dtype='int32')
                    np.random.shuffle(choices)

                    for start_idx in range(0, len(cur_semantic_seg), self.npoints):
                        choice = np.zeros((self.npoints), dtype='int32')

                        #choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                        idx = start_idx
                        for t in range(self.npoints):
                            choice[t] = choices[idx]
                            idx += 1
                            if idx >= len(cur_semantic_seg):
                                idx = 0

                        point_set = cur_point_set[choice,:] # Nx3
                        frame_set = cur_frame_set[choice,:]
                        if self.use_color > 0:
                            color_set = cur_color_set[choice,:]
                        semantic_seg = cur_semantic_seg[choice] # N
                        barycenter_set = cur_barycenter_set[choice,:]
                        baryind_set = cur_baryind_set[choice]

                        mask = mask0[choice]
                        if sum(mask)/float(len(mask))<0.01:
                            continue
                        sample_weight = self.labelweights[semantic_seg]
                        sample_weight *= mask # N

                        sample_weight = sample_weight.astype('float32')

                        feature_set = np.zeros((point_set.shape[0], 6), dtype='float32')
                        feature_set[:,0:3] = point_set
                        feature_set[:,3:6] = frame_set[:,3:6]
                        color_pointcloud = color_set

                        s1024 = np.zeros([1024],dtype='int32')
                        s256 = np.zeros([256],dtype='int32')
                        s64 = np.zeros([64],dtype='int32')
                        s16 = np.zeros([16],dtype='int32')

                        num_neighbors = 48
                        g1024 = np.zeros([8192,num_neighbors],dtype='int32')
                        g256 = np.zeros([1024, num_neighbors],dtype='int32')
                        g64 = np.zeros([256, num_neighbors],dtype='int32')
                        g16 = np.zeros([64, num_neighbors],dtype='int32')

                        c1 = np.zeros([8192,num_neighbors],dtype='int32')
                        c2 = np.zeros([1024,num_neighbors],dtype='int32')
                        c3 = np.zeros([256,num_neighbors],dtype='int32')
                        c4 = np.zeros([64,num_neighbors],dtype='int32')

                        t1 = np.zeros([8192,num_neighbors,2],dtype='float32')
                        t2 = np.zeros([1024,num_neighbors,2],dtype='float32')
                        t3 = np.zeros([256, num_neighbors,2],dtype='float32')
                        t4 = np.zeros([64, num_neighbors,2],dtype='float32')

                        self.Neighbor.FurthestSampling(c_void_p(point_set.ctypes.data), c_void_p(s1024.ctypes.data), c_int(self.npoints), c_int(1024))
                        point_set_1024 = point_set[s1024,:]
                        frame_set_1024 = frame_set[s1024,:]
                        baryind_set_1024 = baryind_set[s1024]
                        self.Neighbor.FurthestSampling(c_void_p(point_set_1024.ctypes.data), c_void_p(s256.ctypes.data), c_int(1024), c_int(256))
                        point_set_256 = point_set_1024[s256,:]
                        frame_set_256 = frame_set_1024[s256,:]
                        baryind_set_256 = baryind_set_1024[s256]
                        self.Neighbor.FurthestSampling(c_void_p(point_set_256.ctypes.data), c_void_p(s64.ctypes.data), c_int(256), c_int(64))
                        point_set_64 = point_set_256[s64,:]
                        frame_set_64 = frame_set_256[s64,:]
                        baryind_set_64 = baryind_set_256[s64]
                        self.Neighbor.FurthestSampling(c_void_p(point_set_64.ctypes.data), c_void_p(s16.ctypes.data), c_int(64), c_int(16))
                        point_set_16 = point_set_64[s16,:]
                        frame_set_16 = frame_set_64[s16,:]
                        baryind_set_16 = baryind_set_64[s16]

                        if self.use_geodesic:
                            self.Neighbor.GetNeighborhood(c_float(0.1), c_void_p(point_set.ctypes.data), c_void_p(frame_set.ctypes.data), c_void_p(baryind_set.ctypes.data), 8192,
                                c_void_p(point_set.ctypes.data), c_void_p(baryind_set.ctypes.data), c_int(baryind_set.shape[0]), c_void_p(g1024.ctypes.data), c_void_p(t1.ctypes.data), num_neighbors)
                            self.Neighbor.GetNeighborhood(c_float(0.2), c_void_p(point_set_1024.ctypes.data), c_void_p(frame_set_1024.ctypes.data), c_void_p(baryind_set_1024.ctypes.data), 1024,
                                c_void_p(point_set_1024.ctypes.data), c_void_p(baryind_set_1024.ctypes.data), c_int(baryind_set_1024.shape[0]), c_void_p(g256.ctypes.data), c_void_p(t2.ctypes.data), num_neighbors)
                            self.Neighbor.GetNeighborhood(c_float(0.4), c_void_p(point_set_256.ctypes.data), c_void_p(frame_set_256.ctypes.data), c_void_p(baryind_set_256.ctypes.data), 256,
                                c_void_p(point_set_256.ctypes.data), c_void_p(baryind_set_256.ctypes.data), c_int(baryind_set_256.shape[0]), c_void_p(g64.ctypes.data), c_void_p(t3.ctypes.data), num_neighbors)
                            self.Neighbor.GetNeighborhood(c_float(0.8), c_void_p(point_set_64.ctypes.data), c_void_p(frame_set_64.ctypes.data), c_void_p(baryind_set_64.ctypes.data), 64,
                                c_void_p(point_set_64.ctypes.data), c_void_p(baryind_set_64.ctypes.data), c_int(baryind_set_64.shape[0]), c_void_p(g16.ctypes.data), c_void_p(t4.ctypes.data), num_neighbors)
                            

                            if self.use_conv == 1:
                                self.Neighbor.GetNeighborhood(c_float(0.05), c_void_p(point_set.ctypes.data), c_void_p(frame_set.ctypes.data), c_void_p(baryind_set.ctypes.data), 8192,
                                    c_void_p(point_set.ctypes.data), c_void_p(baryind_set.ctypes.data), c_int(baryind_set.shape[0]), c_void_p(c1.ctypes.data), c_void_p(t1.ctypes.data), num_neighbors)
                                self.Neighbor.GetNeighborhood(c_float(0.25), c_void_p(point_set_1024.ctypes.data), c_void_p(frame_set_1024.ctypes.data), c_void_p(baryind_set_1024.ctypes.data), 1024,
                                    c_void_p(point_set_1024.ctypes.data), c_void_p(baryind_set_1024.ctypes.data), c_int(baryind_set_1024.shape[0]), c_void_p(c2.ctypes.data), c_void_p(t2.ctypes.data), num_neighbors)
                                self.Neighbor.GetNeighborhood(c_float(0.5), c_void_p(point_set_256.ctypes.data), c_void_p(frame_set_256.ctypes.data), c_void_p(baryind_set_256.ctypes.data), 256,
                                    c_void_p(point_set_256.ctypes.data), c_void_p(baryind_set_256.ctypes.data), c_int(baryind_set_256.shape[0]), c_void_p(c3.ctypes.data), c_void_p(t3.ctypes.data), num_neighbors)
                                self.Neighbor.GetNeighborhood(c_float(1.0), c_void_p(point_set_64.ctypes.data), c_void_p(frame_set_64.ctypes.data), c_void_p(baryind_set_64.ctypes.data), 64,
                                    c_void_p(point_set_64.ctypes.data), c_void_p(baryind_set_64.ctypes.data), c_int(baryind_set_64.shape[0]), c_void_p(c4.ctypes.data), c_void_p(t4.ctypes.data), num_neighbors)

                        target_name = self.output_dir + '/' + name + '_chunk_' + str(iters) + '.pkl'
                        print('prechunk ', target_name, index, iters)
                        fp = open(target_name, 'wb')
                        pickle.dump(point_set, fp)
                        pickle.dump(feature_set, fp)
                        pickle.dump(color_set, fp)
                        pickle.dump(semantic_seg, fp)
                        pickle.dump(sample_weight, fp)
                        pickle.dump(s1024, fp)
                        pickle.dump(s256, fp)
                        pickle.dump(s64, fp)
                        pickle.dump(s16, fp)
                        pickle.dump(g1024, fp)
                        pickle.dump(g256, fp)
                        pickle.dump(g64, fp)
                        pickle.dump(g16, fp)
                        pickle.dump(t1, fp)
                        pickle.dump(t2, fp)
                        pickle.dump(t3, fp)
                        pickle.dump(t4, fp)

                        fp.close()

                        iters += 1

        return None

    def __len__(self):
        return len(self.scenes)
