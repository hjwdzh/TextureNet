# Data Preparation
## Guidance
Before doing these steps, make sure to run everything in screen/tmux mode since they takes a long time. Additionally, it will consume a lot of disc space (roughly 4 TB). Also, please run it on a multicore CPU. It may take more than one day to process all data with a 40-core CPU. If you don't wish to process data by your own, please use our processed dataset from [**this link**](http://download.cs.stanford.edu/orion/texturenet/scannet-chunks.zip) (roughly 600 GB). After finish the data processing, please refer to [**src**](https://github.com/hjwdzh/TextureNet/raw/master/evaluate/) directory for training and testing.

## Download and Split Data
Download Scannet to the your preferred data folder (called DATASET here) by
```
python download.py -o DATASET
```
Edit root.py and set root=DATASET
It will provide two folders (DATASET/scans and DATASET/scans_test). However, we will additionally split scans into scans_train and scans_val, following the scannet_v2 task split. This is achieved by running
```
python gen_split.py
sh split.sh
```

## Compile binaries and scripts used for data preprocessing.
Dependencies include libigl, glm, Eigen and opencv. The precompiled libraries can be downloaded from [**this link**](http://download.cs.stanford.edu/orion/texturenet/3rd.zip). Unzip the downloaded file and move them to the 3rd directory in this project. However, opencv might need to be reinstalled if the binary is unsupported by the platform.
```
sh compile.sh
```

## Preprocess the data
Please set report_step in run_parallel.py as the number of the cpu cores on the machine.

We document the detailed logics of preprocessing. But this is achieved by simply running
```
sh preprocess.sh
```
It takes a long while, but after that you are all set. Please refer to [**src**](https://github.com/hjwdzh/TextureNet/raw/master/evaluate/) directory for training and testing.

## Explanation
The preprocessing pipeline is shown in the following figure:

![Preprocessing pipeline](https://github.com/hjwdzh/TextureNet/raw/master/img/preprocessing.png)

We will use the following data as input for preprocessing
1. ScanID/ScanID_vh_clean_2.labels.ply&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contains the triangle mesh and labels
2. ScanID/ScanID.sens&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contains the color information

### Step 1
We decode sens file and extract frames gen_sens.sh. This will output a folder called
1. ScanID/image_data.

### Step 2
We convert ply file to obj mesh file and label array (associated with mesh vertices) by gen_objs.sh. This will output two files
1. ScanID/scanID.labels.obj&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contains the triangle mesh
2. ScanID/scanID.labels.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contains the vertices' labels array

### Step 3
Most importantly, we run Quadriflow to compute the surface parametrization, which generates grid positions on geometry and local tangent directions. Specifically, we run gen_textiles.sh to achieve that. The input is
1. ScanID/ScanID.labels.obj

and the output is
1. ScanID/ScanID_textiles002.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contains the list of grid positions
2. ScanID/ScanID_frame002.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contains the list of <tv, n> where tv is one tangent vector and n is the surface normal

### Step 4
We figure out the grid positions' labels and their belonging triangle indices and barycentric coordinates by finding the nearest position on top of the triangle mesh. This is achieved by running gen_labels.sh. Specifically the input is
1. ScanID/ScanID_textiles002.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the list of grid positions
2. ScanID/ScanID.labels.obj&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the original triangle mesh
3. ScanID/ScanID.labels.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the labels of the triangle mesh

and the output is
1. ScanID/ScanID_labels002.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the list of labels of the grid positions
2. ScanID/ScanID_bary002.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the barycentric coordinates and the indices of belonging nearest triangle for grid positions

### Step 5
We extract a 10x10 patch according to the local frame for each grid position by gen_color.sh. The input is
1. ScanID/ScanID_textiles002.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the grid positions
2. ScanID/ScanID_frame002.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the local frames <tv, n>

and the output is
1. ScanID/ScanID_color002.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the highresolution patches

### Step 6
We will randomly generate chunks of each scene and store the necessary information for learning into packed pickle files using gen_chunks.sh. For details, please refer to prechunk.py and scannet_dataset.py. The output pickle file contains the following things
1. point_set&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the original 8192 points of the chunk (8192x3)
2. feature_set&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the features of the points (xyz+normal) (8192x6) 
3. cloud_pointcloud&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the highres colors (for each point, it is a 10x10x3 patch) (8192x300)
4. semantic_seg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the semantic labels (8192x1)
5. sample_weight&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the weights of labels (8192x1)
6. s1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the L1 downsample indices (1024x1)
7. s2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the L2 downsample indices (256x1)
8. s3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the L3 downsample indices (64x1)
9. s4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the L4 downsample indices (16x1)
10. g1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the neighborhood for L1, each point associated with 48 indices (8192x48)
11. g2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the neighborhood for L2, each point associated with 48 indices (1024x48)
12. g3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the neighborhood for L3, each point associated with 48 indices (256x48)
13. g4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the neighborhood for L4, each point associated with 48 indices (64x48)
14. t1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the texcoords for L1, each point associated with 48 texcoords (8192x48x2)
15. t2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the texcoords for L2, each point associated with 48 texcoords (1024x48x2)
16. t3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the texcoords for L3, each point associated with 48 texcoords (256x48x2)
17. t4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#contain the texcoords for L4, each point associated with 48 texcoords (64x48x2)
