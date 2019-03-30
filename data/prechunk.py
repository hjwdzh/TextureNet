import argparse
import math
from datetime import datetime
#import h5pyprovider
import numpy as np
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import scannet_dataset


root = sys.argv[1]
split = sys.argv[2]
chunks = int(sys.argv[3])
use_whole = int(sys.argv[4])
output_dir = sys.argv[5]
scene = sys.argv[6]
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

dataset = scannet_dataset.ScannetDataset(root=root, npoints=8192, split=split, chunks=chunks, use_color=2, use_conv=0, use_geodesic=1, use_whole=use_whole, output_dir=output_dir, scene=scene)
dataset.ExtractChunks()