/usr/local/cuda/bin/nvcc -std=c++11 tf_interpolate_g.cu -o tf_interpolate_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_interpolate.cpp tf_interpolate_g.cu.o -o tf_interpolate_so.so -shared -fPIC -I /home/jingweih/venv/lib/python3.5/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -lcudart -L /usr/local/cuda-9.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L/home/jingweih/venv/lib/python3.5/site-packages/tensorflow -ltensorflow_framework

# TF1.4
g++ -std=c++11 tf_interpolate.cpp tf_interpolate_g.cu.o -o tf_interpolate_so.so -shared -fPIC -I /orions4-zfs/projects/jingweih/envTensorflow/lib/python3.5/site-packages/tensorflow/include -I /usr/local/cuda/include -I /orions4-zfs/projects/jingweih/envTensorflow/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L/orions4-zfs/projects/jingweih/envTensorflow/lib/python3.5/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
