# Set path
export IGL_INCLUDE_DIR=../3rd
export EIGEN_INCLUDE_DIR=../3rd/eigen3
export OPENCV_INCLUDE_DIR=../3rd/opencv/include
export OPENCV_LIBRARY_DIR=../3rd/opencv/lib

# Compile

echo 'Build SensReader...'
cd SensReader
make
cp ./sens ..
cd ..

echo 'Build Textiles...'
cd QuadriFlow
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cp ./parametrize ../../
cd ../..

echo 'Build Labels...'
g++ -std=c++11 maplabel.cpp -I$IGL_INCLUDE_DIR -I$EIGEN_INCLUDE_DIR -pthread -O3 -o maplabel 
echo 'Build Colors...'
g++ -std=c++11 ptex.cpp -I$EIGEN_INCLUDE_DIR -I$OPENCV_INCLUDE_DIR -L$OPENCV_LIBRARY_DIR -lopencv_core -lopencv_highgui -lopencv_imgproc -O3 -o ptex
echo 'Build IO...'
g++ -std=c++11 -fopenmp fastio.cpp -shared -fPIC -I$EIGEN_INCLUDE_DIR -O3 -o libReindex.so

echo 'Build Neighbors...'
cd Neighbors
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cp ./libNeighbor.so ..
cd ../..

# Gen script
echo 'Generate Scripts...'
python gen_sens.py
python gen_textiles.py
python gen_objs.py
python gen_labels.py
python gen_colors.py
python gen_chunks.py
