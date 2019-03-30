export IGL_INCLUDE_DIR=../3rd
export EIGEN_INCLUDE_DIR=../3rd/eigen3

g++ -std=c++11 extract_labels.cpp -I$IGL_INCLUDE_DIR -I$EIGEN_INCLUDE_DIR -pthread -O3 -o extract_labels 