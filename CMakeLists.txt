cmake_minimum_required(VERSION 2.8)
PROJECT (LoadParam)
SET(CMAKE_CXX_FLAGS "-std=c++0x")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR}/inc)
link_directories(${PROJECT_SOURCE_DIR}/lib/linux_x64)
ADD_EXECUTABLE(main main.cpp ${PROJECT_SOURCE_DIR}/src/caffe.pb.cc
        ${PROJECT_SOURCE_DIR}/src/upgrade_proto.cpp
        ${PROJECT_SOURCE_DIR}/src/net.cpp
        ${PROJECT_SOURCE_DIR}/src/blob_.cpp
        ${PROJECT_SOURCE_DIR}/src/layers/input_layer.cpp
        ${PROJECT_SOURCE_DIR}/src/layers/conv_layer.cpp
        ${PROJECT_SOURCE_DIR}/src/layers/pooling_layer.cpp
        ${PROJECT_SOURCE_DIR}/src/layers/relu_layer.cpp
        ${PROJECT_SOURCE_DIR}/src/layers/prelu_layer.cpp
        ${PROJECT_SOURCE_DIR}/src/layers/batch_norm_layer.cpp
        ${PROJECT_SOURCE_DIR}/src/layers/fc_layer.cpp
        ${PROJECT_SOURCE_DIR}/src/layers/softmax_layer.cpp
        )
TARGET_LINK_LIBRARIES(main protobuf glog armadillo opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc pthread)
