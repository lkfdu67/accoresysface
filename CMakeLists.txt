cmake_minimum_required(VERSION 2.8)
PROJECT (LoadParam)
SET(CMAKE_CXX_FLAGS "-std=c++0x")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# aux_source_directory(./ DIR_SRC)
include_directories(${PROJECT_SOURCE_DIR}/inc)
link_directories(${PROJECT_SOURCE_DIR}/lib/linux_x64)
ADD_EXECUTABLE(main main.cpp ${PROJECT_SOURCE_DIR}/src/caffe.pb.cc
        ${PROJECT_SOURCE_DIR}/src/upgrade_proto.cpp)
#ADD_LIBRARY(main SHARED ${PROJECT_SOURCE_DIR}/lib/linux_x64/libprotobuf.so)
TARGET_LINK_LIBRARIES(main protobuf gflags glog pthread)
#ADD_LIBRARY(main STATIC ./lib/linux_x64/libprotobuf.a)
