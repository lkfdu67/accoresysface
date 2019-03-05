cmake_minimum_required(VERSION 3.13)
project(ForwardNet)

set(CMAKE_CXX_STANDARD 14)

add_executable(ForwardNet main.cpp include/net.h include/net_caffe.h)