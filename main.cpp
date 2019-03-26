#include <iostream>
#include "net.hpp"

int main() {
    const string& model_file = "../res/deploy.prototxt";
    const string& trained_file = "";
    caffe::Net net;
    caffe::NetParameter param;
    caffe::ReadNetParamsFromTextFile(model_file, &param);
    net.Init(param);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}