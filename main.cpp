#include <iostream>
#include <net.hpp>

int main() {
    const string& model_file = "../res/det1.prototxt";
    const string& trained_file = "";
    caffe::Net net;
    caffe::NetParameter param;
    caffe::ReadNetParamsFromTextFile(model_file, &param);
    net.Init(param);
    //net.CopyTrainedParams("../res/det1.caffemodel");
    std::cout << "Hello, World!" << std::endl;
    return 0;
}