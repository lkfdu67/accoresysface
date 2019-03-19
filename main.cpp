#include <iostream>
#include "net.hpp"

int main() {
    const string& model_file = "../res/det1.prototxt";
    const string& trained_file = "";
    caffe::Net net(model_file, trained_file);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}