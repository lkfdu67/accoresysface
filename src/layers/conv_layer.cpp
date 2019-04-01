//
// Created by jbk on 19-3-19.
//

#include <vector>
#include "layers/conv_layer.hpp"


using namespace std;

namespace caffe{

    void ConvLayer::SetUp(const LayerParameter& param, const vector<Blob<double>*>& bottom, vector<Blob<double>*>& top)
{
    cout << "ConvLayer::SetUp()" << param.name() << endl;
    return;
}


    void ConvLayer::Forward(const vector<Blob<double>*>& bottom, vector<Blob<double>*>& top)
    {
        cout << "ConvLayer::forward()..." << endl;
        return;
    }

    void ConvLayer::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "ConvLayer::calc_shape()..." << endl;
        return;
    }
}
