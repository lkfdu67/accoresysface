//
// Created by jbk on 19-3-14.
//
#include <vector>
#include "batch_norm_layer.hpp"


using namespace std;

namespace caffe{

    void BNLayer::SetUp(const LayerParameter& param, const vector<Blob*>& bottom, vector<Blob*>& top)
    {
        cout << "BNLayer::SetUp()" << param.name() << endl;
        return;
    }

    void BNLayer::Forward(const vector<Blob*>& bottom, vector<Blob*>& top)
    {
        cout << "BNLayer::forward()..." << endl;
        return;
    }

    void BNLayer::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "BNLayer::calc_shape()..." << endl;
        return;
    }
}