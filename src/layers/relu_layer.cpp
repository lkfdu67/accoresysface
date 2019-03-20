//
// Created by jbk on 19-3-19.
//

#include <vector>
#include "relu_layer.hpp"

using namespace std;

namespace caffe{

    void ReluLayer::SetUp(const LayerParameter& param, const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top)
    {
        cout << "ReluLayer::SetUp()" << param.name() << endl;
        return;
    }


    void ReluLayer::Forward(const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top)
    {
        cout << "ReluLayer::forward()..." << endl;
        return;
    }

    void ReluLayer::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "ReluLayer::calc_shape()..." << endl;
        return;
    }
}
