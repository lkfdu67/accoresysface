//
// Created by jbk on 19-3-14.
//
#include <vector>
#include "layer.hpp"
#include "batch_norm_layer.hpp"
#include "transformer_param.hpp"

using std;

namespace caffe{

    void BNLayer::SetUp(const LayerParameter& param, const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top)
    {
        cout << "BNLayer::SetUp()" << param.layer_name << endl;
        return;
    }
}