//
// Created by jbk on 19-3-19.
//

#include <vector>
#include "layer.h"
#include "pooling_layer.h"
#include "transformer_param.hpp"

using std;

namespace caffe{

    void ConvLayer::SetUp(const LayerParameter& param, const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top)
{
    cout << "ConvLayer::SetUp()" << param.layer_name << endl;
//        layer_param_ = param.layer_param;
//        weight_blob_ = param.weight_blob;
//        is_bias_ = param.is_bias;
//        in_shape_ = param.in_shape;
//        out_shape_ = calc_shape();
    return;
}
}
