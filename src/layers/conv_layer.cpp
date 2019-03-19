//
// Created by jbk on 19-3-19.
//

#include <vector>
#include "layer.hpp"
#include "conv_layer.hpp"
#include "transformer_param.hpp"

using std;

namespace caffe{

    void ConvLayer::SetUp(const LayerParameter& param, const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top)
{
    cout << "ConvLayer::SetUp()" << param.layer_name << endl;
    return;
}
}
