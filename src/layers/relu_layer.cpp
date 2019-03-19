//
// Created by jbk on 19-3-19.
//

#include <vector>
#include "layer.hpp"
#include "relu_layer.hpp"
#include "transformer_param.hpp"

using std;

namespace caffe{

    void ReluLayer::SetUp(const LayerParameter& param, const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top)
{
    cout << "ReluLayer::SetUp()" << param.layer_name << endl;
    return;
}
}
