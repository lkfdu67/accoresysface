//
// Created by jbk on 19-3-14.
//
#include <vector>
#include "layer.hpp"
#include "fc_layer.hpp"
#include "transformer_param.hpp"

using std;

namespace caffe{

    void FCLayer::SetUp(const LayerParameter& param, const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top)
    {
        cout << "FCLayer::SetUp()" << param.layer_name << endl;
        return;
    }
}
