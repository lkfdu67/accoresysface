//
// Created by jbk on 19-3-14.
//

#ifndef LOADPARAM_FLATTEN_LAYER_H
#define LOADPARAM_FLATTEN_LAYER_H

#include "layer.hpp"

using std::vector;
using std::shared_ptr;
using std::string;
using std::pair;

namespace caffe{

    class FCLayer : public Layer
    {
    public:
        FCLayer(){}
        ~FCLayer(){}
        void SetUp(const LayerParameter& param, const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top);
        void Forward(const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top);

    private:
        LayerParameter layer_param_;
        vector<int> in_shape_;
        vector<int> out_shape_;
        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };

}
#endif
