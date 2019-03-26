//
// Created by jbk on 19-3-14.
//

#ifndef LOADPARAM_POOLING_LAYER_H
#define LOADPARAM_POOLING_LAYER_H

#include "layer.hpp"

using std::vector;
using std::shared_ptr;
using std::string;
using std::pair;

namespace caffe{

    class PoolLayer : public Layer
    {
    public:
        PoolLayer(){}
        ~PoolLayer(){}
        void SetUp(const LayerParameter& param, const vector<Blob*>& bottom, vector<Blob*>& top);
        void Forward(const vector<Blob*>& bottom, vector<Blob*>& top);

    private:

        int pad_w;
        int pad_h;
        int stride_w;
        int stride_h;
        int kernel_w;
        int kernel_h;
        string pool_types;
        bool global_pooling;
        vector<int> in_shape_;
        vector<int> out_shape_;
        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };

}

#endif
