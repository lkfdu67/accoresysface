//
// Created by jbk on 19-3-22.
//

#ifndef LOADPARAM_SOFTMAX_LAYER_HPP
#define LOADPARAM_SOFTMAX_LAYER_HPP

#include "layer.hpp"

using std::vector;
using std::shared_ptr;
using std::string;
using std::pair;

namespace caffe{

    class SoftmaxLayer : public Layer
    {
    public:
        SoftmaxLayer(){}
        ~SoftmaxLayer(){}
        void SetUp(const LayerParameter& param, const vector<Blob*>& bottom, vector<Blob*>& top);
        void Forward(const vector<Blob*>& bottom, vector<Blob*>& top);

    private:
        vector<int> in_shape_;
        vector<int> out_shape_;
        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };

}

#endif
//LOADPARAM_SOFTMAX_LAYER_HPP
