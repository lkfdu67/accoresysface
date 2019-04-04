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
        void SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);

    private:
        vector<int> in_shape_;
//        vector<int> out_shape_;
    };

}

#endif
//LOADPARAM_SOFTMAX_LAYER_HPP
