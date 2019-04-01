//
// Created by jbk on 19-3-19.
//

#ifndef LOADPARAM_RELU_LAYER_HPP
#define LOADPARAM_RELU_LAYER_HPP

#include "layer.hpp"

using std::vector;
using std::shared_ptr;
using std::string;
using std::pair;

namespace caffe{

    class ReluLayer : public Layer
    {
    public:
        ReluLayer(){}
        ~ReluLayer(){}
        void SetUp(const LayerParameter& param, const vector<Blob<double>*>& bottom, vector<Blob<double>*>& top);
        void Forward(const vector<Blob<double>*>& bottom, vector<Blob<double>*>& top);

    private:
        LayerParameter layer_param_;
        vector<int> in_shape_;
        vector<int> out_shape_;
        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };

}


#endif //LOADPARAM_RELU_LAYER_HPP
