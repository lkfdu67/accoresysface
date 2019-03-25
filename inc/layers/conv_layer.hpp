//
// Created by jbk on 19-3-19.
//

#ifndef LOADPARAM_CONV_LAYER_HPP
#define LOADPARAM_CONV_LAYER_HPP

#include "layer.hpp"

using std::vector;
using std::shared_ptr;
using std::string;
using std::pair;

namespace caffe{

    class ConvLayer : public Layer
    {
    public:
        ConvLayer(){}
        ~ConvLayer(){}
        void SetUp(const LayerParameter& param, const vector<Blob*>& bottom, vector<Blob*>& top);
        void Forward(const vector<Blob*>& bottom, vector<Blob*>& top);

    private:
        LayerParameter layer_param_;
        vector<int> in_shape_;
        vector<int> out_shape_;
        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };

}


#endif //LOADPARAM_CONV_LAYER_HPP
