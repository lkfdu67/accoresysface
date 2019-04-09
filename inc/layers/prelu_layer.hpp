//
// Created by liukai on 19-4-4.
//

#ifndef LOADPARAM_PRELU_LAYER_HPP
#define LOADPARAM_PRELU_LAYER_HPP

#include "layer.hpp"

using std::vector;
using std::shared_ptr;
using std::string;
using std::pair;

namespace caffe{

    class PReluLayer : public Layer
    {
    public:
        PReluLayer(){}
        ~PReluLayer(){}
        void SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Reshape(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);

    private:
//        string layer_type_;
//        string layer_name_;
//        vector<string> bottom_names_;
//        vector<string> top_names_;

        vector<int> in_shape_;
        vector<int> out_shape_;
    };

}

#endif //LOADPARAM_PRELU_LAYER_HPP
