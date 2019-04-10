//
// Created by jbk on 19-3-19.
//

#ifndef LOADPARAM_CONV_LAYER_HPP
#define LOADPARAM_CONV_LAYER_HPP

#include "layer.hpp"
#include "blob_.hpp"
#include <memory>

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
        void SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Reshape(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);

    private:
        string layer_type_;
        string layer_name_;
        vector<string> bottom_names_;
        vector<string> top_names_;

        vector<int> pad_;
        vector<int> stride_;
        vector<int> kernel_;

        int num_output_;
        int num_channel_;
        bool bias_term_;

        vector<int> in_shape_;
        vector<int> out_shape_;
        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };

}


#endif //LOADPARAM_CONV_LAYER_HPP
