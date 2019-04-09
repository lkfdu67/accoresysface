//
// Created by jbk on 19-3-14.
//

#ifndef LOADPARAM_POOLING_LAYER_H
#define LOADPARAM_POOLING_LAYER_H

#include <layer.hpp>

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
        void SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Reshape(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);

    private:
        vector<int> pad_;
        vector<int> stride_;
        vector<int> kernel_;
        enum PoolingParameter_PoolMethod {
            PoolMethod_MAX = 0,
            PoolMethod_AVE = 1,
        } pool_methods_;
        enum PoolingParameter_PoolRoundMode {
            PoolRoundMode_FLOOR = 0,
            PoolRoundMode_CEIL = 1,
        } pool_round_mode_;
        bool global_pooling_;
        vector<int> in_shape_;
        vector<int> out_shape_;
        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };
}

#endif
