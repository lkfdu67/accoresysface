//
// Created by jbk on 19-3-14.
//

#ifndef LOADPARAM_BATCH_NORM_LAYER_H
#define LOADPARAM_BATCH_NORM_LAYER_H

#include <layer.hpp>

using std::vector;
using std::shared_ptr;
using std::string;
using std::pair;

namespace caffe{

    class BNLayer : public Layer
    {
    public:
        BNLayer(){}
        ~BNLayer(){}
        void SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Reshape(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);

    private:
        int channels_;
        vector<int> in_shape_;
        vector<int> out_shape_;
        double eps_;
//        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };

}
#endif
