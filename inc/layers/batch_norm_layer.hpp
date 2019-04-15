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

namespace asr{
    template<typename DType>
    class BNLayer : public Layer<DType>
    {
    public:
        BNLayer(){}
        ~BNLayer(){}
        void SetUp(const LayerParameter& param, const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);
        void Forward(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);
        void Reshape(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);

    private:
        int channels_;
        vector<int> in_shape_;
        vector<int> out_shape_;
        DType eps_;
//        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };

}
#endif
