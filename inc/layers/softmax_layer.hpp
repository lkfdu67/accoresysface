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

namespace asr{
    template<typename DType>
    class SoftmaxLayer : public Layer<DType>
    {
    public:
        SoftmaxLayer(){}
        ~SoftmaxLayer(){}
        void SetUp(const LayerParameter& param, const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);
        void Forward(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);
        void Reshape(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);

    private:
        vector<int> in_shape_;
        vector<int> out_shape_;
    };

}

#endif
//LOADPARAM_SOFTMAX_LAYER_HPP
