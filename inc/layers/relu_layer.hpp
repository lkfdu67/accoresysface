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

namespace asr{
    template<typename DType>
    class ReluLayer : public Layer<DType>
    {
    public:
        ReluLayer(){}
        ~ReluLayer(){}
        void SetUp(const LayerParameter& param, const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);
        void Forward(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);
        void Reshape(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);

    private:
//        string layer_type_;
//        string layer_name_;
//        vector<string> bottom_names_;
//        vector<string> top_names_;

        vector<int> in_shape_;
        vector<int> out_shape_;
    };

}


#endif //LOADPARAM_RELU_LAYER_HPP
