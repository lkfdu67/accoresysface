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

namespace asr{
    template<typename DType>
    class PReluLayer : public Layer<DType>
    {
    public:
        PReluLayer(){}
        ~PReluLayer(){}
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

#endif //LOADPARAM_PRELU_LAYER_HPP
