//
// Created by hua on 19-3-14.
//

#ifndef LOADPARAM_LAYER_HPP
#define LOADPARAM_LAYER_HPP

#include "caffe.pb.h"
#include <utility>
#include "transformer_param.hpp"

using std::pair;
using std::vector;
using std::shared_ptr;
using std::string;

namespace caffe{

    class Layer{
    public:
        explicit Layer(){}
        virtual ~Layer(){}
        virtual void SetUp(const LayerParameter& param, const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top) = 0;
        virtual void Forward(const const vector<pair<string, shared_ptr<Blob>>>& bottom, vector<pair<string, shared_ptr<Blob>>>& top) = 0;

    private:
        LayerParameter layer_param_;
        vector<int> in_shape_;
        vector<int> out_shape_;
        virtual void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape) = 0;
    };
}

#endif //LOADPARAM_LAYER_HPP
