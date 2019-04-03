//
// Created by jbk on 19-3-22.
//

#ifndef LOADPARAM_INPUT_LAYER_HPP
#define LOADPARAM_INPUT_LAYER_HPP

#include <layer.hpp>

using std::vector;
using std::shared_ptr;
using std::string;
using std::pair;

namespace caffe{

    class InputLayer : public Layer
    {
    public:
        InputLayer(){}
        ~InputLayer(){}
        void SetUp(const LayerParameter& param,  const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);
        void Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top);

    private:

        vector<int> in_shape_;
        vector<int> out_shape_;
        void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
    };

}

#endif //LOADPARAM_INPUT_LAYER_HPP
