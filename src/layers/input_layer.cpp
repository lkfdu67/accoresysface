//
// Created by jbk on 19-3-22.
//

#include <vector>
#include "layers/input_layer.hpp"


using namespace std;

namespace caffe{

    void InputLayer::SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "InputLayer::SetUp() " << param.name() << endl;

        return;
    }


    void InputLayer::Forward(const vector<Blob<double>*>& bottom, vector<Blob<double>*>& top)
    {
        cout << "InputLayer::forward()..." << endl;

        return;
    }

    void InputLayer::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "InputLayer::calc_shape()..." << endl;

        return;
    }

}

