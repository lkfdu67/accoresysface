//
// Created by jbk on 19-3-22.
//


#include <vector>
#include "layers/softmax_layer.hpp"


using namespace std;

namespace caffe{

    void SoftmaxLayer::SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "SoftmaxLayer::SetUp() " << param.name() << endl;
        CHECK_EQ(bottom.size(), 1)<<"Bottom size for convolution layer must be 1"<<endl;
        CHECK_EQ(top.size(), 1)<<"Top size for convolution layer must be 1"<<endl;

        in_shape_ = bottom[0]->shape();
//        out_shape_ = bottom[0]->shape();
        top[0]->Reshape(in_shape_);
    }


    void SoftmaxLayer::Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "SoftmaxLayer::forward()..." << endl;

        return;
    }

}

