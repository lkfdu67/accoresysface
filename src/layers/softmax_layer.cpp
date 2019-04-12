//
// Created by jbk on 19-3-22.
//


#include <vector>
#include "layers/softmax_layer.hpp"


using namespace std;

namespace asr{

    void SoftmaxLayer::SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "SoftmaxLayer::SetUp() " << param.name() << endl;
        CHECK_EQ(bottom.size(), 1)<<"Bottom size for convolution layer must be 1"<<endl;
        CHECK_EQ(top.size(), 1)<<"Top size for convolution layer must be 1"<<endl;

        in_shape_ = bottom[0]->shape();
        cout<<param.name()<<" top shape: ";
        PrintVector(in_shape_);
        out_shape_ = in_shape_;
        top[0]->Reshape(in_shape_);
    }

    void SoftmaxLayer::Reshape(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        in_shape_ = bottom[0]->shape();
        out_shape_ = in_shape_;
        top[0]->Reshape(in_shape_);
    }

    void SoftmaxLayer::Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "SoftmaxLayer::forward()..." << endl;

        int N = in_shape_[0];
        int C = in_shape_[1];
        int H = in_shape_[2];
        int W = in_shape_[3];
        Blob<double> maxBlob=bottom[0]->max_along_dim(1);
        Blob<double> minusMaxBlob=(*bottom[0])-maxBlob;
        Blob<double> expBlob=minusMaxBlob.exp();
        Blob<double> sumBlob=expBlob.sum_along_channel();
        for (int n=0; n<N; ++n)
        {
            for (int h=0; h<H; ++h)
            {
                for (int w=0; w<W; ++w)
                {
                    for (int c=0; c<C; ++c)
                    {
                        (*top[0]).at(n,c,h,w) = expBlob.at(n,c,h,w)/sumBlob.at(n,0,h,w);
                    }
                }
            }
        }
    }

}

