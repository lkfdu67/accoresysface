//
// Created by liukai on 19-4-4.
//
#include <vector>
#include "layers/prelu_layer.hpp"

using namespace std;

namespace asr{
    template<typename DType>
    void PReluLayer<DType>::SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "PReluLayer::SetUp()" << param.name() << endl;
        CHECK_EQ(bottom.size(), 1)<<"Bottom size for convolution layer must be 1"<<endl;
        CHECK_EQ(top.size(), 1)<<"Top size for convolution layer must be 1"<<endl;

//        layer_name_ = param.name();
//        layer_type_ = param.type();
//        for(int bottom_id=0; bottom_id<param.bottom_size(); ++bottom_id)
//        {
//            bottom_names_.push_back(param.bottom(bottom_id));
//        }
//        for(int top_id=0; top_id<param.top_size(); ++top_id)
//        {
//            top_names_.push_back(param.top(top_id));
//        }

        in_shape_ = bottom[0]->shape();
        vector<int> weight_shape{1,in_shape_[1],1,1};
        this->weights().resize(1);
        this->weights()[0].reset(new Blob<double>(weight_shape));

        cout<<param.name()<<" top shape: ";
        PrintVector(in_shape_);
        cout<<param.name()<<" weights shape: ";
        this->PrintVector(weight_shape);
        out_shape_ = bottom[0]->shape();
        top[0]->Reshape(in_shape_);
    }
    template<typename DType>
    void PReluLayer<DType>::Reshape(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top)
    {
        in_shape_ = bottom[0]->shape();
        out_shape_ = in_shape_;
        top[0]->Reshape(out_shape_);
    }

    template<typename DType>
    void PReluLayer<DType>::Forward(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top)
    {
        cout << "PReluLayer::forward()..." << endl;
        int N = in_shape_[0];
        int C = in_shape_[1];
        int Hx = in_shape_[2];
        int Wx = in_shape_[3];
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                double negSlope=(*this->weights()[0])(0,c,0,0);
                for (int w = 0; w < Wx; ++w)
                {
                    for (int h = 0; h < Hx; ++h)
                    {
                        double tmp=(*bottom[0])(n,c,h,w);
                        (*top[0]).at(n,c,h,w) = tmp>0 ? tmp : negSlope*tmp;
                    }
                }
            }
        }


    }

}
