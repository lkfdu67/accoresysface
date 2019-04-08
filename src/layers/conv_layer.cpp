//
// Created by jbk on 19-3-19.
//

#include <vector>
#include <memory>
//#include <transformer_param.hpp>
#include "glog/logging.h"
#include "layers/conv_layer.hpp"
#include "blob_.hpp"

using namespace std;

namespace caffe{

    void ConvLayer::SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "ConvLayer::SetUp()" << param.name() << endl;
        CHECK_EQ(bottom.size(), 1)<<"Bottom size for convolution layer must be 1"<<endl;
        CHECK_EQ(top.size(), 1)<<"Top size for convolution layer must be 1"<<endl;

        layer_name_ = param.name();
        layer_type_ = param.type();
        for(int bottom_id=0; bottom_id<param.bottom_size(); ++bottom_id)
        {
            bottom_names_.push_back(param.bottom(bottom_id));
        }
        for(int top_id=0; top_id<param.top_size(); ++top_id)
        {
            top_names_.push_back(param.top(top_id));
        }
        const ConvolutionParameter& conv_param = param.convolution_param();

        int padSize = conv_param.pad_size();
        int strideSize = conv_param.stride_size();
        int kernelSize = conv_param.kernel_size_size();
        if(!conv_param.pad_size()){
            pad_.push_back(0);
        } else{
            for (int idx=0;idx<padSize;++idx)
            {
                pad_.push_back(conv_param.pad(idx));
            }
        }
        if(!conv_param.stride_size())
        {
            stride_.push_back(1);
        }else{
            for (int j = 0; j < strideSize; ++j) {
                stride_.push_back(conv_param.stride(j));
            }
        }
        for (int k = 0; k < kernelSize; ++k) {
            kernel_.push_back(conv_param.kernel_size(k));
        }
        in_shape_ = bottom[0]->shape();
        num_output_ = conv_param.num_output();
        num_channel_ = in_shape_[1];
        bias_term_ = conv_param.bias_term();

        if (bias_term_) {
            weights().resize(2);
        } else {
            weights().resize(1);
        }
        vector<int> weight_shape{num_output_,num_channel_,kernel_[0],kernel_[0]};
        vector<int> bias_shape{num_output_,1,1,1};

        weights()[0].reset(new Blob<double>(weight_shape));

        // If necessary, initialize and fill the biases.
        if (bias_term_) {
            weights()[1].reset(new Blob<double>(bias_shape));
        }

        calc_shape_(in_shape_,out_shape_);

        //print shape information
        cout<<layer_name_<<" top shape: ";
        PrintVector(out_shape_);
        cout<<layer_name_<<" weights shape: ";
        PrintVector(weight_shape);
        cout<<layer_name_<<" bias shape: ";
        PrintVector(bias_shape);

        top[0]->Reshape(out_shape_);
    }


    void ConvLayer::Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "ConvLayer::forward()..." << endl;
        CHECK_EQ(bottom.size(), 1)<<"Bottom size for convolution layer must be 1"<<endl;
        CHECK_EQ(top.size(), 1)<<"Top size for convolution layer must be 1"<<endl;
        int N = in_shape_[0];
        int C = in_shape_[1];
        int Hx = in_shape_[2];
        int Wx = in_shape_[3];

        int F = out_shape_[1];
        int Ho = out_shape_[2];
        int Wo = out_shape_[3];

        int pad = pad_[0];
        int kernel = kernel_[0];
        int stride = stride_[0];

        // padding
        Blob<double> padX(N, C, Hx + 2 * pad, Wx + 2 * pad);
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int w = 0; w < Wx; ++w)
                {
                    for (int h = 0; h < Hx; ++h)
                    {
                        padX.at(n,c,h + pad, w + pad) = (*bottom[0])(n,c,h,w);
                    }
                }
            }
        }

        for (int n = 0; n < N; ++n)   //
        {
            for (int f = 0; f < F; ++f)  //
            {
                Blob<double> weightWin=weights()[0]->sub_blob("f:f;:;:;:");
                double bias=(*weights()[1])(f,0,0,0);
                for (int hh = 0; hh < Ho; hh+=stride)   //
                {
                    for (int ww = 0; ww < Wo; ww+=stride)   //
                    {
                        Blob<double> window = padX.sub_blob("n:n;:;hh:hh+kernel-1;ww:ww+kernel-1");
                        window *= weightWin;
                        vector<double> tmpsum=window.sum_all_channel();
                        double sum_scaler{tmpsum[0]};
                        if(bias_term_)
                        {
                            sum_scaler = tmpsum[0]+bias;
                        }

                        (*top[0]).at(n,f,hh,ww) = sum_scaler;
                    }
                }
            }
        }


    }

    void ConvLayer::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
//        cout << "ConvLayer::calc_shape()..." << endl;
        // (N,C,H,W)
        // input and output have the same N
        out_shape.push_back(in_shape[0]);
        // Cout is the number of filters
        out_shape.push_back(num_output_);
        int H_in{in_shape[2]};
        int W_in{in_shape[3]};
        // CHECK re
        int pad = pad_[0];
        int kernel = kernel_[0];
        int stride = stride_[0];

        int H_out, W_out;
        H_out = (H_in+2*pad-kernel)/stride+1;
        W_out = (W_in+2*pad-kernel)/stride+1;
        out_shape.push_back(H_out);
        out_shape.push_back(W_out);
    }
}
