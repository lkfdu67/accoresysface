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

    void ConvLayer::SetUp(const LayerParameter& param, const vector<shared_ptr<Blob<double>>>& bottom, vector<shared_ptr<Blob<double>>>& top)
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
        if (param.blobs_size() > 0) {
            blobs().resize(param.blobs_size());
            for (int i = 0; i < param.blobs_size(); ++i) {
                //blobs()[i].reset(new Blob);
                blobs()[i]->FromProto(param.blobs(i));// re
            }
        }
        ConvolutionParameter conv_param = param.convolution_param();
        pad_w_ = conv_param.pad_w();
        pad_h_ = conv_param.pad_h();
        stride_w_ = conv_param.stride_w();
        stride_h_ = conv_param.stride_h();
        kernel_w_ = conv_param.kernel_w();
        kernel_h_ = conv_param.kernel_h();

        num_output_ = conv_param.num_output();
        num_channel_ = conv_param.axis();
        bias_term_ = conv_param.bias_term();
        in_shape_ = bottom[0]->shape(); //re
        calc_shape_(in_shape_,out_shape_);

//        Blob<double>* toptmp(out_shape_); //re
//        top.push_back(toptmp);
    }


    void ConvLayer::Forward(const vector<shared_ptr<Blob<double>>>& bottom, vector<shared_ptr<Blob<double>>>& top)
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

//        Blob<double> padX(N, C, Hx + 2 * pad_h_, Wx + 2 * pad_w_);
//
//        for (int n = 0; n < N; ++n)
//        {
//            for (int c = 0; c < C; ++c)
//            {
//                for (int h = 0; h < Hx; ++h)
//                {
//                    for (int w = 0; w < Wx; ++w)
//                    {
//                        padX(n,c,h + pad_h_, w + pad_w_) = bottom[0](n,c,h,w);
//                    }
//                }
//            }
//        }

        // pad
//        Blob padX(bottom[0]);
//        double tmpsum;
//        top[0].reset(new Blob(N, F, Ho, Wo));
//        for (int n = 0; n < N; ++n)   //���cube��
//        {
//            for (int f = 0; f < F; ++f)  //���ͨ����
//            {
//                for (int hh = 0; hh < Ho; ++hh)   //���Blob�ĸ�
//                {
//                    for (int ww = 0; ww < Wo; ++ww)   //���Blob�Ŀ�
//                    {
//                        cube window = padX.sub_blob(0,C,hh,hh+kernel_h_-1,ww,ww+kernel_w_-1);
//                        //out = Wx+b
//                        //(*top)[n](hh, ww, f) = accu(window % (*in[1])[f]) + as_scalar((*in[2])[f]);    //b = (F,1,1,1)
//                    }
//                }
//            }
//        }


    }

    void ConvLayer::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "ConvLayer::calc_shape()..." << endl;
        // (N,C,H,W)
        // input and output have the same N
        out_shape.push_back(in_shape[0]);
        // Cout is the number of filters
        out_shape.push_back(num_output_);
        int H_in{in_shape[2]};
        int W_in{in_shape[3]};

        int H_out, W_out;
        H_out = (H_in+2*pad_h_-kernel_h_)/stride_h_+1;
        W_out = (W_in+2*pad_w_-kernel_w_)/stride_w_+1;
        out_shape.push_back(H_out);
        out_shape.push_back(W_out);
    }
}
