//
// Created by jbk on 19-3-14.
//

#include <vector>
#include "layer.h"
#include "pooling_layer.h"

namespace zq{

    void PoolLayer::init_layer(const Param& param)
    {
        cout << "PoolLayer::init_layer()..." << endl;
//        layer_param_ = param.layer_param;
//        weight_blob_ = param.weight_blob;
//        is_bias_ = param.is_bias;
//        in_shape_ = param.in_shape;
//        out_shape_ = calc_shape();
        return;
    }

    void PoolLayer::calc_shape(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "PoolLayer::calc_shape()..." << endl;
//        int Ni = in_shape[0];
//        int Ci = in_shape[1];
//        int Hi = in_shape[2];
//        int Wi = in_shape[3];
//
//        int tH = layer_param_.pool_height;
//        int tW = layer_param_.pool_width;
//        int tS = layer_param_.pool_stride;
//
//        int No = Ni;
//        int Co = Ci;
//        int Ho = (Hi - tH) / tS + 1;
//        int Wo = (Wi - tW) / tS + 1;
//
//        out_shape[0] = No;
//        out_shape[1] = Co;
//        out_shape[2] = Ho;
//        out_shape[3] = Wo;
        return;
    }

    void PoolLayer::forward(const vector<shared_ptr<Blob>>& bottom, vector<shared_ptr<Blob>>& top)
    {
        cout << "PoolLayer::forward()..." << endl;
//        if (top)
//            top.reset();
//
//        int N = bottom[0]->get_N();
//        int C = bottom[0]->get_C();
//        int Hx = bottom[0]->get_H();
//        int Wx = bottom[0]->get_W();
//
//        int Hw = layer_param_.pool_height;
//        int Ww = layer_param_.pool_width;
//
//        int Ho = (Hx  - Hw) / layer_param_.pool_stride + 1;
//        int Wo = (Wx - Ww) / layer_param_.pool_stride + 1;
//
//        top.reset(new Blob(N, C, Ho, Wo));
//
//        for (int n = 0; n < N; ++n)
//        {
//            for (int c = 0; c < C; ++c)
//            {
//                for (int hh = 0; hh < Ho; ++hh)
//                {
//                    for (int ww = 0; ww < Wo; ++ww)
//                    {
//                        (*top)[n](hh, ww, c) = (*bottom[0])[n](span(hh*layer_param_.pool_stride, hh*layer_param_.pool_stride + Hw - 1),
//                                                               span(ww*layer_param_.pool_stride, ww*layer_param_.pool_stride + Ww - 1),
//                                                               span(c, c)).max();
//                    }
//                }
//            }
//        }
        return;
    }

}

