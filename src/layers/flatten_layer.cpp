//
// Created by jbk on 19-3-14.
//
#include <vector>
#include "layer.h"
#include "flatten_layer.h"

namespace zq{

    void FcLayer::init_layer(const Param& param)
    {
        cout << "FcLayer::init_layer()..." << endl;
//        layer_param_ = param.layer_param;
//        weight_blob_ = param.weight_blob;
//        is_bias_ = param.is_bias;
        return;
    }

    void FcLayer::calc_shape(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "FcLayer::calc_shape()..." << endl;
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

    void FcLayer::forward(const vector<shared_ptr<Blob>>& bottom, vector<shared_ptr<Blob>>& top)
    {
        cout << "FcLayer::forward()..." << endl;

        return;
    }

}
