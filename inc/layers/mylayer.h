//
// Created by jbk on 19-3-7.
//

#ifndef MYLAYER_LAYER_H
#define MYLAYER_LAYER_H
#include <iostream>
#include <memory>
#include "myBlob.hpp"

using std::vector;
using std::shared_ptr;
using std::string;

struct LayerParam
{

    int pool_stride_w;
    int pool_stride_h;
    int pool_pad_w;
    int pool_pad_h;
    int pool_width;
    int pool_height;
    string pool_types;

    int fc_kenels;

};

class Layer{
public:
    Layer(){}
    // vector<Layer<Dtype>* > layer_;
    // layer_.push_back(&Layer(layer_param));
    //
    virtual ~Layer(){}
    virtual void init_layer() = 0;
    virtual void calc_shape() = 0;
    virtual void forward() = 0;
    vector<int> get_out_shape() const {return out_shape_;}

private:
    LayerParam layer_param_;
    vector<share_ptr<blob>> weight_blob_;
    vector<int> in_shape_;
    vector<int> out_shape_;
    bool is_bias_;
}

#endif
