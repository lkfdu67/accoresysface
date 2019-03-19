//
// Created by jbk on 19-3-14.
//

#ifndef ZQ_POOLING_LAYER_H
#define ZQ_POOLING_LAYER_H

#include "mylayer.h"

using std::vector;
using std::shared_ptr;
using std::string;


class PoolLayer : public Layer
{
public:
    PoolLayer();
    ~PoolLayer(){}
    void init_layer(const Param& param);
    void calc_shape(const vector<int>& in_shape, vector<int>& out_shape);
    void forward(const vector<shared_ptr<Blob>>& bottom, vector<shared_ptr<Blob>>& top);
};
#endif
