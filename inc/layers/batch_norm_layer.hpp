//
// Created by jbk on 19-3-14.
//

#ifndef ZQ_BATCH_NORM_LAYER_H
#define ZQ_BATCH_NORM_LAYER_H

#include "layer.h"
#include "transformer_param.hpp"

using std::vector;
using std::shared_ptr;
using std::string;

class BNLayer : public Layer
{
public:
    BNLayer() {}
    ~BNLayer() {}
    void LayerSetUp(const LayerParameter& param);
    void Forward(const vector<shared_ptr<Blob>>& bottom, vector<shared_ptr<Blob>>& top);
};
};
#endif
