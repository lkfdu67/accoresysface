//
// Created by hua on 19-3-14.
//

#ifndef LOADPARAM_LAYER_HPP
#define LOADPARAM_LAYER_HPP

#include "caffe.pb.h"

namespace caffe{
    template <typename Dtype>
    class Layer{
        ;
    };

    template <typename Dtype>
    class LayerRegistry{
        Layer* CreateLayer(const LayerParameter& layer_param);
    };
}

#endif //LOADPARAM_LAYER_HPP
