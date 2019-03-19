//
// Created by hua on 19-3-14.
//

#ifndef LOADPARAM_LAYER_HPP
#define LOADPARAM_LAYER_HPP

#include "caffe.pb.h"

namespace caffe{

    class Layer{
        explicit Layer(){}
        virtual ~Layer(){}
        virtual void Init() = 0;
    };
}

#endif //LOADPARAM_LAYER_HPP
