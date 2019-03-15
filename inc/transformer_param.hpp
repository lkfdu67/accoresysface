//
// Created by hua on 19-3-14.
//

#ifndef LOADPARAM_TRANSFORMER_PARAM_HPP
#define LOADPARAM_TRANSFORMER_PARAM_HPP

#include <vector>
#include <string>
#include "blob.hpp"
using std::vector;
using std::string;
using std::shared_ptr;

namespace caffe{
/**
 * @brief Provides parameters to transform in the net.
 *
 * This LayerParameter provides paramers for kinds of layers;
 * NetParameter contains the paramers of the net model.
 */
struct ConvolutionParameter{
    int pad_w;
    int pad_h;
    int stride_w;
    int stride_h;
    int kernel_w;
    int kernel_h;

    int num_filters;
    int num_channel;
    vector<shared_ptr<Blob>> weights;
};
struct PoolingParameter{
;
};
struct ReLUParameter{
;
};
struct FcParameter{
    int num_filters;
};
struct SoftmaxParameter{
;
};


struct LayerParameter{
    string layer_type;
    string layer_name;
    vector<string> bottom_names;
    vector<string> top_names;

    ConvolutionParameter conv_param;
    PoolingParameter pool_param;
    ReLUParameter relu_param;
    SoftmaxParameter softmax_param;
};

struct NetParameter{
    string net_name;
    vector<LayerParameter> layers_parameter;
};

}

#endif //LOADPARAM_TRANSFORMER_PARAM_HPP
