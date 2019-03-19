//
// Created by hua on 19-3-7.
//
#include "net.hpp"
using namespace caffe;

Net::Net(const string& model_file, const string& trained_file) {
    NetParameter param;
    ReadNetParamsFromTextFile(model_file, param);
    Init(param);  // 初始化网络模型
}

void Net::Init(const NetParameter& in_param){
    shared_ptr<Layer> layer(NULL);
    // 循环遍历每一层，进行初始化
    for(int layer_id=0; layer_id<in_param.layer_size(); ++layer_id){
        // Setup layer.
        const LayerParameter& layer_param = param.layer(layer_id);
        if("convolution" == layer_param.layer_type)
        {
            Convolution* conv = NULL;
            layer.reset(conv);

        }
        else if ("pooling" == layer_param.layer_type)
        {
            Pooling* pool = NULL;
            layer.reset(pool);

        }
        else if("relu" == layer_param.layer_type)
        {
            Relu* relu = NULL;
            layer.reset(relu);
        }
        layer.Init();  // 打印消息
        layers_[layer_param.name()].reset(layer);
    }
}

