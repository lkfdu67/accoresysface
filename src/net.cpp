//
// Created by hua on 19-3-7.
//
#include "net.hpp"
#include "caffe.hpp"
using namespace caffe;

//Net::Net(const string& model_file, const string& trained_file) {
//    NetParameter param;
//    ReadNetParamsFromTextFile(model_file, param);
//    Init(param);  // 初始化网络模型
//}

void Net::Init(const NetParameter& in_param){

    const int layers_size = in_param.layer_size();
    bottom_vecs_.resize(layers_size);
    top_vecs_.resize(layers_size);

    shared_ptr<Layer> layer(NULL);
    // 循环遍历每一层，进行初始化
    for(int layer_id=0; layer_id<in_param.layer_size(); ++layer_id){

        const LayerParameter& layer_param = in_param.layer(layer_id);
        pair<string, shared_ptr<Blob>> tmp;
        for(int bottom_id=0; bottom_id<layer_param.bottom_size(); ++bottom_id){
            tmp = make_pair(layer_param.bottom(bottom_id), NULL);
            bottom_vecs_[layer_id].push_back(tmp);
        }

        for(int top_id=0; top_id<layer_param.top_size(); ++top_id){
            tmp = make_pair(layer_param.top(top_id), NULL);
            top_vecs_[layer_id].push_back(tmp);
        }

        if("convolution" == layer_param.type())
        {
            layer.reset(new Convolution);

        }
        else if ("pooling" == layer_param.type())
        {
            layer.reset(new Pooling);

        }
        else if("relu" == layer_param.type())
        {
            layer.reset(new Relu);
        }
        //SetUp中传一个输入尺寸，vector<int>: n, c, h, w?
        layer.SetUp(layer_param, bottom_vecs_[layer_id], top_id_vecs_[layer_id]);  // 打印消息

        layers_[layer_param.name()] = layer;
    }
}
