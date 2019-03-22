//
// Created by hua on 19-3-4.
//

#ifndef LOADPARAM_NET_HPP
#define LOADPARAM_NET_HPP

#include <caffe.pb.h>
#include <vector>
#include <map>
#include <utility>
#include <memory>
#include <string>
#include <set>

#include "upgrade_proto.hpp"
#include "blob.hpp"
#include "layer.hpp"
//#include "transformer_param.hpp"

using std::vector;
using std::map;
using std::string;
using std::pair;
using std::shared_ptr;
using std::set;

namespace caffe{

class Net{
public:
    /// @brief 显示构造函数：网络模型文件, 训练参数文件
//    explicit Net(const string& model_file, const string& trained_file);
    explicit Net() {}
    virtual ~Net() {}
    /// @brief 使用NetParameter初始化网络
    void Init(const NetParameter& param);

    /// @brief 前向计算，输入：处理数据、计算开始的位置+结束的位置，返回：结束位置对应的运算结果
    const Blob* Forward(const Blob& input_data, const string& begin, const string& end);

protected:
    ;
private:
    // Follow a sequence stored every layer's parameter.
    // Traversal layer's parameter layer by layer from layer id.
    vector<std::shared_ptr<Layer> > layers_;
    map<string, int> layer_name_id_; // 获取layers_中指定层对象

    // bottom
    vector<vector<Blob*> > bottom_vecs_;
    //vector<vector<int> > bottom_id_vecs_;  // 和blob_对应(考虑每一层可能有多个bottom)，一定要加

    // top
    vector<vector<Blob*> > top_vecs_;
    vector<vector<int> > top_id_vecs_;    // 和blob_对应(考虑每一层可能有多个top)，除了训练以外，貌似用不到

    //存储了每一层输出结果, id:[0~layer.size())?
    vector<shared_ptr<Blob>> blobs_;
    vector<string> blob_names_;
    // map<string, int> blob_names_index_;

};

}

#endif //LOADPARAM_NET_HPP
