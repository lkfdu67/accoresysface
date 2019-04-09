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

#include <upgrade_proto.hpp>
#include <blob_.hpp>
#include <layer.hpp>
//#include "transformer_param.hpp"

using std::vector;
using std::map;
using std::string;
using std::pair;
using std::shared_ptr;
using std::set;
using std::cout;
using std::endl;

namespace caffe{

// Net类不可继承
class Net final{
public:
    // explicit Net(const string& model_file, const string& trained_file);/// @brief 显示构造函数：网络模型文件, 训练参数文件
    explicit Net(const string& model_file, const string& trained_file);
    ~Net() {}

    /// @brief 使用NetParameter初始化网络
    void Init(const NetParameter& param);

    /// @brief 使用NetParameter初始化网络
    void CopyTrainedParams(const string& trained_file);

    /*实现前向计算
     * note: 使用前请填充input_layer
     * 输入：计算开始的位置、结束的位置，
     * 返回: 从开始位置到结束位置的前向运算结果,类型：vector<Blob*>
     * */
    const vector<Blob<double>* > Forward(const string& begin, const string& end);

    /*实现前向计算
     * note: 使用前请填充input_layer
     * 返回: 网络第一层到最后一层的前向运算结果,类型：vector<Blob*>
     * */
    const vector<Blob<double>* > Forward();

    /// @brief 得到Input blobs
    inline const vector<Blob<double>* >& input_blobs() const {
        return net_input_blobs_;
    }

    /// @brief 得到前向运算结果
    inline const vector<Blob<double>* >& output_blobs() const {
        return net_output_blobs_;
    }

protected:
    ;
private:
    // layer
    vector<std::shared_ptr<Layer> > layers_;
    map<string, int> layer_name_id_; // 获取layers_中指定层对象
    vector<string> layer_names_;

    // bottom
    vector<vector<Blob<double>* > > bottom_vecs_;
    //vector<vector<shared_ptr<Blob<double>> > > bottom_vecs_;

    // top
    vector<vector<Blob<double>* > > top_vecs_;
    //vector<vector<shared_ptr<Blob<double>> > > top_vecs_;
    vector<vector<int> > top_id_vecs_;    // 和blob_对应(考虑每一层可能有多个top)，除了训练以外，貌似用不到

    //存储了每一层输出结果, blobs_.size() >= layers.size()
    vector<shared_ptr<Blob<double> > > blobs_;
    vector<string> blob_names_;  // bottom、top 通过名字访问对应内存

    //net input
    vector<Blob<double>* > net_input_blobs_;


    //net output
    vector<Blob<double>* > net_output_blobs_;

};

}

#endif //LOADPARAM_NET_HPP
