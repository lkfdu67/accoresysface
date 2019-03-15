//
// Created by hua on 19-3-4.
//

#ifndef LOADPARAM_NET_HPP
#define LOADPARAM_NET_HPP

#include <caffe.pb.h>
#include <vector>
#include "upgrade_proto.hpp"
#include "blob.hpp"
#include "layer.hpp"

using std::vector;

namespace caffe{
    template <typename Dtype>
    class Net{
    public:
        /// @brief 显示构造函数：网络模型文件, 训练参数文件
        explicit Net(const string& model_file, const string& trained_file);
        /// @brief 显示构造函数：训练参数文件+均值文件（可省略）
        explicit Net(const string& trained_file);

        /// @brief 使用NetParameter初始化网络
        void Init(const NetParameter& param);
        /// @brief 使用trained_file文件初始化网络
        void Init(const string& trained_file);

        /// @brief 前向计算，输入：处理数据、计算开始的位置+结束的位置，返回：结束位置对应的运算结果
        const Blob<Dtype>* Forward(const Blob<Dtype>& input_data, const string& begin, const string& end);

    protected:
        /// @brief NetParameter为结构体或类，从二进制文件中将参数解析出来，存入NetParameter
        int ReadNetParamsFromBinaryFile(const string& trained_file, NetParameter* param);
        /// @brief 跟新网络参数，通过NetParameter参数，并将更新过的网络参数保存到成员变量中
        int UpdateNet(NetParameter* param);
        /// @brief 为网络设置均值处理
        int set_mean(const string& mean_file);
    private:
        NetParameter net_parameter_;
        // layer
        vector<Layer<Dtype>*> layers_;
        // bottom
        vector<vector<Blob<Dtype>*> > bottom_vecs_;
        // top
        vector<vector<Blob<Dtype>*> > top_vecs_;

    };

}

#endif //LOADPARAM_NET_HPP
