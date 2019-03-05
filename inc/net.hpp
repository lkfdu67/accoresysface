//
// Created by hua on 19-3-4.
//

#ifndef FORWARDNET_NET_H
#define FORWARDNET_NET_H

template <typename Dtype>
class Blob{
    ;
};
template <typename Dtype>
class Layer{
    ;
};

template <typename Dtype>
class Net{
public:
    // 显示构造函数：输入网络结构文件+参数文件+均值文件（可省略）
    explicit Net(const string& param_file, const string& trained_file);
    explicit Net(const string& param_file, const string& trained_file, const string& mean_file);
    // 为net中每一个layer初始化参数
    void CopyTrainedLayersFrom();
    // 前向计算，输入：计算开始的位置+结束的位置，返回：结束位置对应的运算结果
    const Blob<Dtype>* Forward(const string& begin, const string& end);

protected:
    ;
};

#endif //FORWARDNET_NET_H
