//
// Created by hua on 19-3-4.
//

#ifndef NET_HPP_
#define NET_HPP_

namespace caffe{
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
        /// @brief 显示构造函数：训练参数文件+均值文件（可省略）
        explicit Net(const string& trained_file);
//      explicit Net(const string& trained_file, const string& mean_file);
//      explicit Net(const string& param_file, const string& trained_file);
//      explicit Net(const string& param_file, const string& trained_file, const string& mean_file);

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
//      bool set_mean_ = false;

    };

}

#endif //NET_HPP_
