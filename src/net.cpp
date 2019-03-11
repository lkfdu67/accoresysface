//
// Created by hua on 19-3-7.
//
#include "../inc/net.hpp"
using namespace caffe;

template <typename Dtype>
Net<typename Dtype>::Net(const string &trained_file) {
    Init(trained_file);
}

template <typename Dtype>
void Net<typename Dtype>::Init(const string &trained_file) {
    bool success = true;
    NetParameter* param;
    if(!ReadNetParamsFromBinaryFile(trained_file, param))
    {
       cout<<"Reading net parameters is successful!"<<endl;
    }else{
        cout<<"Reading net parameters is failed!"<<endl;
        success = false;
        assert(success);  // 调试
    }
    if(!UpdateNet(param)){
        cout<<"Updating net parameters is successful!"<<endl;
    }else{
        cout<<"Updating net parameters is failed!"<<endl;
        success = false
        assert(success);  //调试
    }
}

template <typename Dtype>
int Net<typename Dtype>::ReadNetParamsFromBinaryFile(const string &trained_file, caffe::NetParameter& netparam) {
    /// 解析正确返回： 0,否则返回： 1
    ifstream caffemodel(trained_file, ifstream::in | ifstream::binary);

    if (&caffemodel == NULL){
        cout << "The ptr of caffemodel is NULL" << endl;
        return 1;
    }

    if (!caffemodel.is_open()){
        cout << "Can not open model" << endl;
        return 1;
    }

    // 解析参数，存入netparam成员变量中
    bool flag = netparam.ParseFromIstream(&caffemodel);

    caffemodel.close();

    if(!flag){
        cout<<"Parsing net_parameter from io stream error!";
        return 1;
    }
    return 0;
}

template <typename Dtype>
int Net<typename Dtype>::UpdateNet(NetParameter* param){

    return 0;
}

template <typename Dtype>
void Net<typename Dtype>::Forward(const caffe::Blob<Dtype>& input_data, const string &begin, const string &end){
    ;
}