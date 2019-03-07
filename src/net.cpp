//
// Created by hua on 19-3-7.
//
#include "../inc/net.hpp"
using namespace caffe;

Net::Net(const string &trained_file) {
    Init(trained_file);
}

///均值处理放到外层去做
//Net::Net(const string &trained_file, const string &mean_file) {
//    Init(trained_file);
//    set_mean(mean_file);
//}

void Net::Init(const string &trained_file) {
    bool success = true;
    NetParameter* param;
    if(!ReadNetParamsFromBinaryFile(trained_file, param))
    {
       cout<<"Reading net parameters is successful!"<<endl;
    }else{
        cout<<"Reading net parameters is failed!"<<endl;
        success = false;
        assert(success);
    }
    if(!UpdateNet(param)){
        cout<<"Updating net parameters is successful!"<<endl;
    }else{
        cout<<"Updating net parameters is failed!"<<endl;
        success = false
        assert(success);
    }
}

int Net::ReadNetParamsFromBinaryFile(const string &trained_file, NetParameter *param) {
    // 判断文件存在与否
    // 。。。

    return 0;
}

int Net::UpdateNet(NetParameter* param){


    return 0;
}

void Net::Forward(const caffe::Blob<Dtype>& input_data, const string &begin, const string &end){
    ;
}