//
// Created by hua on 19-3-6.
//
#include <iostream>
#include <string>
#include <fstream>
#include <istream>

#include <caffe.pb.h>
#include <upgrade_proto.hpp>

using namespace std;

int ReadNetParamsFromBinaryFile(const string &trained_file, caffe::NetParameter& netparam) {
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
    caffe::LayerParameter* layerparam = netparam.mutable_layer(2);
    const string& layername = layerparam->name();
    const int bottom_size = layerparam->bottom_size();
    const int top_size = layerparam->top_size();
    cout<<"layer_name: "<<layername<<" bottom_size: "<<bottom_size<<" top_size: "<<top_size<<endl;
    const string& bottom_name = layerparam->bottom(0);
    const string& topname = layerparam->top(0);
    cout<<" bottom_name:"<<bottom_name<<"; top_name:"<<topname<<endl;

    caffemodel.close();

    if(!flag){
        cout<<"Parsing net_parameter from io stream error!";
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[]){
    /*
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    caffe::NetParameter netparam;
    string model;
    if (2 == argc){
        model = argv[1];
    }else{
        model = "../res/Squeeze.caffemodel";
    }

    int ret = ReadNetParamsFromBinaryFile(model, netparam);
    cout<<"\"ReadNetParamsFromBinaryFile\" code: "<< ret<<endl;
    if(!ret){
        int layer_size = netparam.layer_size();
        cout<<"layer_size: "<<layer_size<<endl;
    }

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
    */


    caffe::NetParameter param;
    std::string model_file = "../res/det1.prototxt";
    caffe::ReadNetParamsFromTextFile(model_file, &param);
    const int layer_size = param.layer_size();
    std::cout<<"layer_size: "<<layer_size<<endl;
    for(int layer_id = 0; layer_id <layer_size; ++layer_id){
        cout<<param.layer(layer_id).name()<<endl;
    }
    return 0;
}
