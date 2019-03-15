//
// Created by hua on 19-3-13.
// 负责将.prototxt文件参数解析到内存中

#ifndef LOADPARAM_UPGRADE_PROTO_HPP
#define LOADPARAM_UPGRADE_PROTO_HPP

#include <fcntl.h>
#include <unistd.h>

#include <string>
#include <map>
#include <caffe.pb.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <glog/logging.h>

using std::string;
using std::map;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

#define O_RDONLY	     00

namespace caffe {
    // Read parameters from a file into a NetParameter proto message.
    void ReadNetParamsFromTextFile(const string& param_file,
                                   NetParameter* param);

    bool UpgradeNetAsNeeded(const string& param_file, NetParameter* param);

    bool NetNeedsV0ToV1Upgrade(const NetParameter& net_param);

    bool UpgradeV0Net(const NetParameter& v0_net_param_padding_layers,
                      NetParameter* net_param);

    void UpgradeV0PaddingLayers(const NetParameter& param,
                                NetParameter* param_upgraded_pad);

    bool UpgradeV0LayerParameter(const V1LayerParameter& v0_layer_connection,
                            V1LayerParameter* layer_param);

    bool ReadProtoFromTextFile(const char* filename, Message* proto);

    inline bool ReadProtoFromTextFile(const string& filename, Message* proto);

    bool NetNeedsDataUpgrade(const NetParameter& net_param);

    void UpgradeNetDataTransformation(NetParameter* net_param);

    bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param);

    bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param);

    bool NetNeedsInputUpgrade(const NetParameter& net_param);

    void UpgradeNetInput(NetParameter* net_param);

    bool NetNeedsBatchNormUpgrade(const NetParameter& net_param);

    void UpgradeNetBatchNorm(NetParameter* net_param);

    V1LayerParameter_LayerType UpgradeV0LayerType(const string& type);

    bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                                 LayerParameter* layer_param);

    const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);
}

#endif //LOADPARAM_UPGRADE_PROTO_HPP
