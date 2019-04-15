//
// Created by hua on 19-3-7.
//
#include <net.hpp>
#include <asr.hpp>
#include <upgrade_proto.hpp>

namespace asr{
template<typename DType>
Net<DType>::Net(const string& model_file, const string& trained_file){
    asr::NetParameter param;
    asr::ReadNetParamsFromTextFile(model_file, &param);
    Init(param);
    CopyTrainedParams(trained_file);
}

template<typename DType>
void Net<DType>::Init(const NetParameter& in_param){
    map<string, int> blob_name_to_idx;  // 可以通过blob_names_来对应id,但需注意vector本身没有实现find方法
    set<string> available_blobs;
    const int layers_size = in_param.layer_size();

    bottom_vecs_.resize(layers_size);
    cout<<"layers_size: "<<layers_size<<endl; //debug
    cout<<"bottom_vecs_ size: "<<bottom_vecs_.size()<<endl; //debug
    top_vecs_.resize(layers_size);
    top_id_vecs_.resize(layers_size);

    // 循环遍历每一层，进行初始化bottom、top对应的blob
    for(int layer_id=0; layer_id<in_param.layer_size(); ++layer_id){
        const LayerParameter& layer_param = in_param.layer(layer_id);
        shared_ptr<Layer<DType>> layer_pointer(NULL);
        layer_name_id_[layer_param.name()] = layer_id;
        layer_names_.push_back(layer_param.name());

        for(int bottom_id=0; bottom_id<layer_param.bottom_size(); ++bottom_id){
            const string& blob_name = layer_param.bottom(bottom_id);
            if(available_blobs.find(blob_name) != available_blobs.end()){
                const int blob_id = blob_name_to_idx[blob_name];
                //bottom_vecs_[layer_id].push_back(blobs_[blob_id]);
                bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
                // available_blobs.erase(blob_name);
            }
            else{
                std::cout<< "Unknown bottom blob '" << blob_name << "' (layer '"
                         << layer_param.name() << "', bottom index " << bottom_id << ")"
                         <<std::endl;
                exit(0);  // debug
            }
        }

        for(int top_id=0; top_id<layer_param.top_size(); ++top_id){
            shared_ptr<Blob<DType> > blob_pointer(new Blob<DType>());
            //top_vecs_[layer_id].push_back(blob_pointer);
            top_vecs_[layer_id].push_back(blob_pointer.get());
            //top_id_vecs_[layer_id].push_back(top_id);
            const string& blob_name = layer_param.top(top_id);
            blob_names_.push_back(blob_name);
            const int blob_id = blobs_.size();
            blobs_.push_back(blob_pointer);
            top_id_vecs_[layer_id].push_back(blob_id);
            blob_name_to_idx[blob_name] = blob_id;
            available_blobs.insert(blob_name);
            // net output
            net_output_blobs_.push_back(blob_pointer.get());  //?
            // net input
            if ("Input" ==  layer_param.type()) {
                const int blob_id_input = blobs_.size() - 1;
                net_input_blobs_.push_back(blobs_[blob_id_input].get());
            }
        }

        //std::cout<<layer_param.type()<<std::endl;
        if("Input" == layer_param.type())
        {
            layer_pointer.reset(new InputLayer<DType>);
        }
        else if("Convolution" == layer_param.type())
        {
            layer_pointer.reset(new ConvLayer<DType>);
        }
        else if("BatchNorm" == layer_param.type()){
            layer_pointer.reset(new BNLayer<DType>);
        }
        else if ("Pooling" == layer_param.type())
        {
            layer_pointer.reset(new PoolLayer<DType>);
        }
        else if("ReLU" == layer_param.type())
        {
            layer_pointer.reset(new ReluLayer<DType>);
        }
        else if("PReLU" == layer_param.type())
        {
            layer_pointer.reset(new PReluLayer<DType>);
        }
        else if("InnerProduct" == layer_param.type()){
            layer_pointer.reset(new FCLayer<DType>);
        }
        else if ("Softmax" == layer_param.type())
        {
            layer_pointer.reset(new SoftmaxLayer<DType>);
        }
        //SetUp中传一个输入尺寸，vector<int>: n, c, h, w?
        if(layer_pointer){
            // 权重blobs和bottom、top reshape成网络每一层所需的对应维度
            layer_pointer->SetUp(layer_param, bottom_vecs_[layer_id], top_vecs_[layer_id]);
            layers_.push_back(layer_pointer);
        }
    }
}

template<typename DType>
void Net<DType>::CopyTrainedParams(const string& trained_file) {
    NetParameter param;
    // 检查trained_file文件是否合法，不合法退出
    CHECK(ReadProtoFromBinaryFile(trained_file, &param))
                    << "Failed to parse NetParameter file: " << trained_file;

    UpgradeNetAsNeeded(trained_file, &param);

    int num_source_layers = param.layer_size();
    for(int i=0;i<num_source_layers; ++i){
        const LayerParameter& source_layer = param.layer(i);
        const string& source_layer_name = source_layer.name();
        int target_layer_id = 0;
        while(target_layer_id != layer_names_.size() &&
              layer_names_[target_layer_id] != source_layer_name){
            ++target_layer_id;
        }
        if(target_layer_id == layer_names_.size()){
            std::cout << "Ignoring source layer " << source_layer_name
                      <<std::endl;
            continue;
        }
        cout<< "Copying source layer " << source_layer_name<<endl;

        vector<shared_ptr<Blob<DType> > >& target_blobs =
                layers_[target_layer_id]->weights();

        if(target_blobs.size() != source_layer.blobs_size()){
            cout<< "Incompatible number of blobs for layer " << source_layer_name
                <<endl;
            exit(0);  //debug
        }
        for(int j = 0; j<target_blobs.size(); ++j){
            // 判断权重和层对应的blob维度是否相同
            // if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j)))
            if(!target_blobs[j]->ShapeEquals(source_layer.blobs(j))){
                Blob<DType> source_blob;
                // 根据参数source_layer.blobs(j)，reshape source_blob。
                cout<< "Cannot copy param " << j << " weights from layer '"
                    << source_layer_name << "'; shape mismatch.  Source param shape is "
                    << source_blob.shape_string() << "; target param shape is "
                    << target_blobs[j]->shape_string() << ". "
                    << "To learn this layer's parameters from scratch rather than "
                    << "copying from a saved net, rename the layer.";
                exit(0);  // debug
            }
            const bool kReshape = false;
            target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
        }
    }
    cout<<"CopyTrainedParams"<<endl;
}

template<typename DType>
const vector<Blob<DType>* > Net<DType>::Forward(const string& begin, const string& end){
    const int begin_id = layer_name_id_[begin];
    const int end_id = layer_name_id_[end];
    CHECK_GE(begin_id, 0);
    CHECK_LE(end_id, layers_.size());

    for(int i =begin_id; i<=end_id; ++i){
        layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    }

    vector<Blob<DType>*> net_out_blobs(end_id - begin_id + 1);
    //vector<Blob<double> > net_out_blobs(end_id - begin_id + 1);
    slice_blobs(net_output_blobs_, net_out_blobs, begin_id, end_id + 1);

    return net_out_blobs;  // 存储的是top_vecs_的指针
}

template<typename DType>
const vector<Blob<DType>* > Net<DType>::Forward(){
    for(int i =0; i<layers_.size(); ++i){
        layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    }

    return net_output_blobs_;  // 存储的是top_vecs_的指针
}

template<typename DType>
void Net<DType>::Reshape(){
    for(int i=0; i< layers_.size(); ++i){
        // const vector<Blob<Dtype>*>& bottom,
        //      const vector<Blob<Dtype>*>& top
        layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
    }
}

template<typename DType>
void Net<DType>::slice_blobs(const vector<Blob<DType>*> & dvcs, vector<Blob<DType>* > dret, int start, int end) {
    for (int i= start, j=0; i< end; ++i, ++j) {
        dret[j] = dvcs[i];
    }
}

INSTANTIATE_CLASS(Net);

}