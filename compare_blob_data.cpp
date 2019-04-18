//
// Created by jbk on 19-4-16.
//

#include <compare_blob_data.hpp>
#include <glog/logging.h>


using namespace std;

namespace asr{
    template<typename DType>
    void CompareTopBlob<DType>::CompareInit(const string& load_folder,
            const vector<Blob<DType>* >& net_output_blobs, const vector<string>& layer_names){
        this->layer_names_ = layer_names;
        this->net_output_blobs_ = net_output_blobs;
        this->LoadTopBlob(load_folder);
        this->Compare();
    }
    template<typename DType>
    void CompareTopBlob<DType>::SaveInit(const string& save_folder, const vector<Blob<DType>* >& net_output_blobs, const vector<string>& layer_names){
        this->layer_names_ = layer_names;
        this->net_output_blobs_ = net_output_blobs;
        this->save_folder_ = save_folder;
        this->SaveTopBlob();
    }
    template<typename DType>
    void CompareTopBlob<DType>::SaveTopBlob(){
        CHECK_EQ(this->layer_names_.size(), this->net_output_blobs_.size()) << "top blob size must be same with layer name!";
        Blob<int> shape_blob(1,1,static_cast<int>(floor(this->net_output_blobs_.size())),4);  // 用来保存每个top　blob 的n,c,h,w
        for (int i = 0; i < this->layer_names_.size(); ++i) {
            for (int j = 0; j < shape_blob.shape()[2]; ++j) {
                shape_blob.at(0,0,i,j) = this->net_output_blobs_[i]->shape()[j];
            }

            this->net_output_blobs_[i]->save_data(this->save_folder_ + "/" + this->layer_names_[i] + ".txt");  // 保存　top blob,每个batch保存一个txt
            ofstream fout(this->save_folder_ + "/layer_names.txt", iostream::app);
            if (!fout){
                cout << "无法打开" + this->save_folder_ + "/layer_names.txt" << endl;  // 保存　layer names
            }else{
                fout << this->layer_names_[i] << endl;
            }
            fout.close();
        }
        shape_blob.save_data(this->save_folder_ + "/top_shapes.txt");  // 保存　top blob shapes
        return;
    }
    template<typename DType>
    void CompareTopBlob<DType>::LoadTopBlob(const string& load_folder){
        //　导入 layer names and top blob shapes;
        this->LoadShapes(load_folder);
        // 导入　top blob
        for (int i = 0; i < this->load_layer_names_.size(); ++i) {
            if (this->load_shapes_[i][0] > 1){
                for (int j = 0; j < this->load_shapes_[i][0]; j++){
                    Blob<DType> tmp_batch_blob;
                    tmp_batch_blob.load_data(load_folder + "/" + this->layer_names_[i] + "_batch" + to_string(j+1) + ".txt",
                                             1, this->load_shapes_[i][1], this->load_shapes_[i][2], this->load_shapes_[i][3]);
                    this->load_output_blobs_[i].expand(tmp_batch_blob);
                }
            } else{
                this->load_output_blobs_[i].load_data(load_folder + "/" + this->layer_names_[i] + ".txt",
                        1, this->load_shapes_[i][1], this->load_shapes_[i][2], this->load_shapes_[i][3]);
            }
        }
        return;
    }

    template<typename DType>
    vector<DType> CompareTopBlob<DType>::Compare(){
        for (int i = 0; i < this->layer_names_.size(); ++i) {
            int count = this->net_output_blobs_[i]->shape()[0] * this->net_output_blobs_[i]->shape()[1] *
                    this->net_output_blobs_[i]->shape()[2] * this->net_output_blobs_[i]->shape()[3];
            Blob<DType> tmp_blob = *(this->net_output_blobs_[i]) - this->load_output_blobs_[i];
            this->error_blobs_.push_back(tmp_blob);
            tmp_blob = tmp_blob * tmp_blob;
            vector<DType> tmp_mean = tmp_blob.sum_all_channel();

            for (int j = 1; j < tmp_mean.size(); ++j){
                tmp_mean[0] += tmp_mean[j];
            }
            this->mse_.push_back(tmp_mean[0]);

            vector<DType> tmp_max = tmp_blob.sum_all_channel();
            for (int j = 1; j < tmp_max.size(); ++j){
                tmp_max[0] += tmp_max[j];
            }
            this->max_error_.push_back(tmp_max[0]);

        }
        return this->mse_;
    }


    template<typename DType>
    void CompareTopBlob<DType>::LoadShapes(const string& load_folder)
    {
        // load layer_names and top blob shapes;
        ifstream fin;
        fin.open(load_folder+"/layer_names.txt", ios::in);
        while (!fin.eof())
        {
            string str_tmp;
            getline(fin, str_tmp, '\n');
            this->load_layer_names_.push_back(str_tmp);
        }
        Blob<int> tmp_shapes;
        vector<int> shape_shape{1,1,static_cast<int>(floor(this->load_layer_names_.size())), 4};
        tmp_shapes.load_data(load_folder+"/top_shapes.txt", shape_shape);
        this->load_shapes_.resize(this->load_layer_names_.size());
        for (int i = 0; i < tmp_shapes.shape()[2]; i++){
            for (int j = 0; j < tmp_shapes.shape()[3]; j++){
                this->load_shapes_[i].push_back(tmp_shapes.at(0,0,i,j));
            }
        }
    }

    INSTANTIATE_CLASS(CompareTopBlob);
}
