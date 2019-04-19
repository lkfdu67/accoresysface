//
// Created by jbk on 19-4-16.
//

#include <compare_blob_data.hpp>
#include <glog/logging.h>
#include <cstdio>
#include <iostream>
#include <fstream>


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
        ofstream foutshape(this->save_folder_ + "/top_shapes.txt", iostream::app);  //删除原有top_shapes.txt
        if (foutshape){
//            system( tmp_shape.c_str() );
              std::remove((this->save_folder_ + "/top_shapes.txt").c_str());
        }
        foutshape.close();
        ofstream foutname(this->save_folder_ + "/layer_names.txt", iostream::app);  //删除原有layer_names.txt
        if (foutname){
            std::remove((this->save_folder_ + "/layer_names.txt").c_str());
        }
        foutname.close();
        this->SaveTopBlob();
    }
    template<typename DType>
    void CompareTopBlob<DType>::SaveTopBlob(){
        CHECK_EQ(this->layer_names_.size(), this->net_output_blobs_.size()) << "top blob size must be same with layer name!";
        for (int i = 0; i < this->layer_names_.size(); ++i) {
            this->net_output_blobs_[i]->save_data(this->save_folder_ + "/" + this->layer_names_[i] + ".txt");  // 保存　top blob,每个batch保存一个txt
            ofstream foutname(this->save_folder_ + "/layer_names.txt", iostream::app);
            if (!foutname){
                cout << "无法打开" + this->save_folder_ + "/layer_names.txt" << endl;  // 保存　layer names
            }else{
                foutname << this->layer_names_[i] << endl;
            }
            foutname.close();
            ofstream foutshape(this->save_folder_ + "/top_shapes.txt", iostream::app);
            if (!foutshape){
                cout << "无法打开" + this->save_folder_ + "/top_shapes.txt" << endl;  // 保存　top shapes
            }else{
                foutshape << this->net_output_blobs_[i]->shape()[0] << " "
                        << this->net_output_blobs_[i]->shape()[1] << " "
                        << this->net_output_blobs_[i]->shape()[2] << " "
                        << this->net_output_blobs_[i]->shape()[3] << endl;
            }
            foutshape.close();
        }
        return;
    }
    template<typename DType>
    void CompareTopBlob<DType>::LoadTopBlob(const string& load_folder){
        //　导入 layer names and top blob shapes;
        this->LoadShapes(load_folder);
        // 导入　top blob
        this->load_output_blobs_.resize(this->load_layer_names_.size());
        for (int i = 0; i < this->load_layer_names_.size(); ++i) {
            if (this->load_shapes_[i][0] > 1){
                for (int j = 0; j < this->load_shapes_[i][0]; j++){
                    Blob<DType> tmp_batch_blob;
                    tmp_batch_blob = tmp_batch_blob.load_data(load_folder + "/" + this->layer_names_[i] + "_batch" + to_string(j+1) + ".txt",
                                             1, this->load_shapes_[i][1], this->load_shapes_[i][2], this->load_shapes_[i][3]);
                    this->load_output_blobs_[i].expand(tmp_batch_blob);
                }
            } else{
                this->load_output_blobs_[i] = this->load_output_blobs_[i].load_data(load_folder + "/" + this->layer_names_[i] + ".txt",
                        1, this->load_shapes_[i][1], this->load_shapes_[i][2], this->load_shapes_[i][3]);
            }
        }
        return;
    }

    template<typename DType>
    vector<DType> CompareTopBlob<DType>::Compare(){
        for (int i = 0; i < this->layer_names_.size(); ++i) {
            Blob<DType> tmp_blob = *(this->net_output_blobs_[i]) - this->load_output_blobs_[i];
            this->error_blobs_.push_back(tmp_blob);
            tmp_blob = tmp_blob * tmp_blob;
            vector<DType> tmp_mean = tmp_blob.sum_all_channel();

            for (int j = 1; j < tmp_mean.size(); ++j){
                tmp_mean[0] += tmp_mean[j] / tmp_mean.size();
            }
            this->mse_.push_back(tmp_mean[0]);

            vector<DType> tmp_max = tmp_blob.max_all_channel();
            for (int j = 1; j < tmp_max.size(); ++j){
                tmp_max[0] += tmp_max[j] / tmp_mean.size();
            }
            this->max_error_.push_back(tmp_max[0]);
        }
        cout << "max_error:" << '\t';
        this->PrintVector(this->max_error_);
        cout << "mse      :" << '\t';
        this->PrintVector(this->mse_);
        return this->mse_;
    }


    template<typename DType>
    void CompareTopBlob<DType>::LoadShapes(const string& load_folder)
    {
        // load layer_names and top blob shapes;
        ifstream fin;
        fin.open(load_folder+"/layer_names.txt", ios::in);
        int i = 0;
        while (!fin.eof() and i < this->layer_names_.size())
        {
            string str_tmp;
            getline(fin, str_tmp, '\n');
            this->load_layer_names_.push_back(str_tmp);
            i++;
        }

        this->load_shapes_.resize(this->layer_names_.size());
        ifstream infile;  // 定义读取文件流，相对于程序来说是in
        infile.open(load_folder+"/top_shapes.txt");  // 打开文件
        for (int i = 0; i < this->load_layer_names_.size(); i++)
        {
            this->load_shapes_[i].resize(4);
            for (int j = 0; j < 4; j++)
            {
                infile >> this->load_shapes_[i][j];// 读取一个值（空格、制表符、换行隔开）就写入到矩阵中
            }
        }
        infile.close();
    }

    INSTANTIATE_CLASS(CompareTopBlob);
}
