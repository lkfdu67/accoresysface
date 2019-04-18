//
// Created by jbk on 19-4-18.
//

#include <iostream>
#include <net.hpp>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <compare_blob_data.hpp>
#include <compare.hpp>

using namespace asr;
namespace compare{
    Classifier::Classifier(const string& model_file,
                           const string& trained_file,
                           const string& mean_file){
        net_.reset(new Net<float>(model_file, trained_file));  // 调用Net的构造函数：初始化网络结构、加载权重文件
        Blob<float>* input_layer = net_->input_blobs()[0]; // 前向网络的第一个输入Blob
        compare_blob_data_.reset(new CompareTopBlob<float>());  // compare对象　　　　　＃＃＃＃＃＃＃
        num_batch_ = input_layer->num();
        num_channels_ = input_layer->channels();
        CHECK(num_channels_ == 3 || num_channels_ == 1)
                        << "Input layer should have 1 or 3 channels.";
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
        if("" != mean_file){
            SetMean(mean_file);
        }
    }

    const std::vector<Blob<float>* > Classifier::Forward(cv::Mat& im,
                                                         int is_compare,
                                                         const string& saved_folder){
        Blob<float>* input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1, num_channels_,
                             im.rows, im.cols);
        Blob<float> input_data(im);
        *input_layer = input_data;

        net_->Reshape();

        /*std::vector<Blob<double>* > out_put_copy(net_->Forward());
        std::vector<Blob<double> > out_put(out_put_copy.size());
        for(int i = 0; i<out_put.size(); ++i){
            out_put[i] = *out_put_copy[i];
        }
        return out_put;*/


        vector<Blob<float>* > output = net_->Forward();

        if (is_compare){
            compare_blob_data_->CompareInit(saved_folder, net_->output_blobs(), net_->get_layer_names());  //保存top结果

        } else{
            compare_blob_data_->SaveInit(saved_folder, net_->output_blobs(), net_->get_layer_names());  // 比较保存的top和计算的top
        }
        return this->net_->output_blobs();
    }

    const std::vector<Blob<float>* > Classifier::Forward(cv::Mat& im,
                                                         const string& begin,
                                                         const string& end,
                                                         int is_compare,
                                                         const string& saved_folder){
        Blob<float>* input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1, num_channels_,
                             im.rows, im.cols);
        Blob<float> input_data(im);
        *input_layer = input_data;

        /*std::vector<Blob<double>* > out_put_copy(net_->Forward(begin, end));
        std::vector<Blob<double> > out_put(out_put_copy.size());
        for(int i = 0; i<out_put.size(); ++i){
            out_put[i] = *out_put_copy[i];
        }
        return out_put;*/

        net_->Forward(begin, end);
        if (is_compare){
            compare_blob_data_->SaveInit(saved_folder, net_->output_blobs(), net_->get_layer_names());

        } else{
            compare_blob_data_->CompareInit(saved_folder, net_->output_blobs(), net_->get_layer_names());
        }

        return this->net_->output_blobs();

    }

    void Classifier::SetMean(const string& mean_file){
        BlobProto blob_proto;
        CHECK(ReadProtoFromBinaryFile(mean_file, &blob_proto));

        // 将BlobProto转换成Blob<float>
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";
        mean_blob.ToCvMat(means_);
    }

    /*
    const std::vector<Blob<double>* > test(){
        cv::Mat im;
        std::shared_ptr<Net> net_ptr;
        net_ptr.reset(new Net("", ""));


        Blob<double>* input_layer = net_ptr->input_blobs()[0];
        input_layer->Reshape(1, 3, im.rows, im.cols);
        Blob<double> input_data(im);
        *input_layer = input_data;

        net_ptr->Reshape();

        return net_ptr->Forward();
    }
    */


}