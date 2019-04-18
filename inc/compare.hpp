//
// Created by jbk on 19-4-18.
//

#ifndef LOADPARAM_COMPARE_HPP
#define LOADPARAM_COMPARE_HPP

#include <iostream>
#include <net.hpp>
#include <vector>
#include <compare_blob_data.hpp>

using namespace asr;
namespace compare{

    class Classifier final{
    private:
        shared_ptr<Net<float>> net_;
        shared_ptr<CompareTopBlob<float>> compare_blob_data_;  //　compare 对象指针
        cv::Size input_geometry_;  // 输入图片的Size:W×H
        int num_channels_;
        int num_batch_;
        vector<cv::Mat> means_;
    public:
        Classifier(const string& model_file,
                   const string& trained_file,
                   const string& mean_file);

        const std::vector<Blob<float>* > Forward(cv::Mat& im, int is_compare = 0, const string& saved_folder = "./saved_data");
        const std::vector<Blob<float>* > Forward(cv::Mat& im, const string& begin, const string& end, int is_compare = 0, const string& saved_folder = "./saved_data");

        /// @brief 设置均值文件.
        void SetMean(const string& mean_file);

        inline vector<cv::Mat> get_means(){
            return means_;
        }
    };

}

#endif //LOADPARAM_COMPARE_HPP
