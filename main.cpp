#include <iostream>
#include <net.hpp>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe{

//    class Classifier final{
//    private:
//        shared_ptr<Net> net_;
//        cv::Size input_geometry_;
//        int num_channels_;
//    public:
//        Classifier(const string& model_file,
//                   const string& trained_file,
//                   const string& mean_file);
//
//        const std::vector<Blob<double>* >& Forward(cv::Mat& im);
//        const std::vector<Blob<double>* >& Forward(cv::Mat& im, const string& begin, const string& end);
//    };
//
//    Classifier::Classifier(const string& model_file,
//                           const string& trained_file,
//                           const string& mean_file,
//                           const string& label_file){
//        net_.reset(new Net(model_file, trained_file));
//        Blob<double>* input_layer = net_->input_blobs()[0];
//        num_channels_ = input_layer->channels();
//        CHECK(num_channels_ == 3 || num_channels_ == 1)
//                        << "Input layer should have 1 or 3 channels.";
//        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
//        // set mean, 暂时省略
//    }
//
//    const std::vector<Blob<double>* >& Classifier::Forward(cv::Mat& im){
//        Blob<double>* input_layer = net_->input_blobs()[0];  // 目前仅支持1个输入的网络
//        //input_layer->Reshape(1, num_channels_,
//        //                     input_geometry_.height, input_geometry_.width);
//        Blob<double> input_data(im);
//        *input_layer = input_data;
//
//        return net_->Forward(*input_layer);
//    }
//
//    const std::vector<Blob<double>* >& Classifier::Forward(cv::Mat& im, const string& begin, const string& end){
//        Blob<double>* input_layer = net_->input_blobs()[0];  // 目前仅支持1个输入的网络
//        //input_layer->Reshape(1, num_channels_,
//        //                     input_geometry_.height, input_geometry_.width);
//        Blob<double> input_data(&im);
//        *input_layer = input_data;
//
//        return net_->Forward(*input_layer, begin, end);
//    }
//
}

int main() {
    const string& model_file = "../res/det1.prototxt";
    const string& trained_file = "";
    using namespace cv;
    cv::Mat im = cv::imread("../res/test.jpg",0);
//    caffe::Classifier classifier(model_file, trained_file, "");
//    std::vector<caffe::Blob<double>* > output = classifier.Forward(im);
    std::cout << "Hello, World!" << std::endl;

    return 0;
}


