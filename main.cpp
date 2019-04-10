#include <iostream>
#include <net.hpp>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe{

    class Classifier final{
    private:
        shared_ptr<Net> net_;
        cv::Size input_geometry_;  // 输入图片的Size:W×H
        int num_channels_;
    public:
        Classifier(const string& model_file,
                   const string& trained_file,
                   const string& mean_file);

        const std::vector<Blob<double>* > Forward(cv::Mat& im);
        const std::vector<Blob<double>* > Forward(cv::Mat& im, const string& begin, const string& end);

        /// @brief 设置均值文件.
        void set_mean(const string& mean_file);
    };

    Classifier::Classifier(const string& model_file,
                           const string& trained_file,
                           const string& mean_file){
        net_.reset(new Net(model_file, trained_file));  // 调用Net的构造函数：初始化网络结构、加载权重文件
        Blob<double>* input_layer = net_->input_blobs()[0]; // 前向网络的第一个输入Blob
        num_channels_ = input_layer->channels();
        CHECK(num_channels_ == 3 || num_channels_ == 1)
                        << "Input layer should have 1 or 3 channels.";
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
        // set mean, 暂时省略
    }

    const std::vector<Blob<double>* > Classifier::Forward(cv::Mat& im){
        Blob<double>* input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1, num_channels_,
                             im.rows, im.cols);

        Blob<double> input_data(im);
        *input_layer = input_data;

        net_->Reshape();

        /*std::vector<Blob<double>* > out_put_copy(net_->Forward());
        std::vector<Blob<double> > out_put(out_put_copy.size());
        for(int i = 0; i<out_put.size(); ++i){
            out_put[i] = *out_put_copy[i];
        }
        return out_put;*/
        return net_->Forward();
    }

    const std::vector<Blob<double>* > Classifier::Forward(cv::Mat& im, const string& begin, const string& end){
        Blob<double>* input_layer = net_->input_blobs()[0];  // 目前仅支持1个输入的网络
        input_layer->Reshape(1, num_channels_,
                             im.rows, im.cols);
        Blob<double> input_data(im);
        *input_layer = input_data;

        /*std::vector<Blob<double>* > out_put_copy(net_->Forward(begin, end));
        std::vector<Blob<double> > out_put(out_put_copy.size());
        for(int i = 0; i<out_put.size(); ++i){
            out_put[i] = *out_put_copy[i];
        }
        return out_put;*/

        return net_->Forward(begin, end);
    }

    void Classifier::set_mean(const string& mean_file){
        ;
    }
}

int main() {
    const string& model_file = "../res/det1.prototxt";
    const string& trained_file = "../res/det1.caffemodel";
    cv::Mat im = cv::imread("../res/test.jpg");
    cv::Mat sample_float, sample_normalized;

    // BRG -- > RGB, (im - 127.5) / 128.0
    cv::imshow("src", im);
    cv::cvtColor(im, im, CV_RGB2BGR);

    sample_float.convertTo(sample_float, CV_32FC3);
    cv::subtract(sample_float, 127.5, sample_normalized);
    for (int i=0;i<sample_normalized.rows;i++){
        sample_normalized.row(i)=(sample_normalized.row(i) / 128.0);
    }

    cout<<CV_VERSION<<endl;
    caffe::Classifier classifier(model_file, trained_file, "");
    //std::vector<caffe::Blob<double>* > output = classifier.Forward(sample_normalized);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}


