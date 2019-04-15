#include <iostream>
#include <net.hpp>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace asr{

    class Classifier final{
    private:
        shared_ptr<Net<float>> net_;
        cv::Size input_geometry_;  // 输入图片的Size:W×H
        int num_channels_;
        int num_batch_;
        vector<cv::Mat> means_;
    public:
        Classifier(const string& model_file,
                   const string& trained_file,
                   const string& mean_file);

        const std::vector<Blob<float>* > Forward(cv::Mat& im);
        const std::vector<Blob<float>* > Forward(cv::Mat& im, const string& begin, const string& end);

        /// @brief 设置均值文件.
        void SetMean(const string& mean_file);

        inline vector<cv::Mat> get_means(){
            return means_;
        }
    };

    Classifier::Classifier(const string& model_file,
                           const string& trained_file,
                           const string& mean_file){
        net_.reset(new Net<float>(model_file, trained_file));  // 调用Net的构造函数：初始化网络结构、加载权重文件
        Blob<float>* input_layer = net_->input_blobs()[0]; // 前向网络的第一个输入Blob
        num_batch_ = input_layer->num();
        num_channels_ = input_layer->channels();
        CHECK(num_channels_ == 3 || num_channels_ == 1)
                        << "Input layer should have 1 or 3 channels.";
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

        if("" != mean_file){

            SetMean(mean_file);
        }
    }

    const std::vector<Blob<float>* > Classifier::Forward(cv::Mat& im){
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
        return net_->Forward();
    }

    const std::vector<Blob<float>* > Classifier::Forward(cv::Mat& im, const string& begin, const string& end){
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

        return net_->Forward(begin, end);
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

int main() {
    const string& model_file = "../res/det1.prototxt";
    const string& trained_file = "../res/det1.caffemodel";

    asr::Classifier classifier(model_file, trained_file, "../res/mean.binaryproto");
    vector<cv::Mat> means = classifier.get_means();
    if(0 < means.size()){
        ; // cv::subtract(sample_float, means[0], sample_normalized);
    }
    /*将均值文件转成Mat, 并可视化, 仅调试用*/
/*    cv::Mat im_show;
    means[0].copyTo(im_show);
    normalize(im_show, im_show, 1.0, 0.0, cv::NORM_MINMAX); //归一到0~1之间
    im_show.convertTo(im_show, CV_8UC3, 255, 0);
    cv::imshow("", im_show);
    cv::waitKey(0);*/


    cv::Mat im = cv::imread("../res/test.jpg");
    cv::transpose(im, im);
    cv::Mat sample_float, sample_normalized;

    // RGB -- > BRG, (im - 127.5) / 128.0, ubuntu c++ opencv default: RGB
    cv::cvtColor(im, sample_float, CV_RGB2BGR);

    sample_float.convertTo(sample_float, CV_32FC3);
    cv::subtract(sample_float, 127.5, sample_normalized);
    for (int i=0;i<sample_normalized.rows;i++){
        sample_normalized.row(i)=(sample_normalized.row(i) / 128.0);
    }

    cout<<CV_VERSION<<endl;

    std::vector<asr::Blob<float>* > output = classifier.Forward(sample_normalized);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}


