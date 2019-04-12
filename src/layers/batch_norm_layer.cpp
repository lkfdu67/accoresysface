//
// Created by jbk on 19-3-14.
//
#include <vector>
#include <layers/batch_norm_layer.hpp>
#include <math.h>


using namespace std;

namespace asr{

    void BNLayer::SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "BNLayer::SetUp()" << param.name() << endl;

//        in_shape_.push_back(bottom[0]->num());
//        in_shape_.push_back(bottom[0]->channels());
//        in_shape_.push_back(bottom[0]->height());
//        in_shape_.push_back(bottom[0]->width());
        in_shape_ = bottom[0]->shape();
        out_shape_ = in_shape_;
        top[0]->Reshape(out_shape_);


        channels_ = bottom[0]->shape(1);
        eps_ = param.batch_norm_param().eps();  // 防止方差为０的偏移值

        if (param.batch_norm_param().has_use_global_stats()){
            CHECK(!param.batch_norm_param().use_global_stats())
                            << "use_global_stats must be 1.";
        }  //　确保只能使用预定义的方差和均值

        if (this->weights().size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            this->weights().resize(3);
            vector<int> sz;
            sz.resize(4);
            sz[0] = 1;
            sz[2] = 1;
            sz[3] = 1;
            sz[1] = channels_;
            this->weights()[0].reset(new Blob<double>(sz));  // mean
            this->weights()[1].reset(new Blob<double>(sz));  // var
            sz[1] = 1;
            this->weights()[2].reset(new Blob<double>(sz));  // 滑动平均系数
        }



        cout << "top.shape:" << "\t";
        PrintVector(out_shape_);
        if (this->weights().size() > 0){
            cout << "mean.shape:" << "\t";
            PrintVector(weights()[0]->shape());
        }
        if (this->weights().size() > 1) {

            cout << "var.shape:" << "\t";
            PrintVector(weights()[1]->shape());
        }

        return;
    }

    void BNLayer::Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "BNLayer::forward()..." << endl;

        Blob<double> mean = (*(this->weights()[0])) * (*(this->weights()[2]))(0, 0, 0, 0);
        Blob<double> var = (*(this->weights()[1])) * (*(this->weights()[2]))(0, 0, 0, 0);
        var += this->eps_;
        var.elem_wise_inplace([](double val) {return sqrt(val); });

        for (int c = 0; c < out_shape_[1]; ++c) {
            Blob<double> tmp_blob = bottom[0]->sub_blob(vector<vector<int>>{{},{c},{},{}})
                     + (*(this->weights()[0]))(0, c, 0, 0);
            tmp_blob /= (*(this->weights()[1]))(0, c, 0, 0);
            for (int ph = 0; ph < out_shape_[2]; ++ph) {
                for (int pw = 0; pw < out_shape_[3]; ++pw) {
                    for (int n = 0; n < out_shape_[0]; ++n) {
                            (*top[0]).at(n, c, ph, pw) = tmp_blob(n, c, ph, pw);
                    }
                }
            }
        }
        return;
    }

//    void BNLayer::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
//    {
//        cout << "BNLayer::calc_shape()..." << endl;
//        int Ni = in_shape[0];
//        int Ci = in_shape[1];
//        int Hi = in_shape[2];
//        int Wi = in_shape[3];
//
//
//        out_shape.push_back(Ni);
//        out_shape.push_back(Ci);
//        out_shape.push_back(Hi);
//        out_shape.push_back(Wi);
//        return;
//    }


    void BNLayer::Reshape(const vector<asr::Blob<double> *> & bottom, vector<asr::Blob<double> *> &top) {
        in_shape_ = bottom[0]->shape();
        out_shape_ = in_shape_;
        top[0]->Reshape(out_shape_);

    }
}