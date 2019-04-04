//
// Created by jbk on 19-3-14.
//
#include <vector>
#include <layers/batch_norm_layer.hpp>


using namespace std;

namespace caffe{

    void BNLayer::SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "BNLayer::SetUp()" << param.name() << endl;

        in_shape_.push_back(bottom[0]->num());
        in_shape_.push_back(bottom[0]->channels());
        in_shape_.push_back(bottom[0]->height());
        in_shape_.push_back(bottom[0]->width());
        
        if (bottom[0]->shape().size() == 1)
            channels_ = 1;
        else
            channels_ = bottom[0]->shape(1);
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
            this->weights()[0].reset(new Blob<double>(sz));
            this->weights()[1].reset(new Blob<double>(sz));
            sz[1] = 1;
            this->weights()[2].reset(new Blob<double>(sz));
        }

        calc_shape_(in_shape_, out_shape_);
        top[0]->Reshape(out_shape_);
        return;
    }

    void BNLayer::Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)
    {
        cout << "BNLayer::forward()..." << endl;
        return;
    }

    void BNLayer::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "BNLayer::calc_shape()..." << endl;
        int Ni = in_shape[0];
        int Ci = in_shape[1];
        int Hi = in_shape[2];
        int Wi = in_shape[3];


        // resize(4) ??
        out_shape.push_back(Ni);
        out_shape.push_back(Ci);
        out_shape.push_back(Hi);
        out_shape.push_back(Wi);
        return;
    }
}