//
// Created by jbk on 19-3-14.
//
#include <algorithm>
#include <cfloat>
#include <vector>
#include <layers/pooling_layer.hpp>
#include <blob_.hpp>
#include <glog/logging.h>


using namespace std;
namespace caffe{
    void PoolLayer::SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)

    {
        cout << "PoolLayer::SetUp()" << param.name() << endl;
        // 分配权重空间
        if (param.blobs_size() > 0) {
            weights().resize(param.blobs_size());
            for (int i = 0; i < param.blobs_size(); ++i) {
                weights()[i].reset(new Blob<double>());
//                weights()[i]->FromProto(param.blobs(i));
            }
        }

        pad_.resize(2);
        kernel_.resize(2);
        stride_.resize(2);

        PoolingParameter pool_param = param.pooling_param();
        if (pool_param.global_pooling()) {
            CHECK(!(pool_param.has_kernel_size() ||
                    pool_param.has_kernel_h() || pool_param.has_kernel_w()))
                    << "With Global_pooling: true Filter size cannot specified";
        } else {
            CHECK(!pool_param.has_kernel_size() !=
                  !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
                    << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
            CHECK(pool_param.has_kernel_size() ||
                  (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
                    << "For non-square filters both kernel_h and kernel_w are required.";
        }
        CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
               && pool_param.has_pad_w())
              || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
                << "pad is pad OR pad_h and pad_w are required.";
        CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
               && pool_param.has_stride_w())
              || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
                << "Stride is stride OR stride_h and stride_w are required.";
        global_pooling_ = pool_param.global_pooling();
        if (global_pooling_) {
            kernel_[0] = bottom[0]->height();
            kernel_[1] = bottom[0]->width();
        }
        else {
            if (pool_param.has_kernel_size()) {
                kernel_[0] = kernel_[1] = pool_param.kernel_size();
            } else {
                kernel_[0] = pool_param.kernel_h();
                kernel_[1] = pool_param.kernel_w();
            }
        }
        CHECK_GT(kernel_[0], 0) << "Filter dimensions cannot be zero.";
        CHECK_GT(kernel_[1], 0) << "Filter dimensions cannot be zero.";
        if (!pool_param.has_pad_h()) {
            pad_[0] = pad_[1] = pool_param.pad();
        } else {
            pad_[0] = pool_param.pad_h();
            pad_[1] = pool_param.pad_w();
        }
        if (!pool_param.has_stride_h()) {
            stride_[0] = stride_[1] = pool_param.stride();
        } else {
            stride_[0] = pool_param.stride_h();
            stride_[1] = pool_param.stride_w();
        }
        if (global_pooling_) {
            CHECK(pad_[0] == 0 && pad_[1] == 0 && stride_[0] == 1 && stride_[1] == 1)
                    << "With Global_pooling: true; only pad = 0 and stride = 1";
        }

        CHECK(param.pooling_param().pool() == PoolingParameter_PoolMethod_AVE
              || param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX)
                        << "only for average and max pooling.";

        if (param.pooling_param().pool() == PoolingParameter_PoolMethod_AVE){
            pool_methods_ = PoolMethod_AVE;
        }else{
            pool_methods_ = PoolMethod_MAX;
        }

        CHECK(param.pooling_param().round_mode() == PoolingParameter_RoundMode_CEIL
        || param.pooling_param().round_mode() == PoolingParameter_RoundMode_FLOOR)
                       <<"only for floor and ceil mode pooling.";

        if (param.pooling_param().round_mode() == PoolingParameter_RoundMode_CEIL){
            pool_round_mode_ = PoolRoundMode_CEIL;
        }else{
            pool_round_mode_ = PoolRoundMode_FLOOR;
        }

        if (pad_[0] != 0 || pad_[1] != 0) {
            CHECK_LT(pad_[0], kernel_[0]);
            CHECK_LT(pad_[1], kernel_[1]);
        }


        CHECK_EQ(1, bottom.size()) << "bottom size must be 1 ";
        CHECK_EQ(1, top.size()) << "top size must be 1 ";
        CHECK_EQ(4, bottom[0]->shape().size()) << "Input must have 4 axes, "
                                           << "corresponding to (num, channels, height, width)";

        in_shape_.push_back(bottom[0]->num());
        in_shape_.push_back(bottom[0]->channels());
        in_shape_.push_back(bottom[0]->height());
        in_shape_.push_back(bottom[0]->width());

        calc_shape_(in_shape_, out_shape_);
        top[0]->Reshape(out_shape_);

        cout << "top.shape:" << "\t";
        PrintVector(out_shape_);

        return;
    }


    void PoolLayer::Forward(const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top)

    {

        cout << "PoolLayer::forward()..." << endl;

        switch (pool_methods_) {
            case PoolMethod_MAX:
                // The main loop
                for (int ph = 0; ph < out_shape_[2]; ++ph) {
                    for (int pw = 0; pw < out_shape_[3]; ++pw) {
                        int hstart = ph * stride_[0] - pad_[0];
                        int wstart = pw * stride_[1] - pad_[1];
                        int hend = min(hstart + kernel_[0], in_shape_[2]);
                        int wend = min(wstart + kernel_[1], in_shape_[3]);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        top[0]->sub_blob(vector<vector<int>>{{},{},{ph},{pw}}) = bottom[0]->sub_blob(vector<vector<int>>{{},{},{hstart, hend},{wstart, wend}}).max();
                    }
                }
                break;
            case PoolMethod_AVE:{
                // The main loop
                for (int ph = 0; ph < out_shape_[2]; ++ph) {
                    for (int pw = 0; pw < out_shape_[3]; ++pw) {
                        int hstart = ph * stride_[0] - pad_[0];
                        int wstart = pw * stride_[1] - pad_[1];
                        int hend = min(hstart + kernel_[0], in_shape_[2]);
                        int wend = min(wstart + kernel_[1], in_shape_[3]);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        top[0]->sub_blob(vector<vector<int>>{{},{},{ph},{pw}}) = bottom[0]->sub_blob(vector<vector<int>>{{},{},{hstart, hend},{wstart, wend}}).ave();
                    }
                }
                break;
            }
        }
        return;
    }

    void PoolLayer::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "PoolLayer::calc_shape()..." << endl;
        int Ni = in_shape[0];
        int Ci = in_shape[1];
        int Hi = in_shape[2];
        int Wi = in_shape[3];

        int No = Ni;
        int Co = Ci;
        int Ho = 1;
        int Wo = 1;

        switch (pool_round_mode_) {
            case PoolRoundMode_CEIL:{
                Ho = static_cast<int>(ceil(static_cast<float>(Hi + 2 * pad_[0] - kernel_[0]) / stride_[0])) + 1;
                Wo = static_cast<int>(ceil(static_cast<float>(Wi + 2 * pad_[1] - kernel_[1]) / stride_[1])) + 1;
                break;
            }
            case PoolRoundMode_FLOOR:{
                Ho = static_cast<int>(floor(static_cast<float>(Hi + 2 * pad_[0] - kernel_[0]) / stride_[0])) + 1;
                Wo = static_cast<int>(floor(static_cast<float>(Wi + 2 * pad_[1] - kernel_[1]) / stride_[1])) + 1;
                break;
            }
        }


        if (pad_[0] || pad_[1]) {
            // If we have padding, ensure that the last pooling starts strictly
            // inside the image (instead of at the padding); otherwise clip the last.
            if ((Ho - 1) * stride_[0] >= Hi + pad_[0]) {
                --Ho;
            }
            if ((Wo - 1) * stride_[1] >= Wi + pad_[1]) {
                --Wo;
            }
            CHECK_LT((Ho - 1) * stride_[0], Ho + pad_[0]);
            CHECK_LT((Wo - 1) * stride_[1], Wo + pad_[1]);
        }

        // resize(4) ??
        out_shape.push_back(No);
        out_shape.push_back(Co);
        out_shape.push_back(Ho);
        out_shape.push_back(Wo);

        return;
    }

}


