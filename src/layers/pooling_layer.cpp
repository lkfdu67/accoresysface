//
// Created by jbk on 19-3-14.
//
#include <algorithm>
#include <cfloat>
#include <vector>
#include "layers/pooling_layer.hpp"
#include "blob.hpp"
#include "glog/logging.h"


using namespace std;
namespace caffe{
    void PoolLayer::SetUp(const LayerParameter& param, const vector<Blob*>& bottom, vector<Blob*>& top)
    {
        cout << "PoolLayer::SetUp()" << param.name() << endl;
        // 分配权重空间
        if (param.blobs_size() > 0) {
            blobs().resize(param.blobs_size());
            for (int i = 0; i < param.blobs_size(); ++i) {
                blobs()[i].reset(new Blob());
                blobs()[i]->FromProto(param.blobs(i));
            }
        }


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
            kernel_h_ = bottom[0]->height();
            kernel_w_= bottom[0]->width();
        }
        else {
            if (pool_param.has_kernel_size()) {
                kernel_h_ = kernel_w_ = pool_param.kernel_size();
            } else {
                kernel_h_ = pool_param.kernel_h();
                kernel_w_ = pool_param.kernel_w();
            }
        }
        CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
        CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
        if (!pool_param.has_pad_h()) {
            pad_h_ = pad_w_ = pool_param.pad();
        } else {
            pad_h_ = pool_param.pad_h();
            pad_w_ = pool_param.pad_w();
        }
        if (!pool_param.has_stride_h()) {
            stride_h_ = stride_w_ = pool_param.stride();
        } else {
            stride_h_ = pool_param.stride_h();
            stride_w_ = pool_param.stride_w();
        }
        if (global_pooling_) {
            CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
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

        if (pad_h_ != 0 || pad_w_ != 0) {
            CHECK_LT(pad_h_, kernel_h_);
            CHECK_LT(pad_w_, kernel_w_);
        }


        CHECK_EQ(1, bottom.size()) << "bottom size must be 1 ";
        CHECK_EQ(1, top.size()) << "top size must be 1 ";
        CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
                                           << "corresponding to (num, channels, height, width)";

        in_shape_.push_back(bottom[0]->num());
        in_shape_.push_back(bottom[0]->channels());
        in_shape_.push_back(bottom[0]->height());
        in_shape_.push_back(bottom[0]->width());

        calc_shape_(in_shape_, out_shape_);
        top[0]->Reshape(out_shape_);

        return;
    }



    void PoolLayer::Forward(const vector<Blob*>& bottom, vector<Blob*>& top)
    {

        cout << "PoolLayer::forward()..." << endl;

        switch (pool_methods_) {
            case PoolMethod_MAX:
                // The main loop
                for (int n = 0; n < out_shape_[0]; ++n) {
                    for (int c = 0; c < out_shape_[1]; ++c) {
                        for (int ph = 0; ph < out_shape_[2]; ++ph) {
                            for (int pw = 0; pw < out_shape_[3]; ++pw) {
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;
                                int hend = min(hstart + kernel_h_, in_shape_[2]);
                                int wend = min(wstart + kernel_w_, in_shape_[3]);
                                hstart = max(hstart, 0);
                                wstart = max(wstart, 0);

                                // ????

                            }
                        }
                    }
                }
                break;
            case PoolMethod_AVE:{
                // The main loop
                for (int n = 0; n < out_shape_[0]; ++n) {
                    for (int c = 0; c < out_shape_[1]; ++c) {
                        for (int ph = 0; ph < out_shape_[2]; ++ph) {
                            for (int pw = 0; pw < out_shape_[3]; ++pw) {
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;
                                int hend = min(hstart + kernel_h_, in_shape_[2] + pad_h_);
                                int wend = min(wstart + kernel_w_, in_shape_[3] + pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = max(hstart, 0);
                                wstart = max(wstart, 0);
                                hend = min(hend, in_shape_[2]);
                                wend = min(wend, in_shape_[3]);


                                // ???

                            }
                        }
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

        int Ho = static_cast<int>(floor(static_cast<float>(Hi + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
        int Wo = static_cast<int>(floor(static_cast<float>(Wi + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;

        if (pad_h_ || pad_w_) {
            // If we have padding, ensure that the last pooling starts strictly
            // inside the image (instead of at the padding); otherwise clip the last.
            if ((Ho - 1) * stride_h_ >= Hi + pad_h_) {
                --Ho;
            }
            if ((Wo - 1) * stride_w_ >= Wi + pad_w_) {
                --Wo;
            }
            CHECK_LT((Ho - 1) * stride_h_, Ho + pad_h_);
            CHECK_LT((Wo - 1) * stride_w_, Wo + pad_w_);
        }

        // resize(4) ??
        out_shape.push_back(No);
        out_shape.push_back(Co);
        out_shape.push_back(Ho);
        out_shape.push_back(Wo);

        return;
    }

}


