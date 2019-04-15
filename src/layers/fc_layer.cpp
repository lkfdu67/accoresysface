//
// Created by jbk on 19-3-14.
//
#include <vector>
#include <layers/fc_layer.hpp>
#include <blob_.hpp>


using namespace std;

namespace asr{
    template<typename DType>
    void FCLayer<DType>::SetUp(const LayerParameter& param, const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top)
    {
        cout << "FCLayer::SetUp()" << param.name() << endl;

        nums_out_ = param.inner_product_param().num_output();
        bias_term_ = param.inner_product_param().bias_term();
        transpose_ = param.inner_product_param().transpose();

        in_shape_.push_back(bottom[0]->num());
        in_shape_.push_back(bottom[0]->channels());
        in_shape_.push_back(bottom[0]->height());
        in_shape_.push_back(bottom[0]->width());
        // Check if we need to set up the weights
        if (this->weights().size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            if (bias_term_) {
                this->weights().resize(2);
            } else {
                this->weights().resize(1);
            }
            // Initialize the weights
            vector<int> weight_shape(4);
            weight_shape[0] = nums_out_;
            weight_shape[1] = in_shape_[1];
            weight_shape[2] = in_shape_[2];
            weight_shape[3] = in_shape_[3];
//            if (transpose_) {
//                weight_shape[0] = in_shape_[1];
//                weight_shape[1] = nums_out_;
//            } else {
//                weight_shape[0] = nums_out_;
//                weight_shape[1] = in_shape_[1];
//            }
            this->weights()[0].reset(new Blob<DType>(weight_shape));

            // If necessary, initialize and fill the bias term
            if (bias_term_) {
                this->weights()[1].reset(new Blob<DType>(1, nums_out_, 1, 1));
            }
        }

        calc_shape_(in_shape_, out_shape_);
        top[0]->Reshape(out_shape_);

        cout << "top.shape:" << "\t";
        this->PrintVector(out_shape_);
        if (this->weights().size() > 0){
            cout << "weight.shape:" << "\t";
            this->PrintVector(this->weights()[0]->shape());
        }
        if (this->weights().size() > 1) {

            cout << "bias.shape:" << "\t";
            this->PrintVector(this->weights()[1]->shape());
        }
        return;
    }

    template<typename DType>
    void FCLayer<DType>::Forward(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top)
    {
        cout << "FCLayer::forward()..." << endl;
        for (int n = 0; n < out_shape_[0]; ++n) {
            for (int c = 0; c < out_shape_[1]; ++c) {
//                Blob<DType> tmp_blob = bottom[0]->sub_blob(vector<vector<int>>{{n},{},{},{}}) * weights()[0]->sub_blob(vector<vector<int>>{{c},{},{},{}});
//                DType tmp = tmp_blob.sum_all_channel()[0];
                DType tmp = (bottom[0]->sub_blob(vector<vector<int>>{{n},{},{},{}}) *
                        this->weights()[0]->sub_blob(vector<vector<int>>{{c},{},{},{}})).sum_all_channel()[0];
                if (bias_term_) {
                    tmp += this->weights()[1]->at(0,c,0,0);
                }

                top[0]->at(n,c,0,0) = tmp;
            }
        }
        return;
    }
    template<typename DType>
    void FCLayer<DType>::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "FCLayer::calc_shape()..." << endl;

        int Ni = in_shape[0];
        int Ci = in_shape[1];
        int Hi = in_shape[2];
        int Wi = in_shape[3];

        int No = Ni;
        int Co = nums_out_;
        int Ho = 1;
        int Wo = 1;

        // resize(4) ??
        out_shape.push_back(No);
        out_shape.push_back(Co);
        out_shape.push_back(Ho);
        out_shape.push_back(Wo);

        return;
    }
    template<typename DType>
    void FCLayer<DType>::Reshape(const vector<asr::Blob<DType> *> &bottom, vector<asr::Blob<DType> *> &top) {
        return;
    }
    INSTANTIATE_CLASS(FCLayer);

}
