//
// Created by jbk on 19-3-22.
//

#include <vector>
#include "layers/input_layer.hpp"


using namespace std;

namespace asr{

    /*change by hua*/
    template<typename DType>
    void InputLayer<DType>::SetUp(const LayerParameter& param, const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top)
    {

        cout << "InputLayer::SetUp() " << param.name() << endl;

        const int num_top = top.size();
        const InputParameter& input_param = param.input_param();
        const int num_shape = input_param.shape_size();
        CHECK(num_shape == 0 || num_shape == 1 || num_shape == num_top)
                        << "Must specify 'shape' once, once per top blob, or not at all: "
                        << num_top << " tops vs. " << num_shape << " shapes.";
        if (num_shape > 0) {
            for (int i = 0; i < num_top; ++i) {
                const int shape_index = (input_param.shape_size() == 1) ? 0 : i;
                // top[i] = new Blob<double>(input_param.shape(shape_index);
                top[i]->Reshape(input_param.shape(shape_index));
            }
        }

        return;
    }

    template<typename DType>
    void InputLayer<DType>::Forward(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top)
    {
        cout << "InputLayer::forward()..." << endl;

        return;
    }

    template<typename DType>
    void InputLayer<DType>::calc_shape_(const vector<int>& in_shape, vector<int>& out_shape)
    {
        cout << "InputLayer::calc_shape()..." << endl;

        return;
    }

    INSTANTIATE_CLASS(InputLayer);
}

