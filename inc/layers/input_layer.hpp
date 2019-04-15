//
// Created by jbk on 19-3-22.
//

#ifndef LOADPARAM_INPUT_LAYER_HPP
#define LOADPARAM_INPUT_LAYER_HPP

#include <layer.hpp>

using std::vector;
using std::shared_ptr;
using std::string;
using std::pair;

namespace asr{

template<typename DType>
class InputLayer : public Layer<DType>
{
public:
    InputLayer(){}
    ~InputLayer(){}
    void SetUp(const LayerParameter& param,  const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);
    void Forward(const vector<Blob<DType>* >& bottom, vector<Blob<DType>* >& top);
    // Data layers have no bottoms, so reshaping is trivial.
    virtual void Reshape(const vector<Blob<DType> *>& bottom,
            vector<Blob<DType>*>& top) {}

private:
    vector<int> in_shape_;
    vector<int> out_shape_;
    void calc_shape_(const vector<int>& in_shape, vector<int>& out_shape);
};

}

#endif //LOADPARAM_INPUT_LAYER_HPP
