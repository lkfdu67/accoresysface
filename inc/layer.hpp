//
// Created by hua on 19-3-14.
//

#ifndef LOADPARAM_LAYER_HPP
#define LOADPARAM_LAYER_HPP

#include <caffe.pb.h>
#include <utility>
#include <blob_.hpp>

using std::pair;
using std::vector;
using std::shared_ptr;
using std::string;

namespace caffe{

class Layer{
public:
    explicit Layer(){}
    virtual ~Layer(){}
    virtual void SetUp(const LayerParameter& param, const vector<Blob<double>* >& bottom, vector<Blob<double>* >& top) = 0;
    virtual void Forward(const vector<Blob<double>* >&, vector<Blob<double>* >& top) = 0;
    /**
     * @brief Returns the vector of learnable parameter blobs.
    */
    vector<shared_ptr<Blob<double> > >& blobs() {
        return blobs_;
    }

private:
    /** The vector that stores the learnable parameters as a set of blobs. */
    vector<shared_ptr<Blob<double> > > blobs_;
};
}

#endif //LOADPARAM_LAYER_HPP
