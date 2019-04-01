//
// Created by hua on 19-3-14.
//

#ifndef LOADPARAM_BLOB_HPP
#define LOADPARAM_BLOB_HPP

#include "caffe.pb.h"
#include <armadillo>
#include <gflags/gflags.h>
#include <glog/logging.h>
using namespace arma;
using namespace std;


namespace caffe{

template<typename DType>
class Blob{
public:
	Blob(): data_(), shape_() {}

	Blob(const int num, const int channels, const int height, const int width);

	explicit Blob(const vector<int>& shape);

	explicit Blob(const vector<Cube<DType>>& cubes);

	explicit Blob(const BlobProto& proto);

	Blob(const Blob&);

	//usage: b2 = b1, b3 = b2 = b1
	Blob& operator=(const Blob&);

	virtual ~Blob() {}


	
	inline const vector<int>& shape() const {
		return this->shape_;
	}

	inline int shape(int index) const {		
		if (index < 0 || index >= 4) {
			return 1;
		}
		return this->shape_[index];
	}

	inline int num() const {
		CHECK_EQ(shape_.size(), 4);
		return shape(0);
	}

	inline int channels() const {
		CHECK_EQ(shape_.size(), 4);
		return shape(1);
	}

	inline int height() const {
		CHECK_EQ(shape_.size(), 4);
		return shape(2);
	}

	inline int width() const {
		CHECK_EQ(shape_.size(), 4);
		return shape(3);
	}

	inline bool is_shape_equal(const Blob& rhs) const {
		return this->shape_ == rhs.shape_;
	}

	inline const vector<Cube<DType>>& data() const {
		return this->data_;
	}

	//usage: b1.shape_string()
	string shape_string() const;

	//usage: b1.size()
	vector<int> size() const; 

	//usage: b1.print_data()
	void print_data() const;

	//usage: b1.save_data("blob.txt")
	void save_data(const string& txt_path) const;

	//usage: b2 = b1.load_data("blob.txt", 2, 64, 112, 112)
	Blob load_data(const string& txt_path, const int num, const int channel,
		const int height, const int width) const;



	//usage: b2 = b1.sub_blob(10)
	Blob sub_blob(const int channel_start) const;

	//usage: b2 = b1.sub_blob(10, 63)
	Blob sub_blob(const int channel_start, const int channel_end) const;

	//usage: b2 = b1.sub_blob(10, 0, 0)
	Blob sub_blob(const int channel_start, const int height_start, const int width_start) const;

	//usage: b2 = b1.sub_blob(10, 63, 0, 112, 0, 112)
	Blob sub_blob(const int channel_start, const int channel_end, const int height_start,
		const int height_end, const int width_start, const int width_end) const;

	//usage: b2 = b1.sub_blob(vector<int>{0,1,2,3,5,8})
	Blob sub_blob(const vector<int>& channel) const;

	//usage: b2 = b1.sub_blob(vector<int>{10,63}, vector<int>{0,112}, vector<int>{0,112})
	Blob sub_blob(const vector<int>& channel, const vector<int>& height, const vector<int>& width) const;

	//usage: b1(0,63,3,4)
	DType operator()(const int num, const int channel, const int height, const int width);

	//usage: b1(vector<int>{0,63,3,4})
	DType operator()(const vector<int>& shape);

	//usage: b1[1000]
	vector<DType> operator[](const int index) const;



	//usage: b1.sum()
	vector<DType> sum() const;

	//usage: b1.ave()
	vector<DType> ave() const;

	//usage: b1.max()
	vector<DType> max() const;

	//usage: b1.min()
	vector<DType> min() const;

	//usage: b2 = b1.exp()
	Blob exp() const;

	//usage: b1.exp_inplace()
	Blob& exp_inplace();

	//usage: b2 = b1.log()
	Blob log() const;

	//usage: b1.log_inplace()
	Blob& log_inplace();



	//usage: b3 = b2.join(b1)
	Blob join(const Blob& rhs) const;

	//usage: b2.join_inplace(b1)
	Blob& join_inplace(const Blob& rhs);

	//usage: b3 = b2 + b1
	Blob operator+(const Blob& rhs) const;

	//usage: b2 += b1
	Blob& operator+=(const Blob& rhs);

	//usage: b3 = b2 - b1
	Blob operator-(const Blob& rhs) const;

	//usage: b2 -= b1
	Blob& operator-=(const Blob& rhs);

	//usage: b3 = b2 * b1
	Blob operator*(const Blob& rhs) const;

	//usage: b2 *= b1
	Blob& operator*=(const Blob& rhs);

	//usage: b3 = b2.mat_mul(b1)
	Blob mat_mul(const Blob& rhs) const;

	//usage: b2.mat_mul(b1)
	Blob& mat_mul_inplace(const Blob& rhs);

	//usage: b3 = b2 / b1
	Blob operator/(const Blob& rhs) const;

	//usage: b2 /= b1
	Blob& operator/=(const Blob& rhs);

	//usage: b2 = 10.0 * b1
	//friend Blob<DType> operator*(const DType scale_factor, const Blob<DType>& other);
	template<typename Ty>
	inline friend Blob operator*(const Ty scale_factor, const Blob<Ty>& other) {
		Blob<Ty> b;
		for (auto d : other.data_) {
			b.data_.push_back(scale_factor * d);
		}
		b.shape_ = other.shape_;
		return b;
	}

	//usage: b2 = b1 * 10.0
	Blob operator*(const DType scale_factor) const;

	//usage: b1 *= 10.0
	Blob& operator*=(const DType scale_factor);

	//usage: b1.scale(10.0)
	void scale(const DType scale_factor);

	//usage: b2 = b1.elem_wise([](float val) {return val * 100.0f; })
	//template<typename functor>
	//Blob elem_wise(functor const &lambda_function) const;
	template<typename functor>
	inline Blob elem_wise(functor const &lambda_function) const {
		Blob<DType> b;
		for (auto d : this->data_) {
			b.data_.push_back(d.transform(lambda_function));
		}
		b.shape_ = this->shape_;
		return b;
	}

	//usage: b1.elem_wise_inplace([](float val) {return val * 100.0f; })
	//template<typename functor>
	//Blob& elem_wise_inplace(functor const &lambda_function)
	template<typename functor>
	inline Blob& elem_wise_inplace(functor const &lambda_function) {
		for (auto &d : this->data_) {
			d.transform(lambda_function);
		}
		return *this;
	}

private:
	vector<Cube<DType>> data_;
	vector<int> shape_;
};	//class Blob

}	//namespace caffe

#endif //LOADPARAM_BLOB_HPP
