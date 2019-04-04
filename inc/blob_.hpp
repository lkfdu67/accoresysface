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

#define SUB_BLOB_FORMAT(n_beg,n_end,c_beg,c_end,h_beg,h_end,w_beg,w_end)\
	to_string(n_beg)+":"+to_string(n_end)+";"+\
	to_string(c_beg)+":"+to_string(c_end)+";"+\
	to_string(h_beg)+":"+to_string(h_end)+";"+\
	to_string(w_beg)+":"+to_string(w_end)

namespace caffe{

template<typename DType>
class Blob{
public:
	Blob(): data_(), shape_() {}

	Blob(const int num, const int channels, const int height, const int width);

	explicit Blob(const vector<int>& shape);

	explicit Blob(const vector<Cube<DType>>& cubes);

	explicit Blob(const BlobShape& shape);

	explicit Blob(const BlobProto& proto);

	Blob(const Blob&);

	virtual ~Blob() {}

	//usage: b2 = b1, b3 = b2 = b1
	Blob& operator=(const Blob&);

	void FromProto(const BlobProto& proto, bool reshape = true);

	

	//usage: b1.Reshape(shape)
	Blob& Reshape(const vector<int>& shape);

	//usage: b1.Reshape(shape)
	Blob& Reshape(const BlobShape& shape);
	
	inline bool ShapeEquals(const Blob& rhs) const {
		return this->shape_ == rhs.shape_;
	}

	//usage: b1.ShapeEquals(proto)
	bool ShapeEquals(const BlobProto& proto) const;

	//usage: b1.shape_string()
	string shape_string() const;

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

	//usage: b1.size()
	vector<int> size() const; 



	inline const vector<Cube<DType>>& data() const {
		return this->data_;
	}

	//usage: b1.print_data()
	void print_data() const;

	//usage: b1.save_data("blob.txt")
	void save_data(const string& txt_path) const;

	//usage: b2 = b1.load_data("blob.txt", 2, 64, 112, 112)
	Blob load_data(const string& txt_path, const int num, const int channel,
		const int height, const int width) const;



	//usage: b1(0,63,3,4)
	DType operator()(const int num, const int channel, const int height, const int width) const;

	//usage: b1(vector<int>{0,63,3,4})
	DType operator()(const vector<int>& shape) const;

	//usage: b1.at(0,63,3,4) = 100.0
	DType& at(const int num, const int channel, const int height, const int width);

	//usage: b1.at(vector<int>{0,63,3,4}) = 100.0
	DType& at(const vector<int>& shape);

	//usage: b1[1000]
	vector<DType> operator[](const int index) const;



	//usage: b1.sum_all_channel()
	vector<DType> sum_all_channel() const;

	//usage: b2 = b1.sum()
	Blob sum() const;

	//usage: b1.ave_all_channel()
	vector<DType> ave_all_channel() const;

	//usage: b2 = b1.ave()
	Blob ave() const;

	//usage: b1.max_all_channel()
	vector<DType> max_all_channel() const;

	//usage: b2 = b1.max()
	Blob max() const;

	//usage: b2 = b1.exp()
	Blob exp() const;

	//usage: b1.exp_inplace()
	Blob& exp_inplace();



	//usage: b2 = b1.sub_blob("0:2;10:63;56:112;56:112")	
	//usage: b2 = b1.sub_blob(":;10:63;:;:")
	//usage: b2 = b1.sub_blob(SUB_BLOB_FORMAT(n1,n2,c1,c2,h1,h2,w1,w2))
	//@param: format, indicate the start index to(:) end index separated by semicolon for num,channel,height and weight
	Blob sub_blob(const string& format) const;

	//usage: b1.sub_blob(vector<vector<int>>{{},{0,63},{0,112},{0,112}})
	//usage: b1.sub_blob(vector<vector<int>>{{0},{10},{56,112},{}})
	Blob sub_blob(const vector<vector<int>>& nchw) const;

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
