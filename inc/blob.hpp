//
// Created by hua on 19-3-14.
//

#ifndef LOADPARAM_BLOB_HPP
#define LOADPARAM_BLOB_HPP
/*
#include <armadillo>
using namespace arma;
using namespace std;


namespace caffe{

class Blob{
public:
	Blob(): data_(), shape_() {}

	Blob(const int num, const int channels, const int height, const int width);

	explicit Blob(const vector<int>& shape);

	explicit Blob(const vector<cube>& cubes);

	Blob(const Blob&);

	Blob& operator=(const Blob&);

	virtual ~Blob() {}


	///
	inline const vector<int>& shape() const {
		return shape_;
	}

	inline int shape(int index) const {
		if (index < 0 || index >= 4) {
			return 1;
		}
		return shape_[index];
	}

	inline int num() const {
		return data_.size();
	}

	inline int channels() const {
		return data_.size() > 0 ? data_[0].n_slices : 0;
	}

	inline int height() const {
		return data_.size() > 0 ? data_[0].n_rows : 0;
	}

	inline int width() const {
		return data_.size() > 0 ? data_[0].n_cols : 0;
	}

	inline const vector<cube>& data() const {
		return data_;
	}

	void print_data() const;


	///
	Blob sub_blob(const int channel_start) const;

	Blob sub_blob(const int channel_start, const int channel_end) const;

	Blob sub_blob(const int channel_start, const int height_start, const int width_start) const;

	Blob sub_blob(const int channel_start, const int channel_end, const int height_start, 
		const int height_end, const int width_start, const int width_end) const;

	Blob operator()(vector<int> channel) const;

	Blob operator()(vector<int> channel, vector<int> height, vector<int> width) const;

	vector<double> operator[](const int index) const;


	///
	vector<double> sum() const;

	void scale(const double scale_factor);

	Blob operator+(const Blob& other) const;

	Blob& operator+=(const Blob& other);

	Blob operator*(const Blob& other) const;

	Blob& operator*=(const Blob& other);

	//friend Blob& operator*(const Blob& other, const double scale_factor);

	Blob operator*(const double scale_factor) const;

	Blob& operator*=(const double scale_factor);

	template<typename functor>
	Blob& elem_wise_op(functor lambda_function);

private:
	//int num_;
	//int channels_;
	//int height_;
	//int width_;
	vector<cube> data_;
	vector<int> shape_;

};	//class Blob

}	//namespace caffe

*/

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <sstream>
#include <string>
#include <vector>
#include "caffe.pb.h"

namespace caffe{

class Blob{
public:
	/**
   * @brief 判断other与本地Blob形状是否相同
   * params: BlobProto caffe.proto定义的参数类型
   * return： 维度相同返回true, 否则返回false
   */
	bool ShapeEquals(const BlobProto& other);

	/**
   * @brief 由BlobProto（序列化为proto的blob，解析成BlobProto变量）对Blob进行赋值操作。
   * reshape代表是否允许修改shape_的大小。
   */
	void FromProto(const BlobProto& proto, bool reshape = true);

	/// @brief 将shape_转成字符串，以便于打印
	inline std::string shape_string() const {
		std::ostringstream stream;
		for (int i = 0; i < shape_.size(); ++i) {
			stream << shape_[i] << " ";
		}
		return stream.str();
	}
private:
	/// @brief data的维度: [0,1,2,3]-->[n, c, h, w]
	std::vector<int> shape_;

};

}


#endif //LOADPARAM_BLOB_HPP
