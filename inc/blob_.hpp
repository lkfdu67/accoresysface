//
// Created by hua on 19-3-14.
//

#ifndef LOADPARAM_BLOB_HPP
#define LOADPARAM_BLOB_HPP

#include "caffe.pb.h"
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

	explicit Blob(const BlobProto& proto);

	Blob(const Blob&);

	//usage: b2 = b1, b3 = b2 = b1
	Blob& operator=(const Blob&);

	virtual ~Blob() {}


	
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
		_ASSERT(shape_.size() == 4);
		return shape(0);
	}

	inline int channels() const {
		_ASSERT(shape_.size() == 4);
		return shape(1);
	}

	inline int height() const {
		_ASSERT(shape_.size() == 4);
		return shape(2);
	}

	inline int width() const {
		_ASSERT(shape_.size() == 4);
		return shape(3);
	}

	inline bool is_shape_equal(const Blob& rhs) const {
		return this->shape_ == rhs.shape_;
	}

	inline const vector<cube>& data() const {
		return data_;
	}

	//usage: b1.shape_string()
	string shape_string() const;

	//usage: b1.size()
	vector<int> size() const; 

	//usage: b1.print_data()
	void print_data() const;

	//usage: b1.save_data("blob.txt")
	void save_data(const std::string txt_path) const;

	//usage: b2 = b1.load_data("blob.txt", 2, 64, 112, 112)
	Blob load_data(const std::string txt_path, const int num, const int channel, 
		const int height, const int width) const;


	//usage: b2 = b1(10)
	Blob sub_blob(const int channel_start) const;

	//usage: b2 = b1(10, 63)
	Blob sub_blob(const int channel_start, const int channel_end) const;

	//usage: b2 = b1(10, 0, 0)
	Blob sub_blob(const int channel_start, const int height_start, const int width_start) const;

	//usage: b2 = b1(10, 63, 0, 112, 0, 112)
	Blob sub_blob(const int channel_start, const int channel_end, const int height_start, 
		const int height_end, const int width_start, const int width_end) const;

	//usage: b2 = b1(vector<int>{0,1,2,3,5,8})
	Blob operator()(vector<int> channel) const;

	//usage: b2 = b1(vector<int>{10,63}, vector<int>{0,112}, vector<int>{0,112})
	Blob operator()(vector<int> channel, vector<int> height, vector<int> width) const;

	//usage: b1[1000]
	vector<double> operator[](const int index) const;


	//usage: b1.ave()
	vector<double> ave() const;

	//usage: b1.max()
	vector<double> max() const;

	//usage: b1.sum()
	vector<double> sum() const;

	//usage: b1.scale(10.0)
	void scale(const double scale_factor);

	//usage: b3 = b2 + b1
	Blob operator+(const Blob& rhs) const;

	//usage: b2 += b1
	Blob& operator+=(const Blob& rhs);

	//usage: b3 = b2 * b1
	Blob operator*(const Blob& rhs) const;

	//usage: b2 *= b1
	Blob& operator*=(const Blob& rhs);

	//usage: b2 = 10.0 * b1
	friend Blob operator*(const double scale_factor, const Blob& other);

	//usage: b2 = b1 * 10.0
	Blob operator*(const double scale_factor) const;

	//usage: b1 *= 10.0
	Blob& operator*=(const double scale_factor);

	//not available now
	template<typename functor>
	Blob& elem_wise_op(functor const &lambda_function);

private:
	vector<cube> data_;
	vector<int> shape_;
};	//class Blob

}	//namespace caffe

#endif //LOADPARAM_BLOB_HPP
