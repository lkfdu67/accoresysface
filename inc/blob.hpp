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

using namespace caffe;

class Blob{
	;
};
#endif //LOADPARAM_BLOB_HPP
