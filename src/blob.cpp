//
// Created by hua on 19-3-14.
//
#include "blob.hpp"

namespace caffe {

Blob::Blob(const int num, const int channels, const int height, const int width)
{
	this->shape_.push_back(num);
	this->shape_.push_back(channels);
	this->shape_.push_back(height);
	this->shape_.push_back(width);
	for (int i = 0; i < num; i++) {
		cube c(height, width, channels);
		c.fill(0);
		this->data_.push_back(c);
	}
}

Blob::Blob(const vector<int>& shape)
{
	this->shape_ = shape;
	for (int i = 0; i < shape[0]; i++) {
		cube c(shape[2], shape[3], shape[1]);
		c.fill(0);
		this->data_.push_back(c);
	}
}

Blob::Blob(const vector<cube>& cubes)
{
	for (auto cube : cubes) {
		this->data_.push_back(cube);
	}
	this->shape_.push_back(cubes.size());
	this->shape_.push_back(cubes[0].n_slices);
	this->shape_.push_back(cubes[0].n_rows);
	this->shape_.push_back(cubes[0].n_cols);
}

Blob::Blob(const Blob& other)
{
	this->data_ = other.data_;
	this->shape_ = other.shape_;
}

Blob& Blob::operator=(const Blob& other)
{
	if (this != &other) {
		if (!this->data_.empty()) {
			this->data_.clear();
		}
		this->data_ = other.data_;

		if (!this->shape_.empty()) {
			this->shape_.clear();
		}
		this->shape_ = other.shape_;
	}
	return *this;
}

void Blob::print_data() const 
{
	int i = 1;
	for (auto d : data_) {
		d.print("Batch " + to_string(i++) + " :");
	}
}

Blob Blob::sub_blob(const int channel_start) const
{
	Blob b;
	for (auto d : data_) {
		_ASSERT(channel_start < d.n_slices);
		b.data_.push_back(d.slices(channel_start, d.n_slices));
	}
	return b;
}

Blob Blob::sub_blob(const int channel_start, const int channel_end) const
{
	//_ASSERT(channel_start <= channel_end);
	Blob b;
	for (auto d : data_) {
		//_ASSERT(channel_end < d.n_slices);
		b.data_.push_back(d.slices(channel_start, channel_end));
	}
	return b;
}

Blob Blob::sub_blob(const int height_start, const int width_start, const int channel_start) const
{	
	Blob b;
	for (auto d : data_) {
		//_ASSERT(height_start < d.n_rows && width_start < d.n_cols && channel_start < d.n_slices);
		b.data_.push_back(d.subcube(height_start, height_start, channel_start, d.n_rows, d.n_cols, d.n_slices));
	}
	return b;
}

Blob Blob::sub_blob(const int height_start, const int height_end, const int width_start,
	const int width_end, const int channel_start, const int channel_end) const
{
	//_ASSERT(height_start < height_end && width_start < width_end && channel_start <= channel_end);
	Blob b;
	for (auto d : data_) {		
		//_ASSERT(height_end < d.n_rows && width_end < d.n_cols && channel_end < d.n_slices);
		b.data_.push_back(d.subcube(height_start, height_start, channel_start, height_end, width_end, channel_end));
	}
	return b;
}

Blob Blob::operator()(vector<int> channel) const
{
	_ASSERT(channel.size() == 2);
	Blob b;
	for (auto d : data_) {
		b.data_.push_back(d.slices(channel[0], channel[1]));
	}
	return b;
}

Blob Blob::operator()(vector<int> channel, vector<int> height, vector<int> width) const
{	
	_ASSERT(height.size() == 2 && width.size() == 2 && channel.size() == 2);
	//_ASSERT(height[0] < height[1] && width[0] < width[1]);
	Blob b;
	for (auto d : data_) {
		//_ASSERT(height[1] < d.n_rows && width[1] < d.n_cols);
		b.data_.push_back(d.subcube(height[0], width[0], channel[0], height[1], width[1], channel[1]));
	}
	return b;
}

vector<double> Blob::operator[](const int index) const
{
	vector<double> elem;
	for (auto d : data_) {
		elem.push_back(d(index));
	}
	return elem;
}

vector<double> Blob::sum() const
{
	vector<double> sum;
	for (auto d : data_) {
		sum.push_back(accu(d));
	}
	return sum;
}

void Blob::scale(const double scale_factor) {
	for (auto &d : data_) {
		d = scale_factor * d;
	}
}

Blob Blob::operator+(const Blob& other) const
{
	_ASSERT(this->data_.size() == other.data_.size());
	int i = 0;
	Blob b;
	for (auto d : data_) {
		b.data_.push_back(d + other.data_[i++]);
	}
	return b;
}

Blob& Blob::operator+=(const Blob& other) 
{
	_ASSERT(this->data_.size() == other.data_.size());
	int i = 0;
	for (auto &d : data_) {
		d += other.data_[i++];
	}
	return *this;
}

Blob Blob::operator*(const Blob& other) const
{
	_ASSERT(this->data_.size() == other.data_.size());
	int i = 0;
	Blob b;
	for (auto d : data_) {
		cube c(d.n_rows, d.n_cols, d.n_slices);
		for (int j = 0; j < d.n_slices; j++) {
			c.slice(j) = d.slice(j) * other.data_[i].slice(j);
		}
		b.data_.push_back(c);
		i++;
	}
	return b;
}

Blob& Blob::operator*=(const Blob& other) 
{
	_ASSERT(this->data_.size() == other.data_.size());
	int i = 0;
	for (auto &d : data_) {
		for (int j = 0; j < d.n_slices; j++) {
			d.slice(j) *= other.data_[i].slice(j);
		}
		i++;
	}
	return *this;
}

Blob Blob::operator*(const double scale_factor) const
{
	Blob b;
	for (auto d : data_) {
		b.data_.push_back(scale_factor * d);
	}
	return b;
}

Blob& Blob::operator*=(const double scale_factor)
{
	for (auto &d : data_) {
		d = scale_factor * d;
	}
	return *this;
}

template<typename functor>
Blob& Blob::elem_wise_op(functor lambda_function)
{
	for (auto &d : data_) {
		d.transform(lambda_function);
	}
}

}	//namespace caffe