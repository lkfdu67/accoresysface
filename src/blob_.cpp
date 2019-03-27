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
		cube c(height, width, channels, fill::zeros);
		this->data_.push_back(c);
	}
}

Blob::Blob(const vector<int>& shape)
{
	this->shape_ = shape;
	for (int i = 0; i < shape[0]; i++) {
		cube c(shape[2], shape[3], shape[1], fill::zeros);
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

Blob::Blob(const BlobProto& proto)
{
	//shape_[0] = proto.has_num() ? proto.num() : 0;
	//shape_[1] = proto.has_channels() ? proto.channels() : 0;
	//shape_[2] = proto.has_height() ? proto.height() : 0;
	//shape_[3] = proto.has_width() ? proto.width() : 0;

	int dim = 0;
	if (proto.has_shape()) {
		auto shape = proto.shape();
		dim = shape.dim_size();
		for (int i = 0; i < dim; i++) {
			shape_.push_back(shape.dim(i));
		}
	}

	int count = 0;
	double* data_array = nullptr;
	if (count = proto.double_data_size() > 0) {
		data_array = new double[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = proto.double_data(i);
		}
	}
	else if (count = proto.data_size() > 0) {
		data_array = new double[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = static_cast<double>(proto.data(i));
		}
	}

	_ASSERT(dim >= 4);
	_ASSERT(count == shape_[0] * shape_[1] * shape_[2] * shape_[3]);

	int n_bias = shape_[1] * shape_[2] * shape_[3];		// c*h*w
	int c_bias = shape_[2] * shape_[3];					// h*w
	int beg, end;
	for (int n = 0; n < shape_[0]; n++)
	{	
		cube cu;
		for (int c = 0; c < shape_[1]; c++)
		{
			beg = c * c_bias + n * n_bias;
			end = (c + 1) * c_bias + n * n_bias;
			vector<double> x(data_array + beg, data_array + end);
			mat m(x);
			m.reshape(shape_[3], shape_[2]);		// (w, h)
			cu.slice(c) = m.t();					// (h, w)
		}
		data_.push_back(cu);
	}

	delete []data_array;
}

Blob::Blob(const Blob& rhs)
{
	this->data_ = rhs.data_;
	this->shape_ = rhs.shape_;
}

Blob& Blob::operator=(const Blob& rhs)
{
	if (this != &rhs) {
		if (!this->data_.empty()) {
			this->data_.clear();
		}
		this->data_ = rhs.data_;

		if (!this->shape_.empty()) {
			this->shape_.clear();
		}
		this->shape_ = rhs.shape_;
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

string Blob::shape_string() const {
	stringstream ss;
	for (int i = 0; i < shape_.size(); i++) {
		ss << shape_[i] << " ";
	}
	return ss.str();
}

vector<int> Blob::size() const 
{
	vector<int> size_;
	for (auto d : data_) {
		size_.push_back(d.n_elem);
	}
	return size_;
}

void Blob::save_data(const std::string txt_path) const
{
	int i = 1;
	std::string batch = txt_path;
	std::string base_name;
	if (data_.size() > 1) {
		base_name = txt_path.substr(0, txt_path.length() - 4);
	}
	for (auto d : data_) {
		if (data_.size() > 1) {
			batch = base_name + "_batch" + to_string(i++) + ".txt";
		}		
		d.save(batch, raw_ascii);
	}
}

Blob Blob::load_data(const std::string txt_path, const int num, const int channel, const int height, const int width) const
{
	mat in;
	in.load(txt_path);

	vector<cube> cubes;
	for (int n = 0; n < num; n++) {
		int row_bias = n*channel*height;
		cube cu(height, width, channel);
		for (int c = 0; c < channel; c++) {
			int row1 = c*height + n*channel*height;
			cu.slice(c) = in.submat(row_bias + c*height, 0, row_bias + (c+1)*height - 1, width - 1);
		}
		cubes.push_back(cu);
	}

	return Blob(cubes);
}

Blob Blob::sub_blob(const int channel_start) const
{
	Blob b;
	for (auto d : data_) {
		//_ASSERT(channel_start < d.n_slices);
		b.data_.push_back(d.slices(channel_start, d.n_slices - 1));
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
		b.data_.push_back(d.subcube(height_start, height_start, channel_start, d.n_rows - 1, d.n_cols - 1, d.n_slices - 1));
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
	_ASSERT(channel.size() > 0);
	//std::find(channel.begin(), channel.end(), [](int lhs, int rhs) {
	//	return lhs > rhs; });
	auto max = max_element(channel.begin(), channel.end());
	int idx = max - channel.begin();

	Blob b;
	int i;
	for (auto d : data_) {
		_ASSERT(channel[idx] < d.n_slices);
		cube cu(d.n_rows, d.n_cols, channel.size(), fill::zeros);
		i = 0;
		for (auto c : channel) {
			cu.slice(i++) = d.slice(c);
		}
		b.data_.push_back(cu);
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

vector<double> Blob::ave() const
{
	vector<double> ave;
	for (auto d : data_) {
		ave.push_back(accu(d)/ static_cast<double>(d.n_elem));
	}
	return ave;
}

vector<double> Blob::max() const
{
	vector<double> max;
	for (auto d : data_) {
		max.push_back(d.max());
	}
	return max;
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

Blob Blob::operator+(const Blob& rhs) const
{
	_ASSERT(this->data_.size() == rhs.data_.size());
	int i = 0;
	Blob b;
	for (auto d : data_) {
		b.data_.push_back(d + rhs.data_[i++]);
	}
	return b;
}

Blob& Blob::operator+=(const Blob& rhs)
{
	_ASSERT(this->data_.size() == rhs.data_.size());
	int i = 0;
	for (auto &d : data_) {
		d += rhs.data_[i++];
	}
	return *this;
}

Blob Blob::operator*(const Blob& rhs) const
{
	_ASSERT(this->data_.size() == rhs.data_.size());
	int i = 0;
	Blob b;
	for (auto d : data_) {
		cube cu(d.n_rows, d.n_cols, d.n_slices);
		for (int j = 0; j < d.n_slices; j++) {
			cu.slice(j) = d.slice(j) * rhs.data_[i].slice(j);
		}
		b.data_.push_back(cu);
		i++;
	}
	return b;
}

Blob& Blob::operator*=(const Blob& rhs)
{
	_ASSERT(this->data_.size() == rhs.data_.size());
	int i = 0;
	for (auto &d : data_) {
		for (int j = 0; j < d.n_slices; j++) {
			d.slice(j) *= rhs.data_[i].slice(j);
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

Blob operator*(const double scale_factor, const Blob& other)
{
	Blob b;
	for (auto d : other.data_) {
		b.data_.push_back(scale_factor * d);
	}
	return b;
}

template<typename functor>
Blob& Blob::elem_wise_op(functor const &lambda_function)
{
	for (auto &d : data_) {
		d.transform(lambda_function);
	}
	return *this;
}

}	//namespace caffe