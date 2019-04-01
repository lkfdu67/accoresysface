//
// Created by hua on 19-3-14.
//
#include <blob_.hpp>
#include <cmath>

namespace caffe {

template<typename DType>
Blob<DType>::Blob(const int num, const int channels, const int height, const int width)
{
	this->shape_.push_back(num);
	this->shape_.push_back(channels);
	this->shape_.push_back(height);
	this->shape_.push_back(width);
	for (int i = 0; i < num; i++) {
		Cube<DType> c(height, width, channels, fill::zeros);
		this->data_.push_back(c);
	}
}

template<typename DType>
Blob<DType>::Blob(const vector<int>& shape)
{
	this->shape_ = shape;
	for (int i = 0; i < shape[0]; i++) {
		Cube<DType> c(shape[2], shape[3], shape[1], fill::zeros);
		this->data_.push_back(c);
	}
}

template<typename DType>
Blob<DType>::Blob(const vector<Cube<DType>>& cubes)
{
	for (auto cube : cubes) {
		this->data_.push_back(cube);
	}
	this->shape_.push_back(cubes.size());
	this->shape_.push_back(cubes[0].n_slices);
	this->shape_.push_back(cubes[0].n_rows);
	this->shape_.push_back(cubes[0].n_cols);
}

template<typename DType>
Blob<DType>::Blob(const BlobProto& proto)
{
	//shape_[0] = proto.has_num() ? proto.num() : 0;
	//shape_[1] = proto.has_channels() ? proto.channels() : 0;
	//shape_[2] = proto.has_height() ? proto.height() : 0;
	//shape_[3] = proto.has_width() ? proto.width() : 0;
	
	CHECK(proto.has_shape());

	int dim = 0;
	auto pshape = proto.shape();
	dim = pshape.dim_size();
	for (int i = 0; i < dim; i++) {
		this->shape_.push_back(pshape.dim(i));
	}
	
	int count = 0;
	DType* data_array = nullptr;
	if (count = proto.double_data_size() > 0) {
		data_array = new DType[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = static_cast<DType>(proto.double_data(i));
		}
	}
	else if (count = proto.data_size() > 0) {
		data_array = new DType[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = static_cast<DType>(proto.data(i));
		}
	}

	CHECK_EQ(dim, 4);
	CHECK_EQ(count, this->shape_[0] * this->shape_[1] * this->shape_[2] * this->shape_[3]);

	int n_bias = this->shape_[1] * this->shape_[2] * this->shape_[3];		// c*h*w
	int c_bias = this->shape_[2] * this->shape_[3];					// h*w
	int beg, end;
	for (int n = 0; n < this->shape_[0]; n++) {
		Cube<DType> cu;
		for (int c = 0; c < this->shape_[1]; c++)	{
			beg = c * c_bias + n * n_bias;
			end = (c + 1) * c_bias + n * n_bias;
			vector<DType> x(data_array + beg, data_array + end);
			Mat<DType> m(x);
			m.reshape(this->shape_[3], this->shape_[2]);	// (w, h)
			cu.slice(c) = m.t();							// (h, w)
		}
		this->data_.push_back(cu);
	}

	delete []data_array;
	data_array = nullptr;
}

template<typename DType>
Blob<DType>::Blob(const Blob<DType>& rhs)
{
	this->data_ = rhs.data_;
	this->shape_ = rhs.shape_;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator=(const Blob<DType>& rhs)
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

template<typename DType>
void Blob<DType>::FromProto(const BlobProto& proto, bool reshape)
{
	CHECK(proto.has_shape());

	int dim = 0;
	vector<int> shape;
	auto pshape = proto.shape();
	dim = pshape.dim_size();
	for (int i = 0; i < dim; i++) {
		shape.push_back(pshape.dim(i));
	}

	if (reshape) {
		if (!this->shape_.empty()) {
			this->shape_.clear();
		}
		this->shape_ = shape;

		if (!this->data_.empty()) {
			this->data_.clear();
		}
	}
	else {
		CHECK(this->shape_ == shape);
		if (!this->data_.empty()) {
			this->data_.clear();
		}
	}

	int count = 0;
	DType* data_array = nullptr;
	if (count = proto.double_data_size() > 0) {
		data_array = new DType[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = static_cast<DType>(proto.double_data(i));
		}
	}
	else if (count = proto.data_size() > 0) {
		data_array = new DType[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = static_cast<DType>(proto.data(i));
		}
	}

	CHECK_EQ(dim, 4);
	CHECK_EQ(count, this->shape_[0] * this->shape_[1] * this->shape_[2] * this->shape_[3]);

	int n_bias = this->shape_[1] * this->shape_[2] * this->shape_[3];		// c*h*w
	int c_bias = this->shape_[2] * this->shape_[3];					// h*w
	int beg, end;
	for (int n = 0; n < this->shape_[0]; n++) {
		Cube<DType> cu;
		for (int c = 0; c < this->shape_[1]; c++) {
			beg = c * c_bias + n * n_bias;
			end = (c + 1) * c_bias + n * n_bias;
			vector<DType> x(data_array + beg, data_array + end);
			Mat<DType> m(x);
			m.reshape(this->shape_[3], this->shape_[2]);	// (w, h)
			cu.slice(c) = m.t();							// (h, w)
		}
		this->data_.push_back(cu);
	}

	delete[]data_array;
	data_array = nullptr;
}

//template<typename DType>
//Blob<DType> Blob<DType>::Reshape(const bool channel_priority, const bool col_priority) const
//{
//	CHECK_EQ(this->shape_.size(), 4);
//	Blob<DType> b;
//
//	return b;
//}

//template<typename DType>
//Blob<DType>& Blob<DType>::Reshape(const bool channel_priority, const bool col_priority)
//{
//	CHECK_EQ(this->shape_.size(), 4);
//	
//	if (channel_priority) {
//
//	}
//	else {
//		//for (auto &d : this->data_) {
//		//	vector<DType> x;
//		//	for (int i = 0; i < d.n_elem_slice; i++) {
//		//		
//		//	}
//		//}
//	}
//
//	return *this;
//}

template<typename DType>
bool Blob<DType>::ShapeEquals(const BlobProto& proto) const
{
	CHECK(proto.has_shape());

	int dim = 0;
	vector<int> shape;
	auto pshape = proto.shape();
	dim = pshape.dim_size();
	for (int i = 0; i < dim; i++) {
		shape.push_back(pshape.dim(i));
	}

	return this->shape_ == shape;
}

template<typename DType>
string Blob<DType>::shape_string() const {
	std::stringstream ss;
	for (int i = 0; i < this->shape_.size(); i++) {
		ss << this->shape_[i] << " ";
	}
	return ss.str();
}

template<typename DType>
vector<int> Blob<DType>::size() const
{
	vector<int> size_vec;
	for (auto d : this->data_) {
		size_vec.push_back(d.n_elem);
	}
	return size_vec;
}

template<typename DType>
void Blob<DType>::print_data() const
{
	int i = 1;
	for (auto d : this->data_) {
		d.print("Batch " + to_string(i++) + " :");
	}
}

template<typename DType>
void Blob<DType>::save_data(const string& txt_path) const
{
	int i = 1;
	string batch = txt_path;
	string base_name;
	if (this->data_.size() > 1) {
		base_name = txt_path.substr(0, txt_path.length() - 4);
	}
	for (auto d : this->data_) {
		if (this->data_.size() > 1) {
			batch = base_name + "_batch" + to_string(i++) + ".txt";
		}		
		d.save(batch, raw_ascii);
	}
}

template<typename DType>
Blob<DType> Blob<DType>::load_data(const string& txt_path, const int num, const int channel, const int height, const int width) const
{
	Mat<DType> in;
	in.load(txt_path);

	vector<Cube<DType>> cubes;
	for (int n = 0; n < num; n++) {
		int row_bias = n*channel*height;
		Cube<DType> cu(height, width, channel);
		for (int c = 0; c < channel; c++) {
			int row1 = c*height + n*channel*height;
			cu.slice(c) = in.submat(row_bias + c*height, 0, row_bias + (c+1)*height - 1, width - 1);
		}
		cubes.push_back(cu);
	}
	return Blob(cubes);
}

template<typename DType>
Blob<DType> Blob<DType>::sub_blob(const int channel_start) const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		CHECK_LT(channel_start, d.n_slices);
		b.data_.push_back(d.slices(channel_start, d.n_slices - 1));
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType> Blob<DType>::sub_blob(const int channel_start, const int channel_end) const
{
	CHECK_LE(channel_start, channel_end);
	Blob<DType> b;
	for (auto d : this->data_) {
		CHECK_LT(channel_end, d.n_slices);
		b.data_.push_back(d.slices(channel_start, channel_end));
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType> Blob<DType>::sub_blob(const int channel_start, const int height_start, const int width_start) const
{	
	Blob<DType> b;
	for (auto d : this->data_) {
		CHECK_LT(channel_start, d.n_slices);
		CHECK_LT(height_start, d.n_rows);
		CHECK_LT(width_start, d.n_cols);		
		b.data_.push_back(d.subcube(height_start, height_start, channel_start, d.n_rows - 1, d.n_cols - 1, d.n_slices - 1));
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType> Blob<DType>::sub_blob(const int height_start, const int height_end, const int width_start,
	const int width_end, const int channel_start, const int channel_end) const
{
	CHECK_LE(channel_start, channel_end);
	CHECK_LE(height_start, height_end);
	CHECK_LE(width_start, width_end);
	Blob<DType> b;
	for (auto d : this->data_) {		
		CHECK_LT(channel_end, d.n_slices);
		CHECK_LT(height_end, d.n_rows);
		CHECK_LT(width_end, d.n_cols);
		b.data_.push_back(d.subcube(height_start, height_start, channel_start, height_end, width_end, channel_end));
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType> Blob<DType>::sub_blob(const vector<int>& channel) const
{
	CHECK_GT(channel.size(), 0);
	//std::find(channel.begin(), channel.end(), [](int lhs, int rhs) {
	//	return lhs > rhs; });
	auto max_iter = std::max_element(channel.begin(), channel.end());
	int max_idx = max_iter - channel.begin();

	Blob<DType> b;
	int i;
	for (auto d : this->data_) {
		CHECK_LT(channel[max_idx], d.n_slices);
		Cube<DType> cu(d.n_rows, d.n_cols, channel.size(), fill::zeros);
		i = 0;
		for (auto c : channel) {
			cu.slice(i++) = d.slice(c);
		}
		b.data_.push_back(cu);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType> Blob<DType>::sub_blob(const vector<int>& channel, const vector<int>& height, const vector<int>& width) const
{	
	CHECK_EQ(channel.size(), 2);
	CHECK_EQ(height.size(), 2);
	CHECK_EQ(width.size(), 2);
	CHECK_LE(channel[0], channel[1]);
	CHECK_LE(height[0], height[1]);
	CHECK_LE(width[0], width[1]);

	Blob<DType> b;
	for (auto d : this->data_) {
		CHECK_LT(channel[1], d.n_slices);
		CHECK_LT(height[1], d.n_rows);
		CHECK_LT(width[1], d.n_cols);
		b.data_.push_back(d.subcube(height[0], width[0], channel[0], height[1], width[1], channel[1]));
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
DType Blob<DType>::operator()(const int num, const int channel, const int height, const int width)
{
	CHECK_LT(num, this->shape_[0]);
	CHECK_LT(channel, this->shape_[1]);
	CHECK_LT(height, this->shape_[2]);
	CHECK_LT(width, this->shape_[3]);
	return this->data_[num](height, width, channel);
}

template<typename DType>
DType Blob<DType>::operator()(const vector<int>& shape)
{
	CHECK_EQ(shape.size(), 4);
	CHECK_LT(shape[0], this->shape_[0]);
	CHECK_LT(shape[1], this->shape_[1]);
	CHECK_LT(shape[2], this->shape_[2]);
	CHECK_LT(shape[3], this->shape_[3]);
	return this->data_[shape[0]](shape[2], shape[3], shape[1]);
}

template<typename DType>
vector<DType> Blob<DType>::operator[](const int index) const
{
	CHECK_GE(index, 0);
	vector<DType> elem_vec;
	for (auto d : this->data_) {
		CHECK_LT(index, d.n_elem);
		elem_vec.push_back(d(index));
	}
	return elem_vec;
}

template<typename DType>
vector<DType> Blob<DType>::sum() const
{
	vector<DType> sum_vec;
	for (auto d : this->data_) {
		sum_vec.push_back(accu(d));
	}
	return sum_vec;
}

template<typename DType>
vector<DType> Blob<DType>::ave() const
{
	vector<DType> ave_vec;
	for (auto d : this->data_) {
		ave_vec.push_back(accu(d) / static_cast<DType>(d.n_elem));
	}
	return ave_vec;
}

template<typename DType>
vector<DType> Blob<DType>::max() const
{
	vector<DType> max_vec;
	for (auto d : this->data_) {
		max_vec.push_back(d.max());
	}
	return max_vec;
}

template<typename DType>
vector<DType> Blob<DType>::min() const
{
	vector<DType> min_vec;
	for (auto d : this->data_) {
		min_vec.push_back(d.min());
	}
	return min_vec;
}

template<typename DType>
Blob<DType> Blob<DType>::exp() const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		Cube<DType> cu(d.n_rows, d.n_cols, d.n_slices, fill::zeros);
		DType* cu_ptr = cu.memptr();
		DType* ptr = d.memptr();
		const uword N = d.n_elem;
		uword ii, jj;
		for (ii = 0, jj = 1; jj < N; ii += 2, jj += 2){
			DType temp_ii = ptr[ii];
			DType temp_jj = ptr[jj];
			temp_ii = std::exp(temp_ii);
			temp_jj = std::exp(temp_jj);
			cu_ptr[ii] = temp_ii;
			cu_ptr[jj] = temp_jj;
		}
		if (ii < N){
			cu_ptr[ii] = std::exp(ptr[ii]);
		}
		b.data_.push_back(cu);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::exp_inplace()
{
	for (auto &d : this->data_) {
		DType* ptr = d.memptr();
		const uword N = d.n_elem;
		uword ii, jj;
		for (ii = 0, jj = 1; jj < N; ii += 2, jj += 2) {
			DType temp_ii = ptr[ii];
			DType temp_jj = ptr[jj];
			temp_ii = std::exp(temp_ii);
			temp_jj = std::exp(temp_jj);
			ptr[ii] = temp_ii;
			ptr[jj] = temp_jj;
		}
		if (ii < N) {
			ptr[ii] = std::exp(ptr[ii]);
		}
	}
	return *this;
}

template<typename DType>
Blob<DType> Blob<DType>::log() const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		Cube<DType> cu(d.n_rows, d.n_cols, d.n_slices, fill::zeros);
		DType* cu_ptr = cu.memptr();
		DType* ptr = d.memptr();
		const uword N = d.n_elem;
		uword ii, jj;
		for (ii = 0, jj = 1; jj < N; ii += 2, jj += 2) {
			DType temp_ii = ptr[ii];
			DType temp_jj = ptr[jj];
			temp_ii = std::log(temp_ii);
			temp_jj = std::log(temp_jj);
			cu_ptr[ii] = temp_ii;
			cu_ptr[jj] = temp_jj;
		}
		if (ii < N) {
			cu_ptr[ii] = std::log(ptr[ii]);
		}
		b.data_.push_back(cu);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::log_inplace()
{
	for (auto &d : this->data_) {
		DType* ptr = d.memptr();
		const uword N = d.n_elem;
		uword ii, jj;
		for (ii = 0, jj = 1; jj < N; ii += 2, jj += 2) {
			DType temp_ii = ptr[ii];
			DType temp_jj = ptr[jj];
			temp_ii = std::log(temp_ii);
			temp_jj = std::log(temp_jj);
			ptr[ii] = temp_ii;
			ptr[jj] = temp_jj;
		}
		if (ii < N) {
			ptr[ii] = std::log(ptr[ii]);
		}
	}
	return *this;
}

template<typename DType>
Blob<DType> Blob<DType>::join(const Blob<DType>& rhs) const
{
	CHECK_EQ(this->shape_.size(), rhs.shape_.size());
	CHECK_EQ(this->shape_[0], rhs.shape_[0]);	//n
	CHECK_EQ(this->shape_[2], rhs.shape_[2]);	//h
	CHECK_EQ(this->shape_[3], rhs.shape_[3]);	//w

	int i = 0;
	Blob<DType> b;
	for (auto d : this->data_) {
		Cube<DType> cu(d);
		cu.insert_slices(this->shape_[1], rhs.data_[i++]);
		b.data_.push_back(cu);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::join_inplace(const Blob<DType>& rhs)
{
	CHECK_EQ(this->shape_.size(), rhs.shape_.size());
	CHECK_EQ(this->shape_[0], rhs.shape_[0]);	//n
	CHECK_EQ(this->shape_[2], rhs.shape_[2]);	//h
	CHECK_EQ(this->shape_[3], rhs.shape_[3]);	//w

	int i = 0;
	for (auto &d : this->data_) {
		d.insert_slices(this->shape_[1], rhs.data_[i++]);
	}
	return *this;
}

template<typename DType>
Blob<DType> Blob<DType>::operator+(const Blob<DType>& rhs) const
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	Blob<DType> b;
	for (auto d : this->data_) {
		b.data_.push_back(d + rhs.data_[i++]);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator+=(const Blob<DType>& rhs)
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	for (auto &d : this->data_) {
		d += rhs.data_[i++];
	}
	return *this;
}

template<typename DType>
Blob<DType> Blob<DType>::operator-(const Blob<DType>& rhs) const
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	Blob<DType> b;
	for (auto d : this->data_) {
		b.data_.push_back(d - rhs.data_[i++]);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator-=(const Blob<DType>& rhs)
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	for (auto &d : this->data_) {
		d -= rhs.data_[i++];
	}
	return *this;
}

template<typename DType>
Blob<DType> Blob<DType>::operator*(const Blob<DType>& rhs) const
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	Blob<DType> b;
	for (auto d : this->data_) {
		Cube<DType> cu(d.n_rows, d.n_cols, d.n_slices);
		for (int j = 0; j < d.n_slices; j++) {
			cu.slice(j) = d.slice(j) % rhs.data_[i].slice(j);
		}
		b.data_.push_back(cu);
		i++;
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator*=(const Blob<DType>& rhs)
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	for (auto &d : this->data_) {
		for (int j = 0; j < d.n_slices; j++) {
			d.slice(j) %= rhs.data_[i].slice(j);
		}
		i++;
	}
	return *this;
}

template<typename DType>
Blob<DType> Blob<DType>::mat_mul(const Blob<DType>& rhs) const
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	Blob<DType> b;
	for (auto d : this->data_) {
		Cube<DType> cu(d.n_rows, d.n_cols, d.n_slices);
		for (int j = 0; j < d.n_slices; j++) {
			cu.slice(j) = d.slice(j) * rhs.data_[i].slice(j);
		}
		b.data_.push_back(cu);
		i++;
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::mat_mul_inplace(const Blob<DType>& rhs)
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	for (auto &d : this->data_) {
		for (int j = 0; j < d.n_slices; j++) {
			d.slice(j) *= rhs.data_[i].slice(j);
		}
		i++;
	}
	return *this;
}

template<typename DType>
Blob<DType> Blob<DType>::operator/(const Blob<DType>& rhs) const
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	Blob<DType> b;
	for (auto d : this->data_) {
		Cube<DType> cu(d.n_rows, d.n_cols, d.n_slices);
		for (int j = 0; j < d.n_slices; j++) {
			cu.slice(j) = d.slice(j) / rhs.data_[i].slice(j);
		}
		b.data_.push_back(cu);
		i++;
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator/=(const Blob<DType>& rhs)
{
	CHECK_EQ(this->shape_.size(), 4);
	CHECK(this->ShapeEquals(rhs));

	int i = 0;
	for (auto &d : this->data_) {
		for (int j = 0; j < d.n_slices; j++) {
			d.slice(j) /= rhs.data_[i].slice(j);
		}
		i++;
	}
	return *this;
}

template<typename DType>
Blob<DType> Blob<DType>::operator*(const DType scale_factor) const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		b.data_.push_back(scale_factor * d);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator*=(const DType scale_factor)
{
	for (auto &d : this->data_) {
		d = scale_factor * d;
	}
	return *this;
}

//template<typename DType>
//Blob<DType> operator*(const DType scale_factor, const Blob<DType>& other)
//{
//	Blob<DType> b;
//	for (auto d : other.data_) {
//		b.data_.push_back(scale_factor * d);
//	}
//	b.shape_ = other.shape_;
//	return b;
//}

template<typename DType>
void Blob<DType>::scale(const DType scale_factor) {
	for (auto &d : this->data_) {
		d = scale_factor * d;
	}
}

//template<typename DType>
//template<typename functor>
//Blob<DType> Blob<DType>::elem_wise(functor const &lambda_function) const
//{
//	Blob<DType> b;
//	for (auto d : this->data_) {
//		b.data_.push_back(d.transform(lambda_function));
//	}
//	b.shape_ = this->shape_;
//	return b;
//}

//template<typename DType>
//template<typename functor>
//Blob<DType>& Blob<DType>::elem_wise_inplace(functor const &lambda_function)
//{
//	for (auto &d : this->data_) {
//		d.transform(lambda_function);
//
//		//DType* out_mem = d.memptr();
//		//const uword N = d.n_elem;
//		//uword ii, jj;
//		//for (ii = 0, jj = 1; jj < N; ii += 2, jj += 2) {
//		//	DType tmp_ii = out_mem[ii];
//		//	DType tmp_jj = out_mem[jj];
//		//	tmp_ii = DType(lambda_function(tmp_ii));
//		//	tmp_jj = DType(lambda_function(tmp_jj));
//		//	out_mem[ii] = tmp_ii;
//		//	out_mem[jj] = tmp_jj;
//		//}
//		//if (ii < N)	{
//		//	out_mem[ii] = DType(lambda_function(out_mem[ii]));
//		//}
//	}
//	return *this;
//}

//explicit instantiate class Blob for declaration and implementation separation
template class Blob<float>;
template class Blob<double>;

}	//namespace caffe