//
// Created by hua on 19-3-14.
//
#include<blob_.hpp>
#include<regex>
#include<cmath>
#include<sstream>
#include <typeinfo>

namespace caffe {

inline vector<string> string_split(const string& in, const string& delim)
{
	if (in == delim) {
		return vector<string>();
	}
	regex re(delim);
	return vector<string>{sregex_token_iterator(in.begin(), in.end(), re, -1),
						  sregex_token_iterator()};
}

inline int str_to_int(const string& in)
{
	stringstream ss(in);
	int i;
	ss >> i;
	return i;
}

template<typename Ty>
void cv_mat_to_arma_mat(const cv::Mat& cv_mat_in, vector<Mat<Ty>>& arma_mat_out)
{
	cv::Mat copy;
	vector<cv::Mat> channels;
	cv_mat_in.copyTo(copy);

	if (copy.channels() == 3 && is_same<Ty, float>::value == true) {
		copy.convertTo(copy, CV_32FC3);
	}
	else if (copy.channels() == 3 && is_same<Ty, double>::value == true) {
		copy.convertTo(copy, CV_64FC3);
	}
	else if (copy.channels() == 1 && is_same<Ty, float>::value == true) {
		copy.convertTo(copy, CV_32FC1);
	}
	else if (copy.channels() == 1 && is_same<Ty, double>::value == true) {
		copy.convertTo(copy, CV_64FC1);
	}
	cv::split(copy, channels);

	for (int c = 0; c < channels.size(); c++) {
		/*Mat<Ty> m(channels[c].rows, channels[c].cols);
		for (int h = 0; h < channels[c].rows; h++) {
			for (int w = 0; w < channels[c].cols; w++) {
				m.at(h, w) = static_cast<Ty>(channels[c].data[h * channels[c].cols + w]);
			}
		}*/
		cv::Mat_<Ty> temp(channels[c].t()); 
		Mat<Ty> m = arma::Mat<Ty>(reinterpret_cast<Ty*>(temp.data),
					static_cast<arma::uword>(temp.cols),
					static_cast<arma::uword>(temp.rows),
					true,
					true);

		arma_mat_out.push_back(m);
	}
}

template<typename Ty>
void arma_mat_to_cv_mat(const Mat<Ty>& arma_mat_in, cv::Mat& cv_mat_out)
{
	//cv::Mat tmp(static_cast<int>(arma_mat_in.n_rows), static_cast<int>(arma_mat_in.n_cols), CV_32FC1);
	//for (int r = 0; r < tmp.rows; r++) {
	//	for (int c = 0; c < tmp.cols; c++) {
	//		tmp.data[r * tmp.cols + c] = static_cast<float>(arma_mat_in(r, c));
	//	}
	//}	
	//tmp.copyTo(cv_mat_out);

	cv::transpose(cv::Mat_<Ty>(static_cast<int>(arma_mat_in.n_cols),
				static_cast<int>(arma_mat_in.n_rows),
				const_cast<Ty*>(arma_mat_in.memptr())),
				cv_mat_out);
}

template<typename DType>
Blob<DType>::Blob(const int num, const int channels, const int height, const int width)
{
	this->shape_.push_back(num);
	this->shape_.push_back(channels);
	this->shape_.push_back(height);
	this->shape_.push_back(width);
	for (int i = 0; i < num; i++) {
		Cube<DType> cu(height, width, channels, fill::zeros);
		this->data_.push_back(cu);
	}
}

template<typename DType>
Blob<DType>::Blob(const vector<int>& shape)
{
	CHECK_EQ(shape.size(), 4);
	this->shape_ = shape;
	for (int i = 0; i < shape[0]; i++) {
		Cube<DType> cu(shape[2], shape[3], shape[1], fill::zeros);
		this->data_.push_back(cu);
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
Blob<DType>::Blob(const BlobShape& shape)
{
	int dim = shape.dim_size();
	CHECK_EQ(dim, 4);

	for (int i = 0; i < dim; i++) {
		this->shape_.push_back(shape.dim(i));
	}
	for (int i = 0; i < this->shape_[0]; i++) {
		Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
		this->data_.push_back(cu);
	}
}

template<typename DType>
Blob<DType>::Blob(const BlobProto& proto)
{
	CHECK(proto.has_shape());

	int dim = 0;
	auto pshape = proto.shape();
	dim = pshape.dim_size();
	for (int i = 0; i < dim; i++) {
		this->shape_.push_back(pshape.dim(i));
	}
	
	int count = 0;
	DType* data_array = nullptr;
	if (proto.double_data_size() > 0) {
		count = proto.double_data_size();
		data_array = new DType[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = static_cast<DType>(proto.double_data(i));
		}
	}
	else if (proto.data_size() > 0) {
		count = proto.data_size();
		data_array = new DType[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = static_cast<DType>(proto.data(i));
		}
	}

	int n_bias, c_bias, beg, end;
	switch (dim) {
	case 4:
		CHECK_EQ(count, this->shape_[0] * this->shape_[1] * this->shape_[2] * this->shape_[3]);

		n_bias = this->shape_[1] * this->shape_[2] * this->shape_[3];	// c*h*w
		c_bias = this->shape_[2] * this->shape_[3];						// h*w
		for (int n = 0; n < this->shape_[0]; n++) {
			Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
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
		break;
	case 2:
		CHECK_EQ(count, this->shape_[0] * this->shape_[1]);
		this->shape_.push_back(1);
		this->shape_.push_back(1);		//change shape from n*c*h*w to n*c'*1*1

		n_bias = this->shape_[1];	// c*1*1
		c_bias = 1;					// 1*1
		for (int n = 0; n < this->shape_[0]; n++) {
			Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
			for (int c = 0; c < this->shape_[1]; c++) {
				beg = c * c_bias + n * n_bias;
				end = (c + 1) * c_bias + n * n_bias;
				vector<DType> x(data_array + beg, data_array + end);
				Mat<DType> m(x);
				//m.reshape(1, 1);		// (w, h)
				//cu.slice(c) = m.t();	// (h, w)
				cu.slice(c) = m;
			}
			this->data_.push_back(cu);
		}
		break;
	case 1:
		CHECK_EQ(count, this->shape_[0]);
		this->shape_.push_back(1);
		this->shape_.push_back(1);
		this->shape_.push_back(1);
		this->shape_[1] = this->shape_[0];
		this->shape_[0] = 1;				//change shape from n*c*h*w to 1*c'*1*1

		n_bias = this->shape_[1];	// c*1*1
		c_bias = 1;					// 1*1
		for (int n = 0; n < this->shape_[0]; n++) {
			Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
			for (int c = 0; c < this->shape_[1]; c++) {
				beg = c * c_bias + n * n_bias;
				end = (c + 1) * c_bias + n * n_bias;
				vector<DType> x(data_array + beg, data_array + end);
				Mat<DType> m(x);
				//m.reshape(1, 1);		// (w, h)
				//cu.slice(c) = m.t();	// (h, w)
				cu.slice(c) = m;
			}
			this->data_.push_back(cu);
		}
		break;
	}
	
	if (data_array != nullptr) {
		delete []data_array;
		data_array = nullptr;
	}
}

template<typename DType>
Blob<DType>::Blob(const cv::Mat& cv_img)
{
	CHECK(cv_img.data);

	vector<Mat<DType>> mats;
	cv_mat_to_arma_mat(cv_img, mats);

	Cube<DType> cu(cv_img.rows, cv_img.cols, cv_img.channels(), fill::zeros);
	for (int i = 0; i < cv_img.channels(); i++) {
		cu.slice(i) = mats[i];
	}

	this->data_.push_back(cu);
	this->shape_.push_back(1);
	this->shape_.push_back(cv_img.channels());
	this->shape_.push_back(cv_img.rows);
	this->shape_.push_back(cv_img.cols);
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
void Blob<DType>::FromProto(const BlobProto& proto, bool reshape/* = true*/)
{
	int dim = 0;
	vector<int> shape;
	if(proto.has_shape()) {
		auto pshape = proto.shape();
		dim = pshape.dim_size();
		for (int i = 0; i < dim; i++) {
			shape.push_back(pshape.dim(i));
		}
	}else{
		if(proto.has_num()) {
			shape.push_back(proto.num());
			dim++;
		}
		if(proto.has_channels()) {
			shape.push_back(proto.channels());
			dim++;
		}
		if(proto.has_height()) {
			shape.push_back(proto.height());
			dim++;
		}
		if(proto.has_width()) {
			shape.push_back(proto.width());
			dim++;
		}
	}

	int count = 0;
	DType* data_array = nullptr;
	if (proto.double_data_size() > 0) {
		count = proto.double_data_size();
		data_array = new DType[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = static_cast<DType>(proto.double_data(i));
		}
	}
	else if (proto.data_size() > 0) {
		count = proto.data_size();
		data_array = new DType[count];
		for (int i = 0; i < count; ++i) {
			data_array[i] = static_cast<DType>(proto.data(i));
		}

	}

	int n_bias, c_bias, beg, end;
	if (reshape) {
		if (!this->shape_.empty()) {
			this->shape_.clear();
		}
		this->shape_ = shape;

		if (!this->data_.empty()) {
			this->data_.clear();
		}

		switch (dim) {
		case 4:
			CHECK_EQ(count, this->shape_[0] * this->shape_[1] * this->shape_[2] * this->shape_[3]);

			n_bias = this->shape_[1] * this->shape_[2] * this->shape_[3];	// c*h*w
			c_bias = this->shape_[2] * this->shape_[3];						// h*w
			for (int n = 0; n < this->shape_[0]; n++) {
				Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
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
			break;
		case 2:
			CHECK_EQ(count, this->shape_[0] * this->shape_[1]);
			this->shape_.push_back(1);
			this->shape_.push_back(1);		//change shape from n*c*h*w to n*c'*1*1

			n_bias = this->shape_[1];	// c*1*1
			c_bias = 1;					// 1*1
			for (int n = 0; n < this->shape_[0]; n++) {
				Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
				for (int c = 0; c < this->shape_[1]; c++) {
					beg = c * c_bias + n * n_bias;
					end = (c + 1) * c_bias + n * n_bias;
					vector<DType> x(data_array + beg, data_array + end);
					Mat<DType> m(x);
					//m.reshape(1, 1);		// (w, h)
					//cu.slice(c) = m.t();	// (h, w)
					cu.slice(c) = m;
				}
				this->data_.push_back(cu);
			}
			break;
		case 1:
			CHECK_EQ(count, this->shape_[0]);
			this->shape_.push_back(1);
			this->shape_.push_back(1);
			this->shape_.push_back(1);		
			this->shape_[1] = this->shape_[0];
			this->shape_[0] = 1;				//change shape from n*c*h*w to 1*c'*1*1

			n_bias = this->shape_[1];	// c*1*1
			c_bias = 1;					// 1*1
			for (int n = 0; n < this->shape_[0]; n++) {
				Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
				for (int c = 0; c < this->shape_[1]; c++) {
					beg = c * c_bias + n * n_bias;
					end = (c + 1) * c_bias + n * n_bias;
					vector<DType> x(data_array + beg, data_array + end);
					Mat<DType> m(x);
					//m.reshape(1, 1);		// (w, h)
					//cu.slice(c) = m.t();	// (h, w)
					cu.slice(c) = m;
				}
				this->data_.push_back(cu);
			}
			break;
		}
	}
	else {		//reshape == false
		CHECK_EQ(count, this->shape_[0] * this->shape_[1] * this->shape_[2] * this->shape_[3]);

		if (!this->data_.empty()) {
			this->data_.clear();
		}

		n_bias = this->shape_[1] * this->shape_[2] * this->shape_[3];	// c*h*w
		c_bias = this->shape_[2] * this->shape_[3];						// h*w
		for (int n = 0; n < this->shape_[0]; n++) {
			Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
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
	}
	
	if (data_array != nullptr) {
		delete[]data_array;
		data_array = nullptr;
	}
}

template<typename DType>
void Blob<DType>::FromCvMat(const cv::Mat& cv_img)
{
	CHECK(cv_img.data);

	vector<Mat<DType>> mats;
	cv_mat_to_arma_mat(cv_img, mats);

	Cube<DType> cu(cv_img.rows, cv_img.cols, cv_img.channels(), fill::zeros);
	for (int i = 0; i < cv_img.channels(); i++) {
		cu.slice(i) = mats[i];
	}
	
	if (!this->data_.empty()) {
		this->data_.clear();
	}
	this->data_.push_back(cu);

	if (!this->shape_.empty()) {
		this->shape_.clear();
	}
	this->shape_.push_back(1);
	this->shape_.push_back(cv_img.channels());
	this->shape_.push_back(cv_img.rows);
	this->shape_.push_back(cv_img.cols);
}

template<typename DType>
void Blob<DType>::ToCvMat(vector<cv::Mat>& cv_imgs)
{
	CHECK_GT(this->data_.size(), 0);
	cv_imgs.resize(this->shape_[0]);
	int mean_id = 0;
	for (auto d : this->data_) {
		CHECK_GT(d.n_elem, 0);

		vector<cv::Mat> cv_mats;
		cv_mats.resize(d.n_slices);
		for (int i = 0; i < d.n_slices; i++) {
			arma_mat_to_cv_mat(d.slice(i), cv_mats[i]);
		}

		cv::Mat cv_img;
		cv::merge(cv_mats, cv_img);
		cv_imgs[mean_id++] = cv_img;

	}
}

template<typename DType>
Blob<DType>& Blob<DType>::Reshape(const int num, const int channels, const int height, const int width)
{
	if (!this->shape_.empty()) {
		this->shape_.clear();
	}
	if (!this->data_.empty()) {
		this->data_.clear();
	}

	this->shape_.push_back(num);
	this->shape_.push_back(channels);
	this->shape_.push_back(height);
	this->shape_.push_back(width);
	for (int i = 0; i < num; i++) {
		Cube<DType> cu(height, width, channels, fill::zeros);
		this->data_.push_back(cu);
	}

	return *this;
}

template<typename DType>
Blob<DType>& Blob<DType>::Reshape(const vector<int>& shape)
{
	CHECK_EQ(shape.size(), 4);
	if (!this->shape_.empty()) {
		this->shape_.clear();
	}
	if (!this->data_.empty()) {
		this->data_.clear();
	}

	this->shape_ = shape;
	for (int i = 0; i < this->shape_[0]; i++) {
		Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
		this->data_.push_back(cu);
	}
	return *this;
}

template<typename DType>
Blob<DType>& Blob<DType>::Reshape(const BlobShape& shape)
{
	int dim = shape.dim_size();
	CHECK_EQ(dim, 4);

	if (!this->shape_.empty()) {
		this->shape_.clear();
	}
	if (!this->data_.empty()) {
		this->data_.clear();
	}

	for (int i = 0; i < dim; i++) {
		this->shape_.push_back(shape.dim(i));
	}
	for (int i = 0; i < this->shape_[0]; i++) {
		Cube<DType> cu(this->shape_[2], this->shape_[3], this->shape_[1], fill::zeros);
		this->data_.push_back(cu);
	}
	return *this;
}

template<typename DType>
bool Blob<DType>::ShapeEquals(const BlobProto& proto) const
{
	CHECK(proto.has_shape());

	int dim = 0;
	vector<int> shape;
	auto pshape = proto.shape();
	dim = pshape.dim_size();
	for (int i = 0; i < dim; i++) {
		if (dim > 1) {
			shape.push_back(pshape.dim(i));			
		}
		else {
			shape.push_back(1);
			shape.push_back(pshape.dim(i));
			shape.push_back(1);
			shape.push_back(1);
			break;
		}
	}

	return this->shape_ == shape;
}

template<typename DType>
string Blob<DType>::shape_string() const {
	stringstream ss;
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
		Cube<DType> cu(height, width, channel, fill::zeros);
		for (int c = 0; c < channel; c++) {
			int row1 = c*height + n*channel*height;
			cu.slice(c) = in.submat(row_bias + c*height, 0, row_bias + (c+1)*height - 1, width - 1);
		}
		cubes.push_back(cu);
	}
	return Blob(cubes);
}

//template<typename DType>
//Blob<DType> Blob<DType>::sub_blob(const vector<int>& channel) const
//{
//	CHECK_GT(channel.size(), 0);
//	//std::find(channel.begin(), channel.end(), [](int lhs, int rhs) {
//	//	return lhs > rhs; });
//	auto max_iter = std::max_element(channel.begin(), channel.end());
//	int max_idx = max_iter - channel.begin();
//
//	Blob<DType> b;
//	int i;
//	for (auto d : this->data_) {
//		CHECK_LT(channel[max_idx], d.n_slices);
//		Cube<DType> cu(d.n_rows, d.n_cols, channel.size(), fill::zeros);
//		i = 0;
//		for (auto c : channel) {
//			cu.slice(i++) = d.slice(c);
//		}
//		b.data_.push_back(cu);
//	}
//	b.shape_ = this->shape_;
//	return b;
//}

template<typename DType>
Blob<DType> Blob<DType>::sub_blob(const string& format) const
{
	vector<string> nchw_vec = string_split(format, ";");
	CHECK_EQ(nchw_vec.size(), 4);

	int num_beg, num_end, channel_beg, channel_end, height_beg, height_end, width_beg, width_end;
	vector<string> n_vec = string_split(nchw_vec[0], ":");
	vector<string> c_vec = string_split(nchw_vec[1], ":");
	vector<string> h_vec = string_split(nchw_vec[2], ":");
	vector<string> w_vec = string_split(nchw_vec[3], ":");
	
	if (n_vec.size() == 0) {
		num_beg = 0;
		num_end = this->shape_[0] - 1;
	}else if (n_vec.size() == 1) {
		num_beg = str_to_int(n_vec[0]);
		num_end = this->shape_[0] - 1;
		CHECK_LE(num_beg, num_end);
	}else {
		CHECK_EQ(n_vec.size(), 2);
		num_beg = n_vec[0].empty() ? 0 : str_to_int(n_vec[0]);
		num_end = str_to_int(n_vec[1]);
		CHECK_LE(num_beg, num_end);
		CHECK_LT(num_end, this->shape_[0]);	
	}

	if (c_vec.size() == 0) {
		channel_beg = 0;
		channel_end = this->shape_[1] - 1;
	}else if (c_vec.size() == 1) {
		channel_beg = str_to_int(c_vec[0]);
		channel_end = this->shape_[1] - 1;
		CHECK_LE(channel_beg, channel_end);
	}else {
		CHECK_EQ(c_vec.size(), 2);
		channel_beg = c_vec[0].empty() ? 0 : str_to_int(c_vec[0]);
		channel_end = str_to_int(c_vec[1]);
		CHECK_LE(channel_beg, channel_end);
		CHECK_LT(channel_end, this->shape_[1]);
	}

	if (h_vec.size() == 0) {
		height_beg = 0;
		height_end = this->shape_[2] - 1;
	}else if (h_vec.size() == 1) {
		height_beg = str_to_int(h_vec[0]);
		height_end = this->shape_[2] - 1;
		CHECK_LE(height_beg, height_end);
	}else {
		CHECK_EQ(h_vec.size(), 2);
		height_beg = h_vec[0].empty() ? 0 : str_to_int(h_vec[0]);
		height_end = str_to_int(h_vec[1]);
		CHECK_LE(height_beg, height_end);
		CHECK_LT(height_end, this->shape_[2]);
	}

	if (w_vec.size() == 0) {
		width_beg = 0;
		width_end = this->shape_[3] - 1;
	}else if (w_vec.size() == 1) {
		width_beg = str_to_int(w_vec[0]);
		width_end = this->shape_[3] - 1;
		CHECK_LE(width_beg, width_end);
	}else {
		CHECK_EQ(w_vec.size(), 2);
		width_beg = w_vec[0].empty() ? 0 : str_to_int(w_vec[0]);
		width_end = str_to_int(w_vec[1]);
		CHECK_LE(width_beg, width_end);
		CHECK_LT(width_end, this->shape_[3]);	
	}

	Blob<DType> b;
	for (int i = num_beg; i <= num_end; i++) {
		b.data_.push_back(this->data_[i].subcube(height_beg, width_beg, channel_beg, 
										height_end, width_end, channel_end));
	}
	b.shape_ = vector<int>{ num_end - num_beg + 1, channel_end - channel_beg + 1,
							height_end - height_beg + 1, width_end - width_beg + 1 };
	return b;
}

template<typename DType>
Blob<DType> Blob<DType>::sub_blob(const vector<vector<int>>& nchw) const
{
	CHECK_EQ(nchw.size(), 4);
	CHECK_LE(nchw[0].size(), 2);
	CHECK_LE(nchw[1].size(), 2);
	CHECK_LE(nchw[2].size(), 2);
	CHECK_LE(nchw[3].size(), 2);

	int num_beg, num_end, channel_beg, channel_end, height_beg, height_end, width_beg, width_end;
	if (nchw[0].size() == 0) {
		num_beg = 0;
		num_end = this->shape_[0] - 1;
	}
	else if (nchw[0].size() == 1) {
		num_beg = nchw[0][0];
		num_end = num_beg;
		CHECK_LT(num_end, this->shape_[0]);
	}
	else {
		num_beg = nchw[0][0];
		num_end = nchw[0][1];
		CHECK_LE(num_beg, num_end);
		CHECK_LT(num_end, this->shape_[0]);
	}

	if (nchw[1].size() == 0) {
		channel_beg = 0;
		channel_end = this->shape_[1] - 1;
	}
	else if (nchw[1].size() == 1) {
		channel_beg = nchw[1][0];
		channel_end = channel_beg;
		CHECK_LT(channel_end, this->shape_[1]);
	}
	else {
		channel_beg = nchw[1][0];
		channel_end = nchw[1][1];
		CHECK_LE(channel_beg, channel_end);
		CHECK_LT(channel_end, this->shape_[1]);
	}

	if (nchw[2].size() == 0) {
		height_beg = 0;
		height_end = this->shape_[2] - 1;
	}
	else if (nchw[2].size() == 1) {
		height_beg = nchw[2][0];
		height_end = height_beg;
		CHECK_LT(height_end, this->shape_[2]);
	}
	else {
		height_beg = nchw[2][0];
		height_end = nchw[2][1];
		CHECK_LE(height_beg, height_end);
		CHECK_LT(height_end, this->shape_[2]);
	}

	if (nchw[3].size() == 0) {
		width_beg = 0;
		width_end = this->shape_[3] - 1;
	}
	else if (nchw[3].size() == 1) {
		width_beg = nchw[3][0];
		width_end = width_beg;
		CHECK_LT(width_end, this->shape_[3]);
	}
	else {
		width_beg = nchw[3][0];
		width_end = nchw[3][1];
		CHECK_LE(width_beg, width_end);
		CHECK_LT(width_end, this->shape_[3]);
	}

	Blob<DType> b;
	for (int i = num_beg; i <= num_end; i++) {
		b.data_.push_back(this->data_[i].subcube(height_beg, width_beg, channel_beg,
			height_end, width_end, channel_end));
	}
	b.shape_ = vector<int>{ num_end - num_beg + 1, channel_end - channel_beg + 1,
							height_end - height_beg + 1, width_end - width_beg + 1 };
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::sub_blob_inplace(const vector<vector<int>>& nchw)
{
	CHECK_EQ(nchw.size(), 4);
	CHECK_LE(nchw[0].size(), 2);
	CHECK_LE(nchw[1].size(), 2);
	CHECK_LE(nchw[2].size(), 2);
	CHECK_LE(nchw[3].size(), 2);

	int num_beg, num_end, channel_beg, channel_end, height_beg, height_end, width_beg, width_end;
	if (nchw[0].size() == 0) {
		num_beg = 0;
		num_end = this->shape_[0] - 1;
	}
	else if (nchw[0].size() == 1) {
		num_beg = nchw[0][0];
		num_end = num_beg;
		CHECK_LT(num_end, this->shape_[0]);
	}
	else {
		num_beg = nchw[0][0];
		num_end = nchw[0][1];
		CHECK_LE(num_beg, num_end);
		CHECK_LT(num_end, this->shape_[0]);
	}

	if (nchw[1].size() == 0) {
		channel_beg = 0;
		channel_end = this->shape_[1] - 1;
	}
	else if (nchw[1].size() == 1) {
		channel_beg = nchw[1][0];
		channel_end = channel_beg;
		CHECK_LT(channel_end, this->shape_[1]);
	}
	else {
		channel_beg = nchw[1][0];
		channel_end = nchw[1][1];
		CHECK_LE(channel_beg, channel_end);
		CHECK_LT(channel_end, this->shape_[1]);
	}

	if (nchw[2].size() == 0) {
		height_beg = 0;
		height_end = this->shape_[2] - 1;
	}
	else if (nchw[2].size() == 1) {
		height_beg = nchw[2][0];
		height_end = height_beg;
		CHECK_LT(height_end, this->shape_[2]);
	}
	else {
		height_beg = nchw[2][0];
		height_end = nchw[2][1];
		CHECK_LE(height_beg, height_end);
		CHECK_LT(height_end, this->shape_[2]);
	}

	if (nchw[3].size() == 0) {
		width_beg = 0;
		width_end = this->shape_[3] - 1;
	}
	else if (nchw[3].size() == 1) {
		width_beg = nchw[3][0];
		width_end = width_beg;
		CHECK_LT(width_end, this->shape_[3]);
	}
	else {
		width_beg = nchw[3][0];
		width_end = nchw[3][1];
		CHECK_LE(width_beg, width_end);
		CHECK_LT(width_end, this->shape_[3]);
	}

	for (int i = num_beg; i <= num_end; i++) {
		this->data_[i] = this->data_[i].subcube(height_beg, width_beg, channel_beg,
			height_end, width_end, channel_end);
	}
	this->shape_.clear();
	this->shape_ = vector<int>{ num_end - num_beg + 1, channel_end - channel_beg + 1,
		height_end - height_beg + 1, width_end - width_beg + 1 };

	return *this;
}

template<typename DType>
DType Blob<DType>::operator()(const int num, const int channel, const int height, const int width) const
{
	CHECK_LT(num, this->shape_[0]);
	CHECK_LT(channel, this->shape_[1]);
	CHECK_LT(height, this->shape_[2]);
	CHECK_LT(width, this->shape_[3]);
	return this->data_[num](height, width, channel);
}

template<typename DType>
DType Blob<DType>::operator()(const vector<int>& shape) const
{
	CHECK_EQ(shape.size(), 4);
	CHECK_LT(shape[0], this->shape_[0]);
	CHECK_LT(shape[1], this->shape_[1]);
	CHECK_LT(shape[2], this->shape_[2]);
	CHECK_LT(shape[3], this->shape_[3]);
	return this->data_[shape[0]](shape[2], shape[3], shape[1]);
}

template<typename DType>
DType& Blob<DType>::at(const int num, const int channel, const int height, const int width)
{
	CHECK_LT(num, this->shape_[0]);
	CHECK_LT(channel, this->shape_[1]);
	CHECK_LT(height, this->shape_[2]);
	CHECK_LT(width, this->shape_[3]);
	return this->data_[num].at(height, width, channel);
}

template<typename DType>
DType& Blob<DType>::at(const vector<int>& shape)
{
	CHECK_EQ(shape.size(), 4);
	CHECK_LT(shape[0], this->shape_[0]);
	CHECK_LT(shape[1], this->shape_[1]);
	CHECK_LT(shape[2], this->shape_[2]);
	CHECK_LT(shape[3], this->shape_[3]);
	return this->data_[shape[0]].at(shape[2], shape[3], shape[1]);
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
vector<DType> Blob<DType>::sum_all_channel() const
{
	vector<DType> sum_vec;
	for (auto d : this->data_) {
		sum_vec.push_back(accu(d));
	}
	return sum_vec;
}

template<typename DType>
Blob<DType> Blob<DType>::sum() const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		Cube<DType> cu(1, 1, d.n_slices, fill::zeros);
		for (int c = 0; c < d.n_slices; c++) {
			vector<DType> x = { accu(d.slices(c, c)) };
			Mat<DType> m(x);
			//m.reshape(1, 1);		// (w, h)
			//cu.slice(c) = m.t();	// (h, w)
			cu.slice(c) = m;
		}
		b.data_.push_back(cu);
	}
	b.shape_ = vector<int>{ this->shape_[0], this->shape_[1], 1, 1 };
	return b;
}

template<typename DType>
vector<DType> Blob<DType>::ave_all_channel() const
{
	vector<DType> ave_vec;
	for (auto d : this->data_) {
		ave_vec.push_back(accu(d) / static_cast<DType>(d.n_elem));
	}
	return ave_vec;
}

template<typename DType>
Blob<DType> Blob<DType>::ave() const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		Cube<DType> cu(1, 1, d.n_slices, fill::zeros);
		for (int c = 0; c < d.n_slices; c++) {
			vector<DType> x = { accu(d.slices(c,c)) / static_cast<DType>(d.n_elem_slice) };
			Mat<DType> m(x);
			//m.reshape(1, 1);		// (w, h)
			//cu.slice(c) = m.t();	// (h, w)
			cu.slice(c) = m;
		}
		b.data_.push_back(cu);
	}
	b.shape_ = vector<int>{ this->shape_[0], this->shape_[1], 1, 1 };
	return b;
}

template<typename DType>
vector<DType> Blob<DType>::max_all_channel() const
{
	vector<DType> max_vec;
	for (auto d : this->data_) {
		max_vec.push_back(d.max());
	}
	return max_vec;
}

template<typename DType>
Blob<DType> Blob<DType>::max_along_dim(int dim) const
{
	CHECK_GT(dim, 0);
	CHECK_LT(dim, 4);

	Blob<DType> b;
	switch (dim) {
	case 1:
		for (auto d : this->data_) {
			if (dim == 1) {
				Mat<DType> m(d.n_rows, d.n_cols, fill::zeros);
				for (int r = 0; r < d.n_rows; r++) {
					for (int c = 0; c < d.n_cols; c++) {
						m.at(r, c) = d.tube(r, c).max();
					}
				}
				Cube<DType> cu(d.n_rows, d.n_cols, d.n_slices, fill::zeros);
				for (int c = 0; c < d.n_slices; c++) {
					cu.slice(c) = m;
				}
				b.data_.push_back(cu);
			}
		}
		b.shape_ = this->shape_;
		break;

	case 2:
		for (auto d : this->data_) {
			Cube<DType> cu(1, d.n_cols, d.n_slices, fill::zeros);
			for (int c = 0; c < d.n_slices; c++) {
				Mat<DType> m(1, d.n_cols, fill::zeros);
				for (int w = 0; w < d.n_cols; w++) {
					m.at(0, w) = d.slice(c).col(w).max();
				}
				cu.slice(c) = m;
			}
			b.data_.push_back(cu);
		}
		b.shape_ = vector<int>{ this->shape_[0],  this->shape_[1], 1, this->shape_[3] };
		break;
	case 3:
		for (auto d : this->data_) {
			Cube<DType> cu(d.n_rows, 1, d.n_slices, fill::zeros);
			for (int c = 0; c < d.n_slices; c++) {
				Mat<DType> m(d.n_rows, 1, fill::zeros);
				for (int h = 0; h < d.n_rows; h++) {
					m.at(h, 0) = d.slice(c).row(h).max();
				}
				cu.slice(c) = m;
			}
			b.data_.push_back(cu);
		}
		b.shape_ = vector<int>{ this->shape_[0],  this->shape_[1], this->shape_[2], 1 };
		break;
	}	
	return b;
}

template<typename DType>
Blob<DType> Blob<DType>::max() const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		Cube<DType> cu(1, 1, d.n_slices, fill::zeros);
		for (int c = 0; c < d.n_slices; c++) {
			vector<DType> x = { d.slice(c).max() };
			Mat<DType> m(x);
			//m.reshape(1, 1);		// (w, h)
			//cu.slice(c) = m.t();	// (h, w)
			cu.slice(c) = m;
		}
		b.data_.push_back(cu);
	}
	b.shape_ = vector<int>{ this->shape_[0], this->shape_[1], 1, 1 };
	return b;
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
	b.shape_[1] += rhs.shape_[1];
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
	this->shape_[1] += rhs.shape_[1];
	return *this;
}

template<typename DType>
Blob<DType>& Blob<DType>::transpose()
{
	for (auto &d : this->data_) {
		Cube<DType> cu(d.n_cols, d.n_rows, d.n_slices, fill::zeros);
		for (int c = 0; c < d.n_slices; c++) {	
			cu.slice(c) = d.slice(c).t();
		}
		d.reset();
		d = cu;
	}
	this->shape_ = vector<int>{this->shape_[0], this->shape_[1], this->shape_[3], this->shape_[2]};
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
Blob<DType> Blob<DType>::operator+(const DType scalar) const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		b.data_.push_back(d + scalar);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator+=(const DType scalar)
{
	for (auto &d : this->data_) {
		d += scalar;
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
Blob<DType> Blob<DType>::operator-(const DType scalar) const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		b.data_.push_back(d - scalar);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator-=(const DType scalar)
{
	for (auto &d : this->data_) {
		d -= scalar;
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
		Cube<DType> cu(d.n_rows, d.n_cols, d.n_slices, fill::zeros);
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
Blob<DType> Blob<DType>::operator*(const DType scalar) const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		b.data_.push_back(d * scalar);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator*=(const DType scalar)
{
	for (auto &d : this->data_) {
		d *= scalar;
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
		Cube<DType> cu(d.n_rows, d.n_cols, d.n_slices, fill::zeros);
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
		Cube<DType> cu(d.n_rows, d.n_cols, d.n_slices, fill::zeros);
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
Blob<DType> Blob<DType>::operator/(const DType scalar) const
{
	Blob<DType> b;
	for (auto d : this->data_) {
		b.data_.push_back(d / scalar);
	}
	b.shape_ = this->shape_;
	return b;
}

template<typename DType>
Blob<DType>& Blob<DType>::operator/=(const DType scalar)
{
	for (auto &d : this->data_) {
		d /= scalar;
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

//template<typename DType>
//void Blob<DType>::scale(const DType scale_factor) {
//	for (auto &d : this->data_) {
//		d = scale_factor * d;
//	}
//}

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
//	}
//	return *this;
//}

//explicit instantiate class Blob for declaration and implementation separation
template class Blob<float>;
template class Blob<double>;

}	//namespace caffe