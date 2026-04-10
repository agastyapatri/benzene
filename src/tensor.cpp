#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <random> 
#include <cmath>
#include <openblas/cblas.h> 

namespace bz{

tensor::tensor(vi32 shape){
	_shape = shape;
	_ndim  = shape.size(); 
	_numel = 1; 
	for(int i : shape) _numel *= i;
	_strides.resize(_ndim, 1);
	compute_strides();
	_data.resize(_numel, 0.0f);
}

void tensor::compute_strides(){
	for(int i = _ndim - 2; i >= 0; i--){
		_strides[i] = _strides[i + 1] * _shape[i + 1];
	}
}

tensor::tensor(vi32 shape, std::initializer_list<float> values) : tensor(shape){
	assert(values.size() == _numel);
	std::copy(values.begin(), values.end(), _data.begin());
}


float tensor::at(vi32 idxs) const {
	assert(idxs.size() == _numel);
	int offset = 0;
	for(int i = 0; i < _ndim; i++){
		assert(idxs[i] >= 0 && idxs[i] < _shape[i]);
		offset += idxs[i] * _strides[i];
	}
	return _data[offset];
}

float& tensor::at(vi32 idxs){
	assert(idxs.size() == _numel);
	int offset = 0;
	for(int i = 0; i < _ndim; i++){
		assert(idxs[i] >= 0 && idxs[i] < _shape[i]);
		offset += idxs[i] * _strides[i];
	}
	return _data[offset];
}


static inline void print_tensor_recursive(std::ostream& os, const tensor& t, int dim, int offset, int depth){
	if(dim == t.ndim() - 1){
		os << "[";
		for(int i = 0; i < t.shape()[dim]; i++){
			os << t.data()[offset + i * t.strides()[dim]];
			if(i < t.shape()[dim] - 1) os << ", ";
		}
		os << "]";
	} else {
		os << "[";
		for(int i = 0; i < t.shape()[dim]; i++){
			print_tensor_recursive(os, t, dim+1, offset + i*t.strides()[dim], depth + 1);
		if(i < t.shape()[dim] - 1){
				os << ",";
				int newlines = t.ndim() - 1 - dim;
				for(int n = 0; n < newlines; n++) os << "\n";
				for(int d = 0; d < depth + 1; d++) os << " ";
			}
		}
		os << "]";
	}
}

std::ostream& operator<<(std::ostream& os, const tensor& t){
	print_tensor_recursive(os, t, 0, 0, 0);
	os << "\n";
	return os;
}


void tensor::fill(float value){
	for(u64 i = 0; i < _numel; i++)
		_data[i] = value;
}

//
tensor tensor::ones(vi32 shape){
	tensor t(shape);
	t.fill(1.0f);
	return t;
}

tensor tensor::zeros(vi32 shape){
	tensor t(shape);
	return t;
}


tensor tensor::operator+(const tensor& other) const {
	assert(_shape == other._shape);
	tensor out(_shape);
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] + other._data[i];
	return out;
}

tensor tensor::operator+(f32 scalar) const{
	tensor out(_shape);
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] + scalar;
	return out;
}

tensor tensor::operator-(f32 scalar) const{
	tensor out(_shape);
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] - scalar;
	return out;
}

tensor tensor::pow(const u32 exponent) const{
	tensor out(this->_shape);
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = std::pow(this->_data[i], exponent);
	return out;
} 

void tensor::pow_(const u32 exponent) {
	for(u64 i = 0; i < _numel; i++)
		this->_data[i] = std::pow(this->_data[i], exponent);
} 



tensor tensor::operator-(const tensor& other) const {
	assert(_shape == other._shape);
	tensor out(_shape);
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] - other._data[i];
	return out;
}

tensor tensor::operator*(const tensor& other) const {
	assert(_shape == other._shape);
	tensor out(_shape);
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] * other._data[i];
	return out;
}

tensor tensor::operator*(float scalar) const {
	tensor out(_shape);
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] * scalar;
	return out;
}

tensor tensor::operator/(float scalar) const {
	tensor out(_shape);
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] / scalar;
	return out;
}

tensor tensor::rand_uniform(vi32 shape, f32 low, f32 high){
	tensor out(shape);
	static std::random_device rd; 
	static std::mt19937 engine(rd());
	std::uniform_real_distribution<f32> dist(low, high);
	std::generate(out._data.begin(), out._data.end(), [&](){
			return dist(engine);
	});
	return out;
}

tensor tensor::rand_normal(vi32 shape, f32 mean, f32 std){
	tensor out(shape);
	static std::random_device rd; 
	static std::mt19937 engine(rd());
	std::normal_distribution<f32> dist(mean, std);
	std::generate(out._data.begin(), out._data.end(), [&](){
			return dist(engine);
	});
	return out;
}

tensor tensor::randn(vi32 shape){
	return tensor::rand_normal(shape, 0, 1);
}

tensor tensor::rand_he(vi32 shape, f32 fan_in){
	return tensor::rand_normal(shape, 0, (2.0/fan_in));
}

tensor tensor::rand_xavier(vi32 shape, f32 fan_in, f32 fan_out){
	return tensor::rand_normal(shape, 0, (2.0 / (fan_in + fan_out)));
}

tensor tensor::log() const {
	tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](f32 x){
			return std::log(x);
	});
	return out;
}

tensor tensor::exp() const {
	tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](f32 x){
			return std::exp(x);
	});
	return out;
}

tensor tensor::sin() const {
	tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](f32 x){
			return std::sin(x);
	});
	return out;
}

tensor tensor::cos() const {
	tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](f32 x){
			return std::cos(x);
	});
	return out;
}

tensor tensor::tanh() const {
	tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](f32 x){
			return std::tanh(x);
	});
	return out;
}

void tensor::log_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::log(this->_data[i]);
}

void tensor::exp_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::exp(this->_data[i]);
}

void tensor::sin_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::sin(this->_data[i]);
}

void tensor::cos_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::cos(this->_data[i]);
}

void tensor::tanh_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::tanh(this->_data[i]);
}

tensor tensor::max (u32 axis) const {
	vi32 outshape;
	//	calculating the shape of the output; ignoring the reduced dim
	for(i32 i = 0; i < this->_ndim; i++){
		if(i != (i32)axis)
			outshape.push_back(this->_shape[i]); 
	}
	tensor out(outshape);
	out.fill((f32)(-FLT_MAX));

	vi32 coords(_ndim);
	for(u64 i = 0; i < this->_numel; i++){
		//	finding which element in the flat array belongs to which dimension
		u64 temp_idx = i; 
		for(i32 j = 0; j < this->_ndim; j++){
			coords[j] = temp_idx / this->_strides[j];
			temp_idx  = temp_idx % _strides[j];
		}
		u64 out_idx  = 0; 
		i32 out_dim_counter = 0; 
		for(i32 j = 0; j < this->_ndim; j++){
			if(j == (i32)axis)	continue;
			out_idx += coords[j] * out._strides[out_dim_counter];
			out_dim_counter++;
		}
		if(out._data[out_idx] <= this->_data[i])
			out._data[out_idx] = this->_data[i];
	}
	return out;
}

tensor tensor::min (u32 axis) const {
	vi32 outshape;
	//	calculating the shape of the output; ignoring the reduced dim
	for(i32 i = 0; i < this->_ndim; i++){
		if(i != (i32)axis)
			outshape.push_back(this->_shape[i]); 
	}
	tensor out(outshape);
	out.fill((f32)(FLT_MAX));

	vi32 coords(_ndim);
	for(u64 i = 0; i < this->_numel; i++){
		//	finding which element in the flat array belongs to which dimension
		u64 temp_idx = i; 
		for(i32 j = 0; j < this->_ndim; j++){
			coords[j] = temp_idx / this->_strides[j];
			temp_idx  = temp_idx % _strides[j];
		}
		u64 out_idx  = 0; 
		i32 out_dim_counter = 0; 
		for(i32 j = 0; j < this->_ndim; j++){
			if(j == (i32)axis)	continue;
			out_idx += coords[j] * out._strides[out_dim_counter];
			out_dim_counter++;
		}
		if(out._data[out_idx] >= this->_data[i])
			out._data[out_idx] = this->_data[i];
	}
	return out;
}

tensor tensor::sum (u32 axis) const {
	vi32 outshape;
	//	calculating the shape of the output; ignoring the reduced dim
	for(i32 i = 0; i < this->_ndim; i++){
		if(i != (i32)axis)
			outshape.push_back(this->_shape[i]); 
	}
	tensor out(outshape);
	vi32 coords(_ndim);
	for(u64 i = 0; i < this->_numel; i++){
		//	finding which element in the flat array belongs to which dimension
		u64 temp_idx = i; 
		for(i32 j = 0; j < this->_ndim; j++){
			coords[j] = temp_idx / this->_strides[j];
			temp_idx  = temp_idx % _strides[j];
		}
		u64 out_idx  = 0; 
		i32 out_dim_counter = 0; 
		for(i32 j = 0; j < this->_ndim; j++){
			if(j == (i32)axis)	continue;
			out_idx += coords[j] * out._strides[out_dim_counter];
			out_dim_counter++;
		}
		out._data[out_idx] += this->_data[i];
	}
	return out;
}

tensor tensor::mean (u32 axis) const {
	tensor out = this->sum(axis);
	f32 divisor = (f32)this->_shape[axis];
	return out / divisor;
}

tensor tensor::matmul(const tensor& other) const{
	assert((this->_ndim == 2) && (other._ndim == 2));
	assert(this->_shape[1] == other._shape[0]);
	i32 M = this->_shape[0];		// this->rows
	i32 N = other._shape[1];		// other col
	i32 K = this->_shape[1];		// this->cols
	tensor out({M, N});
	cblas_sgemm(
		CblasRowMajor, CblasNoTrans, CblasNoTrans, 
		M, N, K, 
		1.0f, 
		this->_data.data(), K, 
		other._data.data(), N,
		0.0f, 
		out._data.data(), N
	);
	return out;
}

tensor tensor::matmul(const tensor& inp1, const tensor& inp2){
	assert((inp1._ndim == 2) && (inp2._ndim == 2));
	assert(inp1._shape[1] == inp2._shape[0]);
	i32 M = inp1._shape[0];		// inp1 rows
	i32 N = inp2._shape[1];		// inp2 col
	i32 K = inp1._shape[1];		// inp1 cols
	tensor out({M, N});
	cblas_sgemm(
		CblasRowMajor, CblasNoTrans, CblasNoTrans, 
		M, N, K, 
		1.0f, 
		inp1._data.data(), K, 
		inp2._data.data(), N,
		0.0f, 
		out._data.data(), N
	);
	return out;
}

tensor tensor::mat_vec_mul(const tensor& inp1, const tensor& inp2){
	assert(inp1._ndim == 2);
	assert(inp2._ndim == 1);
	assert(inp1._shape[1] == inp2._shape[0]);
	i32 M = inp1._shape[0];
	i32 N = inp1._shape[1];
	tensor out({M});
	cblas_sgemv(
		CblasRowMajor, CblasNoTrans, 
		M, N, 
		1.0f, 
		inp1._data.data(), N, 
		inp2._data.data(), 1,
		0.0f, 
		out._data.data(),
		1
	);
	return out;

}
// tensor vec_mat_mul(const tensor& inp1, const tensor& inp2);
// tensor dot(const tensor& inp1, const tensor& inp2);




}
