#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <initializer_list>
#include <random> 
#include <cmath>
#include <openblas/cblas.h> 
#include <stdexcept>

namespace bz{

std::mt19937 tensor::rand_engine(std::random_device{}());

tensor::tensor(vi32 shape){
	_shape = shape;
	_ndim  = shape.size(); 
	_numel = 1; 
	for(int i : shape) _numel *= i;
	compute_strides();
	_data.resize(_numel, 0.0f);
}


void tensor::compute_strides(){
	_strides.resize(_ndim, 1);
	for(int i = _ndim - 2; i >= 0; i--){
		_strides[i] = _strides[i + 1] * _shape[i + 1];
	}
}

vi32 tensor::flat_idx_to_coord(u64 idx) const {
	vi32 coords(this->_ndim, 0);
	for(i32 j = 0; j < this->_ndim; j++){
		coords[j] = idx / this->_strides[j];
		idx  = idx % this->_strides[j];
	}
	return coords;
}


vi32 tensor::reduced_shape(u32 axis) const{
	vi32 outshape;
	for(i32 i = 0; i < this->_ndim; i++){
		if(i != (i32)axis)
			outshape.push_back(this->_shape[i]); 
	}
	return outshape;
}

std::optional<vi32> tensor::broadcast_shapes(const vi32& shape1, const vi32& shape2) const{
	//	the smaller tensor is stretched over the larger; out takes the shape of the larger. 
	i32 ndim1 = shape1.size(); 
	i32 ndim2 = shape2.size();
	i32 out_ndim = std::max(ndim1, ndim2);
	vi32 out_shape(out_ndim);

	for(i32 i = 0; i < out_ndim; i++){
		i32 d1 = (i < out_ndim - ndim1) ? 1 : shape1[i - (out_ndim - ndim1)];
		i32 d2 = (i < out_ndim - ndim2) ? 1 : shape2[i - (out_ndim - ndim2)];

		if(d1 == d2){
			out_shape[i] = d1; 
		} else if(d1 == 1){
			out_shape[i] = d2;
		} else if(d2 == 1){
			out_shape[i] = d1;
		} else{
			return std::nullopt;
		}
	}
	return out_shape;
}

tensor::tensor(vi32 shape, std::initializer_list<float> values) : tensor(shape){
	assert(values.size() == _numel);
	std::copy(values.begin(), values.end(), _data.begin());
}


float tensor::at(vi32 idxs) const {
	assert((i32)idxs.size() == _ndim);
	int offset = 0;
	for(int i = 0; i < _ndim; i++){
		assert(idxs[i] >= 0 && idxs[i] < _shape[i]);
		offset += idxs[i] * _strides[i];
	}
	return _data[offset];
}

float& tensor::at(vi32 idxs){
	assert((i32)idxs.size() == _ndim);
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
	// assert(this->_shape == other._shape);
	auto outshape = broadcast_shapes(this->_shape, other._shape);
	if(!outshape){
		throw std::runtime_error("Tensors are not broadcast compatible.");
	}
	tensor out(outshape.value());

	//	if both shapes are equal, simply add elementwise.
	if(this->_shape == other._shape){
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		cblas_saxpy(
			out._numel, 
			1.0f,
			other._data.data(), 1, 
			out._data.data(), 1
		);
		return out;
	}

	//	(M, N) + (N)
	if(this->_ndim == 2 && other._ndim  == 1 && this->_shape[1] == other._shape[0]){
		i32 M = this->_shape[0];
		i32 N = this->_shape[1];
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		for(i32 i = 0; i < M; i++){
			cblas_saxpy(
				N,
				1.0f,
				other._data.data(), 1, 
				out._data.data() + (i * N), 1
			);
		}
		return out;
	}
	if(other._numel == 1){
		tensor out(this->_shape);
		float scalar = other._data[0];
		for(u64 i = 0; i < _numel; i++)
			out._data[i] = this->_data[i] + scalar;
		return out;
	}


	//	TODO: handle the generic case. 
	//	Proper output has to be handled, currently returns 0
	throw std::runtime_error("Broacasting pattern not currently handled by Benzene");
}

tensor tensor::operator+(f32 scalar) const{
	tensor out(this->_shape);
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

tensor tensor::pow(const f32 exponent) const{
	tensor out(this->_shape);
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = std::pow(this->_data[i], exponent);
	return out;
} 

void tensor::pow_(const f32 exponent) {
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
	std::uniform_real_distribution<f32> dist(low, high);
	std::generate(out._data.begin(), out._data.end(), [&]()->f32{
			return dist(rand_engine);
	});
	return out;
}

tensor tensor::rand_normal(vi32 shape, f32 mean, f32 std){
	tensor out(shape);
	std::normal_distribution<f32> dist(mean, std);
	std::generate(out._data.begin(), out._data.end(), [&]()->f32{
			return dist(rand_engine);
	});
	return out;
}

tensor tensor::randn(vi32 shape){
	return tensor::rand_normal(shape, 0, 1);
}

tensor tensor::rand_he(vi32 shape, u32 fan_in){
	return tensor::rand_normal(shape, 0, std::sqrt(2.0f / fan_in));
}

tensor tensor::rand_xavier(vi32 shape, u32 fan_in, u32 fan_out){
	return tensor::rand_normal(shape, 0, std::sqrt(2.0f / (fan_in + fan_out)));
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

void tensor::sigmoid_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = (1 / (1 + std::exp(-this->_data[i])));
}

void tensor::relu_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = (this->_data[i] > 0) ? this->_data[i] : 0.0;
}

tensor tensor::max (u32 axis) const {
	vi32 outshape = reduced_shape(axis);
	tensor out(outshape);
	out.fill((f32)(-FLT_MAX));

	for(u64 i = 0; i < this->_numel; i++){
		//	finding which element in the flat array belongs to which dimension
		u64 temp_idx = i; 
		vi32 coords = flat_idx_to_coord(temp_idx);
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
	vi32 outshape = reduced_shape(axis);
	tensor out(outshape);
	out.fill((f32)(FLT_MAX));

	for(u64 i = 0; i < this->_numel; i++){
		//	finding which element in the flat array belongs to which dimension
		u64 temp_idx = i; 
		vi32 coords = flat_idx_to_coord(temp_idx);
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
	vi32 outshape = reduced_shape(axis);
	tensor out(outshape);
	for(u64 i = 0; i < this->_numel; i++){
		//	finding which element in the flat array belongs to which dimension
		u64 temp_idx = i; 
		vi32 coords = flat_idx_to_coord(temp_idx);
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

tensor matmul(const tensor& inp1, const tensor& inp2){
	assert((inp1._ndim == 2) && (inp2._ndim == 2));
	assert(inp1._shape[1] == inp2._shape[0]);
	i32 M = inp1._shape[0];		// inp1.rows
	i32 N = inp2._shape[1];		// inp2.col
	i32 K = inp1._shape[1];		// inp1.cols
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

tensor mat_vec_mul(const tensor& inp1, const tensor& inp2){
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

tensor dot(const tensor& inp1, const tensor& inp2){
	assert(inp1._ndim == 1 && inp2._ndim == 1);
	assert(inp1._shape[0] == inp2._shape[0]);
	i32 N = inp1._shape[0];
	tensor out({1});
	f32 res = cblas_sdot(
		N,
		inp1._data.data(), 1, 
		inp2._data.data(), 1 
	);
	out.at({0}) = res;
	return out;
}


tensor tensor::leakyrelu(f32 negative_slope) const{
	tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [negative_slope](f32 x)->f32{
		return (x > 0) ? x : -negative_slope;
	});
	return out;
}

tensor relu(const tensor& t){
	tensor out(t._shape);
	std::transform(t._data.begin(), t._data.end(), out._data.begin(), [](f32 x)->f32{
		return (x > 0) ? x : 0;
	});
	return out;
}

tensor sigmoid(const tensor& t){
	tensor out(t._shape);
	std::transform(t._data.begin(), t._data.end(), out._data.begin(), [](f32 x)->f32{
		return (1 / (1 + std::exp(x)));
	});
	return out;
}

tensor log(const tensor& t) {
	tensor out(t._shape);
	std::transform(t._data.begin(), t._data.end(), out._data.begin(), [](f32 x)->f32{
			return std::log(x);
	});
	return out;
}


tensor exp(const tensor& t) {
	tensor out(t._shape);
	std::transform(t._data.begin(), t._data.end(), out._data.begin(), [](f32 x)->f32{
			return std::exp(x);
	});
	return out;
}

tensor sin(const tensor& t){
	tensor out(t._shape);
	std::transform(t._data.begin(), t._data.end(), out._data.begin(), [](f32 x)->f32{
			return std::sin(x);
	});
	return out;
}

tensor cos(const tensor& t){
	tensor out(t._shape);
	std::transform(t._data.begin(), t._data.end(), out._data.begin(), [](f32 x)->f32{
			return std::cos(x);
	});
	return out;
}

tensor tanh(const tensor& t){
	tensor out(t._shape);
	std::transform(t._data.begin(), t._data.end(), out._data.begin(), [](f32 x)->f32{
			return std::tanh(x);
	});
	return out;
}

tensor transpose(const tensor& t, u32 dim0, u32 dim1){
	tensor out = t;
	std::swap(out._shape[dim0], out._shape[dim1]);
	std::swap(out._strides[dim0], out._strides[dim1]);
	return out;
}

tensor reshape(const tensor& t, vi32 newshape){
	tensor out;
	out._shape = newshape;
	out._ndim  = newshape.size(); 
	out._numel = 1; 
	for(i32 i : newshape) out._numel *= i;
	if(out._numel != t._numel){
		throw std::runtime_error("Error: Cannot reshape tensor of size " + std::to_string(t._numel) + " into " + std::to_string(out._numel));
	}
	out._strides.resize(out._ndim, 1);
	out.compute_strides();
	out._data = t._data;
	return out;
}


}
