#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <random> 
#include <cmath>
// #include "bz_rand.hpp"

bz::tensor::tensor(bz::vi32 shape){
	_shape = shape;
	_ndim  = shape.size(); 
	_numel = 1; 
	for(int i : shape) _numel *= i;
	_strides.resize(_ndim, 1);
	compute_strides();
	_data.resize(_numel, 0.0f);
}

void bz::tensor::compute_strides(){
	for(int i = _ndim - 2; i >= 0; i--){
		_strides[i] = _strides[i + 1] * _shape[i + 1];
	}
}

bz::tensor::tensor(bz::vi32 shape, std::initializer_list<float> values) : tensor(shape){
	assert(values.size() == _numel);
	std::copy(values.begin(), values.end(), _data.begin());
}


float bz::tensor::at(bz::vi32 idxs) const {
	assert(idxs.size() == _numel);
	int offset = 0;
	for(int i = 0; i < _ndim; i++){
		assert(idxs[i] >= 0 && idxs[i] < _shape[i]);
		offset += idxs[i] * _strides[i];
	}
	return _data[offset];
}

float& bz::tensor::at(bz::vi32 idxs){
	assert(idxs.size() == _numel);
	int offset = 0;
	for(int i = 0; i < _ndim; i++){
		assert(idxs[i] >= 0 && idxs[i] < _shape[i]);
		offset += idxs[i] * _strides[i];
	}
	return _data[offset];
}


static inline void print_tensor_recursive(std::ostream& os, const bz::tensor& t, int dim, int offset, int depth){
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

std::ostream& bz::operator<<(std::ostream& os, const bz::tensor& t){
	print_tensor_recursive(os, t, 0, 0, 0);
	os << "\n";
	return os;
}


void bz::tensor::fill(float value){
	for(bz::u64 i = 0; i < _numel; i++)
		_data[i] = value;
}

//
bz::tensor bz::tensor::ones(bz::vi32 shape){
	tensor t(shape);
	t.fill(1.0f);
	return t;
}

bz::tensor bz::tensor::zeros(bz::vi32 shape){
	tensor t(shape);
	return t;
}


bz::tensor bz::tensor::operator+(const tensor& other) const {
	assert(_shape == other._shape);
	tensor out(_shape);
	for(bz::u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] + other._data[i];
	return out;
}

bz::tensor bz::tensor::operator-(const bz::tensor& other) const {
	assert(_shape == other._shape);
	bz::tensor out(_shape);
	for(bz::u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] - other._data[i];
	return out;
}

bz::tensor bz::tensor::operator*(const tensor& other) const {
	assert(_shape == other._shape);
	tensor out(_shape);
	for(bz::u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] * other._data[i];
	return out;
}

bz::tensor bz::tensor::operator*(float scalar) const {
	bz::tensor out(_shape);
	for(bz::u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] * scalar;
	return out;
}

bz::tensor bz::tensor::operator/(float scalar) const {
	bz::tensor out(_shape);
	for(bz::u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] / scalar;
	return out;
}

bz::tensor bz::tensor::rand_uniform(vi32 shape, f32 low, f32 high){
	bz::tensor out(shape);
	static std::random_device rd; 
	static std::mt19937 engine(rd());
	std::uniform_real_distribution<bz::f32> dist(low, high);
	std::generate(out._data.begin(), out._data.end(), [&](){
			return dist(engine);
	});
	return out;
}

bz::tensor bz::tensor::rand_normal(vi32 shape, f32 mean, f32 std){
	bz::tensor out(shape);
	static std::random_device rd; 
	static std::mt19937 engine(rd());
	std::normal_distribution<bz::f32> dist(mean, std);
	std::generate(out._data.begin(), out._data.end(), [&](){
			return dist(engine);
	});
	return out;
}

bz::tensor bz::tensor::randn(vi32 shape){
	return bz::tensor::rand_normal(shape, 0, 1);
}

bz::tensor bz::tensor::rand_he(bz::vi32 shape, bz::f32 fan_in){
	return bz::tensor::rand_normal(shape, 0, (2.0/fan_in));
}

bz::tensor bz::tensor::rand_xavier(bz::vi32 shape, bz::f32 fan_in, bz::f32 fan_out){
	return bz::tensor::rand_normal(shape, 0, (2.0 / (fan_in + fan_out)));
}

bz::tensor bz::tensor::log() const {
	bz::tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](bz::f32 x){
			return std::log(x);
	});
	return out;
}

bz::tensor bz::tensor::exp() const {
	bz::tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](bz::f32 x){
			return std::exp(x);
	});
	return out;
}

bz::tensor bz::tensor::sin() const {
	bz::tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](bz::f32 x){
			return std::sin(x);
	});
	return out;
}

bz::tensor bz::tensor::cos() const {
	bz::tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](bz::f32 x){
			return std::cos(x);
	});
	return out;
}

bz::tensor bz::tensor::tanh() const {
	bz::tensor out(this->_shape);
	std::transform(this->_data.begin(), this->_data.end(), out._data.begin(), [](bz::f32 x){
			return std::tanh(x);
	});
	return out;
}

void bz::tensor::log_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::log(this->_data[i]);
}

void bz::tensor::exp_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::exp(this->_data[i]);
}

void bz::tensor::sin_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::sin(this->_data[i]);
}

void bz::tensor::cos_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::cos(this->_data[i]);
}

void bz::tensor::tanh_(){
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::tanh(this->_data[i]);
}










