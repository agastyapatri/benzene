#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include "bz_rand.hpp"

bz::tensor::tensor(std::vector<int> shape){
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

bz::tensor::tensor(std::vector<int> shape, std::initializer_list<float> values) : tensor(shape){
	assert(values.size() == _numel);
	std::copy(values.begin(), values.end(), _data.begin());
}


float bz::tensor::at(std::vector<int> idxs) const {
	assert(idxs.size() == _numel);
	int offset = 0;
	for(int i = 0; i < _ndim; i++){
		assert(idxs[i] >= 0 && idxs[i] < _shape[i]);
		offset += idxs[i] * _strides[i];
	}
	return _data[offset];
}

float& bz::tensor::at(std::vector<int> idxs){
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
	for(size_t i = 0; i < _numel; i++)
		_data[i] = value;
}

//
bz::tensor bz::tensor::ones(std::vector<int> shape){
	tensor t(shape);
	t.fill(1.0f);
	return t;
}

bz::tensor bz::tensor::zeros(std::vector<int> shape){
	tensor t(shape);
	return t;
}


bz::tensor bz::tensor::operator+(const tensor& other) const {
	assert(_shape == other._shape);
	tensor out(_shape);
	for(size_t i = 0; i < _numel; i++)
		out._data[i] = _data[i] + other._data[i];
	return out;
}

bz::tensor bz::tensor::operator-(const bz::tensor& other) const {
	assert(_shape == other._shape);
	bz::tensor out(_shape);
	for(size_t i = 0; i < _numel; i++)
		out._data[i] = _data[i] - other._data[i];
	return out;
}

bz::tensor bz::tensor::operator*(const tensor& other) const {
	assert(_shape == other._shape);
	tensor out(_shape);
	for(size_t i = 0; i < _numel; i++)
		out._data[i] = _data[i] * other._data[i];
	return out;
}

bz::tensor bz::tensor::operator*(float scalar) const {
	bz::tensor out(_shape);
	for(size_t i = 0; i < _numel; i++)
		out._data[i] = _data[i] * scalar;
	return out;
}

bz::tensor bz::tensor::operator/(float scalar) const {
	bz::tensor out(_shape);
	for(size_t i = 0; i < _numel; i++)
		out._data[i] = _data[i] / scalar;
	return out;
}


// bz::tensor bz::tensor::mean(int axis) const{
// 	assert(axis < _ndim);
// 	bz::vint out_shape;
// 	for(int i = 0; i < _ndim; i++){
// 		if(i == axis)	continue;
// 		out_shape.push_back(_shape[i]);
// 	}
// 	bz::tensor out(out_shape);
// 	return out;
// }
// tensor tensor::std (int dim) const{}
// tensor tensor::max (int dim) const{}
// tensor tensor::min (int dim) const{}
// tensor tensor::sum (int dim) const{}

