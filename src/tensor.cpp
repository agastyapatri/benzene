#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <random> 
#include <cmath>
#include <openblas/cblas.h> 
#include <stdexcept>
#include <iomanip>

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

void tensor::unsqueeze(){
	_shape.insert(_shape.begin(), 1);
	_ndim = _shape.size(); 
	this->compute_strides();
}


void tensor::compute_strides(){
	_strides.resize(_ndim, 1);
	for(int i = _ndim - 2; i >= 0; i--){
		_strides[i] = _strides[i + 1] * _shape[i + 1];
	}
}

tensor tensor::eye(const i32 size){
	tensor out({size, size});
	for(i32 i = 0; i < size; i++){
		out.at({i, i}) = 1.0f;
	}
	return out;
}

bool tensor::is_contiguous() const{
	i32 expected_stride = 1;
	for(i32 i = _ndim-1; i >= 0; i--){
		if(expected_stride != _strides[i])	return false; 
		expected_stride *= _shape[i];
	}
	return true;
}

// TODO: Finish this
tensor tensor::contiguous() const {
	if(is_contiguous())	return *this;
	
	tensor out = *this;
	vi32 correct_strides(_ndim , 1); 
	for(int i = _ndim - 2; i >= 0; i--){
		correct_strides[i] = correct_strides[i + 1] * _shape[i + 1];
	}
	out._strides = correct_strides;
	for(u64 i = 0; i < out._numel; i++){
		vi32 coords = out.flat_idx_to_coord(i);
		i32 source_offset = 0; 
		for(i32 d = 0; d < _ndim; d++)	source_offset += coords[d]*this->_strides[d];
		out._data[i] = this->_data[source_offset];
	}
	return out;
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

bool tensor::operator==(const tensor& other) const {
	if(this->_shape != other._shape) return false; 
	for(u64 i = 0; i < this->_numel; i++){
		if(this->_data[i] != other._data[i])
			return false;
	}
	return true; 
}

tensor tensor::operator[](const vi32 idxs) const {
	return tensor({1}, {this->at(idxs)});
}




bool tensor::operator!=(const tensor& other) const {
	return !(*this == other);
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
			os << std::setw(10) << std::fixed << std::setprecision(7) << t.data()[offset + i * t.strides()[dim]];
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
	#pragma omp parallel for
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



tensor tensor::operator+(f32 scalar) const{
	tensor out(this->_shape);
	#pragma omp parallel for
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] + scalar;
	return out;
}

tensor tensor::operator-(f32 scalar) const{
	tensor out(_shape);
	#pragma omp parallel for
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] - scalar;
	return out;
}

tensor tensor::pow(const f32 exponent) const{
	tensor out(this->_shape);
	#pragma omp parallel for
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = std::pow(this->_data[i], exponent);
	return out;
} 

void tensor::pow_(const f32 exponent) {
	#pragma omp parallel for
	for(u64 i = 0; i < _numel; i++)
		this->_data[i] = std::pow(this->_data[i], exponent);
} 



//unified helper for broadcasted +, -
tensor tensor::arithmetic(const tensor& inp2, f32 op) const {
	// assert(this->_shape == other._shape);
	auto outshape = broadcast_shapes(this->_shape, inp2._shape);
	if(!outshape){
		throw std::runtime_error("Tensors are not broadcast compatible.");
	}
	tensor out(outshape.value());

	//	if both shapes are equal, simply add elementwise.
	if(this->_shape == inp2._shape){
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		cblas_saxpy(
			out._numel, 
			op,
			inp2._data.data(), 1, 
			out._data.data(), 1
		);
		return out;
	}

	// Generalized (..., N) + (N) case
	if (inp2._ndim == 1 && this->_shape.back() == inp2._shape[0]) {
		i32 N = inp2._shape[0];
		i32 total_rows = this->_numel / N; 
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		#pragma omp parallel for
		for (i32 i = 0; i < total_rows; i++) {
			cblas_saxpy(
				N,
				op,
				inp2._data.data(), 1, 
				out._data.data() + (i * N), 1
			);
		}
		return out;
	}

	//	(....,  K, N) + (K, N)
	if(this->_ndim > 2 && inp2._ndim == 2){
		i32 N = inp2._shape[0]*inp2._shape[1];
		i32 num_iters = std::accumulate(this->_shape.begin(), this->_shape.end() - 2, 1, std::multiplies<i32>()); 
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		#pragma omp parallel for
		for(i32 i = 0; i < num_iters; i++){
			cblas_saxpy(
				N, 
				op, 
				inp2._data.data(), 1, 
				out._data.data() + (i*N), 1
			);
		}
		return out;
	}

	if(inp2._numel == 1){
		float scalar = op * inp2._data[0];
		#pragma omp parallel for
		for(u64 i = 0; i < _numel; i++)
			out._data[i] = this->_data[i] + scalar;
		return out;
	}

	throw std::runtime_error("Broacasting pattern not currently handled by Benzene");

}







tensor tensor::operator+(const tensor& other) const {
	return this->arithmetic(other, 1.0f);
	
}

tensor tensor::operator-(const tensor& other) const {
	return this->arithmetic(other, -1.0f);
}

tensor tensor::operator*(const tensor& other) const {
	auto outshape = broadcast_shapes(this->_shape, other._shape);
	if(!outshape){
		throw std::runtime_error("Tensors are not broadcast compatible.");
	}
	tensor out(outshape.value());

	if(this->_shape == other._shape){
		for(u64 i = 0; i < this->_numel; i++)
			out._data[i] = this->_data[i] * other._data[i];
		return out;
	}

	// Generalized (..., N) + (N) case
	if (other._ndim == 1 && this->_shape.back() == other._shape[0]) {
		i32 N = other._shape[0];
		#pragma omp parallel for
		for(u64 i = 0; i < _numel; i++)
			out._data[i] = this->_data[i] * other._data[i % N];
		return out;
	}

	//	(M, N, K ...) * (1)
	if(other._numel == 1){
		tensor out(this->_shape);
		float scalar = other._data[0];
		#pragma omp parallel for
		for(u64 i = 0; i < _numel; i++)
			out._data[i] = this->_data[i] * scalar;
		return out;
	}

	//	(M, N, K) * (N, K)
	if(this->_ndim == 3 && other._ndim == 2){
		i32 N = other._shape[0]*other._shape[1];
		i32 num_iters = this->_shape[0];
		#pragma omp parallel for
		for(i32 i = 0; i < num_iters; i++){
			for(i32 j = 0; j < N; j++){
				out._data[i*N + j] = this->_data[i*N + j] * other._data[j]; 
			}
		}
		return out;

	
	}


	//	TODO: handle the generic case. 
	//	Proper output has to be handled, currently returns 0
	throw std::runtime_error("Broacasting pattern not currently handled by Benzene");
}


tensor tensor::operator/(const tensor& other) const {
	auto outshape = broadcast_shapes(this->_shape, other._shape);
	if(!outshape){
		throw std::runtime_error("Tensors are not broadcast compatible.");
	}
	tensor out(outshape.value());

	if(this->_shape == other._shape){
		for(u64 i = 0; i < this->_numel; i++)
			out._data[i] = this->_data[i] / other._data[i];
		return out;
	}

	// Generalized (..., N) + (N) case
	if (other._ndim == 1 && this->_shape.back() == other._shape[0]) {
		i32 N = other._shape[0];
		#pragma omp parallel for
		for(u64 i = 0; i < _numel; i++)
			out._data[i] = this->_data[i] / other._data[i % N];
		return out;
	}

	//	(M, N, K ...) * (1)
	if(other._numel == 1){
		tensor out(this->_shape);
		float scalar = other._data[0];
		#pragma omp parallel for
		for(u64 i = 0; i < _numel; i++)
			out._data[i] = this->_data[i] / scalar;
		return out;
	}

	//	(M, N, K) * (N, K)
	if(this->_ndim == 3 && other._ndim == 2){
		i32 N = other._shape[0]*other._shape[1];
		i32 num_iters = this->_shape[0];
		#pragma omp parallel for
		for(i32 i = 0; i < num_iters; i++){
			for(i32 j = 0; j < N; j++){
				out._data[i*N + j] = this->_data[i*N + j] / other._data[j]; 
			}
		}
		return out;


	}

	//	TODO: handle the generic case. 
	//	Proper output has to be handled, currently returns 0
	throw std::runtime_error("Broacasting pattern not currently handled by Benzene");
}



tensor tensor::operator*(float scalar) const {
	tensor out(_shape);
	#pragma omp parallel for
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] * scalar;
	return out;
}

tensor tensor::operator/(float scalar) const {
	tensor out(_shape);
	#pragma omp parallel for
	for(u64 i = 0; i < _numel; i++)
		out._data[i] = _data[i] / scalar;
	return out;
}

tensor tensor::randu(vi32 shape, f32 low, f32 high){
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

tensor tensor::randn_he(vi32 shape, u32 fan_in){
	return tensor::rand_normal(shape, 0, std::sqrt(2.0f / fan_in));
}

tensor tensor::randn_xavier(vi32 shape, u32 fan_in, u32 fan_out){
	return tensor::rand_normal(shape, 0, std::sqrt(2.0f / (fan_in + fan_out)));
}

tensor tensor::randu_he(vi32 shape, u32 fan_in){
	return tensor::randu(shape, -std::sqrt(2.0f / fan_in), std::sqrt(2.0f / fan_in));
}

tensor tensor::randu_xavier(vi32 shape, u32 fan_in, u32 fan_out){
	return tensor::randu(shape, -std::sqrt(2.0f / (fan_in + fan_out)), std::sqrt(2.0f / (fan_in + fan_out)));
}



void tensor::log_(){
    #pragma omp parallel for
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::log(this->_data[i]);
}

void tensor::exp_(){
    #pragma omp parallel for
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::exp(this->_data[i]);
}

void tensor::sin_(){
    #pragma omp parallel for
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::sin(this->_data[i]);
}

void tensor::cos_(){
    #pragma omp parallel for
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::cos(this->_data[i]);
}

void tensor::tanh_(){
    #pragma omp parallel for
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = std::tanh(this->_data[i]);
}

void tensor::sigmoid_(){
    #pragma omp parallel for
	for(u64 i = 0; i < this->_numel; i++)
		this->_data[i] = (1 / (1 + std::exp(-this->_data[i])));
}

void tensor::relu_(){
    #pragma omp parallel for
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




tensor tensor::var() const {
	u32 axis = _ndim - 1;
	vi32 outshape = reduced_shape(axis);
	tensor out(outshape);
	i32 axis_dim = _shape[axis];
	u64 num_rows = _numel / axis_dim;
	for(u64 r = 0; r < num_rows; ++r){
		u64 offset = r * axis_dim; 
		const f32* in_ptr = _data.data() + offset; 
		f32 mean = 0.0f; 
		for(i32 i = 0; i < axis_dim; ++i)	mean += in_ptr[i];
		mean /= axis_dim;
		f32 var = 0.0f; 
		for(i32 i = 0; i < axis_dim; ++i){
			f32 diff = in_ptr[i] - mean; 
			var += diff * diff; 
		}
		var /= axis_dim; 
		out._data[r] = var;
	}
	return out;
}




tensor tensor::std_dev() const {
	tensor out = var(); 
	for(u64 i = 0; i < out._numel; i++){
		out._data[i] = sqrtf(out._data[i]);
	}
	return out;
}






tensor matmul(const tensor& inp1, const tensor& inp2){
	tensor out;
	//	[M, n] x [N, O] = [M x O]
	if(inp1._ndim == 2 && inp2._ndim == 2){
		return mm(inp1, inp2);
	}
	//	[M, N] x [N] = [M]
	if(inp1._ndim == 2 && inp2._ndim == 1){
		return mv(inp1, inp2);
	}
	//	[N] x [N] = 1
	if(inp1._ndim == 1 && inp2._ndim == 1){
		return dot(inp1, inp2);
	}

	// [A, B, C] x [C, D] = [A, B, D]
	if(inp1._ndim == 3 && inp2._ndim == 2){
		assert(inp1._shape.back() == inp2._shape[0]);
		i32 A = inp1._shape[0];
		i32 B = inp1._shape[1];
		i32 M = A * B; 
		i32 K = inp1._shape[2];
		i32 N = inp2._shape[1];
		assert(K == inp2._shape[0]);
		i32 D = inp2._shape[1];
		tensor out({A, B, D});
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

	//	The general case; all dimensions up to the last two are flattened. 
	//	Eg: [B, H, M, K] x [B, H, K, N] = [B, H, M, N]	
	if(inp1._ndim >= 3 && inp2._ndim >= 3){
		for(i32 i = 0; i < inp1._ndim - 2; i++)
			assert(inp1._shape[i] == inp2._shape[i]);
		assert(inp1._shape[inp1._ndim - 1] == inp2._shape[inp2._ndim - 2]);
		i32 K = inp1._shape.back();
		i32 N = inp2._shape.back();
		i32 M = inp1._shape[inp1._ndim - 2];
		vi32 out_shape = inp1._shape;
		out_shape[out_shape.size() - 1] = N;
		tensor out(out_shape);
		u32 stride1 = M * K;
		u32 stride2 = K * N; 
		u32 stride3 = M * N;
		u32 batch_count = (u32)inp1._numel / stride1;
		const float* inp1_ptr = inp1._data.data(); 
		const float* inp2_ptr = inp2._data.data(); 
		float* out_ptr = out._data.data();
		#pragma omp parallel for
		for(u64 i = 0; i < batch_count; i++){
			cblas_sgemm(
				CblasRowMajor, CblasNoTrans, CblasNoTrans,
				M, N, K, 
				1.0f, 
				inp1_ptr + (i * stride1), K, 
				inp2_ptr + (i * stride2), N, 
				0.0f, 
				out_ptr + (i * stride3), N
			);
		}
		return out;
	} 
	
	throw std::runtime_error("Provided bz::tensor dimensions not handled yet");
}




tensor mm(const tensor& inp1, const tensor& inp2){
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

tensor dot(const tensor& inp1, const tensor& inp2){
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



tensor mv(const tensor& inp1, const tensor& inp2){
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



tensor leakyrelu(const tensor& t, f32 negative_slope) {
	tensor out(t._shape);
	#pragma omp parallel for
	for(u64 i = 0 ; i < t._numel; i++)
		out._data[i] = std::max(t._data[i], negative_slope * t._data[i]);
	return out;
}

tensor relu(const tensor& t){
	tensor out(t._shape);
	#pragma omp parallel for
	for(u64 i = 0 ; i < t._numel; i++)
		out._data[i] = std::max(t._data[i], 0.0f);
	return out;
}

tensor gelu(const tensor& t){
	tensor out(t._shape);
	#pragma omp parallel for
	for(u64 i = 0; i < t._numel; i++)
		out._data[i] = (0.5*t._data[i]) * (1 + std::tanh(SQRTTWOBYPI * (t._data[i] + 0.047715 * t._data[i] * t._data[i] * t._data[i])));

	return out;
}

tensor sigmoid(const tensor& t){
	tensor out(t._shape);
	#pragma omp parallel for
	for(u64 i = 0; i < t._numel; i++)
		out._data[i] = (1 / (1 + std::exp(-t._data[i])));
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
	out.compute_strides();

	for(bz::u64 i = 0; i < t._numel; i++){
		bz::vi32 original_loc = t.flat_idx_to_coord(i);
		bz::vi32 new_loc = original_loc;
		std::swap(new_loc[dim0], new_loc[dim1]);
		i32 offset = 0;
		for(bz::i32 j = 0; j < out._ndim; j++){
			offset += new_loc[j] * out._strides[j];
		}
		out._data[offset] = t._data[i];
	}

	return out;
}





void tensor::T(u32 dim0, u32 dim1){
	std::swap(this->_shape[dim0], this->_shape[dim1]);
	this->compute_strides();
	vf32 _old_data = this->_data;

	for(bz::u64 i = 0; i < this->_numel; i++){
		bz::vi32 original_loc = this->flat_idx_to_coord(i);
		bz::vi32 new_loc = original_loc;
		std::swap(new_loc[dim0], new_loc[dim1]);
		i32 offset = 0;
		for(bz::i32 j = 0; j < this->_ndim; j++){
			offset += new_loc[j] * this->_strides[j];
		}
		this->_data[offset] = _old_data[i];
	}
}


tensor reshape(const tensor& t, vi32 new_shape){
	u64 new_numel = 1;
	for(int i : new_shape) new_numel *= i;
	assert(new_numel == t._numel);
	tensor out; 
	out._shape = new_shape;
	out._ndim  = new_shape.size(); 
	out._numel = new_numel;
	out.compute_strides();
	out._data.resize(new_numel, 0.0f);
	std::copy(t._data.begin(), t._data.end(), out._data.begin());
	return out;
}


void tensor::reshape(vi32 new_shape){
	u64 new_numel = 1;
	for(int i : new_shape) new_numel *= i;
	assert(new_numel == this->_numel);
	this->_shape = new_shape;
	this->_ndim  = new_shape.size(); 
	this->_numel = new_numel;
	this->compute_strides();
}




tensor softmax(const tensor& t, i32 dim) {
	// 1. Handle negative dimensions (PyTorch style)
	if (dim < 0) dim += t.ndim();
	
	if (dim < 0 || dim >= t.ndim()) {
		throw std::runtime_error("Softmax: Dimension out of range.");
	}

	tensor out(t.shape());
	
	// 2. Calculate spatial components
	const vi32& shape = t.shape();
	i32 outer_size = 1;
	for (i32 i = 0; i < dim; ++i) outer_size *= shape[i];

	i32 axis_size = shape[dim];

	i32 inner_size = 1;
	for (i32 i = dim + 1; i < t.ndim(); ++i) inner_size *= shape[i];

	// 3. Perform Softmax
	for (i32 o = 0; o < outer_size; ++o) {
		for (i32 i = 0; i < inner_size; ++i) {
			// Determine the offset for this specific slice
			u64 base_idx = (o * axis_size * inner_size) + i;

			// Step A: Find Max for stability
			f32 max_val = t.data()[base_idx];
			for (i32 a = 1; a < axis_size; ++a) {
				f32 val = t.data()[base_idx + (a * inner_size)];
				if (val > max_val) max_val = val;
			}

			// Step B: Compute Exponentials and Sum
			f32 sum_exp = 0.0f;
			for (i32 a = 0; a < axis_size; ++a) {
				u64 current_idx = base_idx + (a * inner_size);
				f32 e = std::exp(t.data()[current_idx] - max_val);
				out._data[current_idx] = e;
				sum_exp += e;
			}

			// Step C: Normalize
			f32 inv_sum = 1.0f / (sum_exp + 1e-9f);
			for (i32 a = 0; a < axis_size; ++a) {
				out._data[base_idx + (a * inner_size)] *= inv_sum;
			}
		}
	}

	return out;
}

tensor rmsnorm(const tensor& t) {
	tensor out(t.shape());
	i32 last_dim = t.shape().back();
	u64 num_rows = t.numel() / last_dim;
	f32 eps = 1e-6f;

	for (u64 r = 0; r < num_rows; ++r) {
		u64 offset = r * last_dim;
		const f32* in_ptr = t.data().data() + offset;
		f32* out_ptr = out._data.data() + offset;

		// 1. Calculate Mean Square
		f32 ss = 0.0f; // sum of squares
		for (i32 i = 0; i < last_dim; ++i) {
			ss += in_ptr[i] * in_ptr[i];
		}
		f32 inv_rms = 1.0f / std::sqrt((ss / last_dim) + eps);

		// 2. Scale
		for (i32 i = 0; i < last_dim; ++i) {
			out_ptr[i] = in_ptr[i] * inv_rms;
		}
	}
	return out;
}



tensor layernorm(const tensor& t) {
	tensor out(t.shape());
	i32 last_dim = t.shape().back();
	u64 num_rows = t.numel() / last_dim;
	f32 eps = 1e-6f;

	for (u64 r = 0; r < num_rows; ++r) {
		u64 offset = r * last_dim;
		const f32* in_ptr = t.data().data() + offset;
		f32* out_ptr = out._data.data() + offset;

		// 1. Calculate Mean
		f32 mean = 0.0f;
		for (i32 i = 0; i < last_dim; ++i) mean += in_ptr[i];
		mean /= last_dim;

		// 2. Calculate Variance
		f32 var = 0.0f;
		for (i32 i = 0; i < last_dim; ++i) {
			f32 diff = in_ptr[i] - mean;
			var += diff * diff;
		}
		f32 inv_std = 1.0f / std::sqrt((var / last_dim) + eps);

		// 3. Normalize: (x - mean) * inv_std
		for (i32 i = 0; i < last_dim; ++i) {
			out_ptr[i] = (in_ptr[i] - mean) * inv_std;
		}
	}
	return out;
}

tensor tensor::gather(const tensor& indices, i32 dim) const {
	assert(dim <= _ndim);
	if(_ndim != 2){
		throw std::runtime_error("tensor::gather is currently only supported when _ndim == 2");
	}
	
	if(dim == 0){
		i32 nrows = indices._numel;
		i32 ncols = this->_shape[this->_ndim - 1];
		tensor out({nrows, ncols});
		for(i32 i = 0; i < nrows; i++){
			i32 current_index = indices._data[i];
			assert(current_index >= 0 && current_index <= _shape[_ndim - 2]);
			const float* source_row_start = _data.data() + (current_index * ncols);
			float* destination_row = out._data.data() + (i * ncols);
			std::copy(source_row_start, source_row_start + ncols, destination_row);
		}
		return out;
	}
	throw std::runtime_error("bz::tensor::gather only works for dim == 0");
}

tensor tensor::gather(const vi32 indices, i32 dim) const {
	assert(dim <= _ndim);
	if(_ndim != 2){
		throw std::runtime_error("tensor::gather is currently only supported when _ndim == 2");
	}
	if(dim == 0){
		i32 nrows = indices.size();
		i32 ncols = _shape[1];
		tensor out({(i32)nrows, (i32)ncols});
		for(i32 i = 0; i < nrows; i++){
			i32 current_index = indices.data()[i];
			assert(current_index >= 0 && current_index <= _shape[0]);
			const float* source_row_start = _data.data() + (current_index * ncols);
			float* destination_row = out._data.data() + (i * ncols);
			std::copy(source_row_start, source_row_start + ncols, destination_row);
		}
		return out;
	}
	throw std::runtime_error("bz::tensor::gather only works for dim == 0");
}


tensor unsqueeze(const tensor& t){
	tensor out;
	out._data.resize(t._data.size());
	std::copy(t._data.begin(), t._data.end(), out._data.begin());
	out._numel = t._numel;
	out._shape.resize(t._ndim + 1, 1);
	std::copy(t._shape.begin(), t._shape.end(), out._shape.begin() + 1);
	out._ndim = out._shape.size();
	out.compute_strides();
	return out;
}


tensor tensor::linspace(f32 start, f32 end, i32 num){
	tensor out({num});
	f32 res = (end - start) / (f32)num;
	for(i32 i = 0; i < num; i++)
		out._data[i] = start + i*res;
	return out;
}

tensor tensor::arange(f32 start, f32 end, f32 step){
	i32 num = (end - start) / step;
	tensor out({num});
	for(i32 i = 0; i < num; i++)
		out._data[i] = start + i*step;
	return out;
}







}
