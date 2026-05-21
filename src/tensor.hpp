#ifndef BZ_TENSOR_HPP
#define BZ_TENSOR_HPP

#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <vector>
#include <initializer_list>

#define DEGREES_TO_RADIANS(deg)	(deg * std::numbers::pi / 180)
#define RADIANS_TO_DEGREES(rad) (rad * 180 / std::numbers::pi)
#define SQRTTWOBYPI 			(std::sqrt(2.0 / std::numbers::pi))

namespace bz{

typedef uint32_t u32; 
typedef uint64_t u64; 
typedef int32_t  i32; 
typedef int64_t  i64; 
typedef float  	 f32; 
typedef double   f64; 
typedef std::vector<uint32_t> vu32; 
typedef std::vector<uint64_t> vu64; 
typedef std::vector<int32_t>  vi32; 
typedef std::vector<int64_t>  vi64; 
typedef std::vector<float> 	  vf32; 
typedef std::vector<double>   vf64; 

class tensor; 
tensor relu(const tensor& t);
tensor leakyrelu(const tensor& t, f32 negative_slope);
tensor gelu(const tensor& t);
tensor sigmoid(const tensor& t);
tensor log(const tensor& t);
tensor exp(const tensor& t);
tensor sin(const tensor& t);
tensor cos(const tensor& t);
tensor tanh(const tensor& t);
tensor softmax(const tensor& t, i32 dim);
tensor rmsnorm(const tensor& t);
tensor layernorm(const tensor& t);
tensor unsqueeze(const tensor& t);
tensor transpose(const tensor& t, u32 dim0, u32 dim1);
tensor reshape(const tensor& t, vi32 new_shape);


tensor matmul(const tensor& inp1, const tensor& inp2);
tensor dot(const tensor& inp1, const tensor& inp2);
tensor mm(const tensor& inp1, const tensor& inp2);
tensor mv(const tensor& inp1, const tensor& inp2);


class tensor{
	vf32 _data;
	vi32 _shape;
	vi32 _strides;
	u64  _numel;
	i32  _ndim;

	//	calculates the strides of the tensor from the n-dim shape vector
	void compute_strides();

	//	given the location of an element in the flat data vector, returns the true location.
	vi32 flat_idx_to_coord(u64 idx) const;

	// calculates the reduced dimesions for the output of tensor reductions 
	vi32 reduced_shape(u32 axis) const;

	//	checks if another tensor is compatible for broadcasted arithmetic according to
	//	the NumPy broadcasting rules.
	std::optional<vi32> broadcast_shapes(const vi32& s1, const vi32& s2) const;

	//	initiating a  random engine
	static std::mt19937 rand_engine;

	tensor arithmetic(const tensor& inp2, f32 op) const ; 

public: 
	tensor(vi32 shape); 
	tensor(vi32 shape, std::initializer_list<float> values); 
	tensor() = default;


	bool is_contiguous() const;
	tensor contiguous() const;
	//	seeding the RNG 
	static void manual_seed(u32 seed){rand_engine.seed(seed);}

	//	accessors 
	f32  at(vi32 idxs) const;
	f32& at(vi32 idxs);
	auto begin() {return _data.begin();}
	auto end()   {return _data.end();}




	const vf32& data()    const	{return _data;}
	const vi32& shape()   const {return _shape;}
	const vi32& strides() const {return _strides;}
	u64  numel()   const {return _numel; }
	i32  ndim()    const {return _ndim;}

	//	overloaded operators
	bool   operator==(const tensor& other) const;
	bool   operator!=(const tensor& other) const;
	tensor operator+(const tensor& other) const;
	tensor operator-(const tensor& other) const;
	tensor operator*(const tensor& other) const;
	tensor operator/(const tensor& other) const;
	tensor operator*(f32 scalar) const;
	tensor operator/(f32 scalar) const;
	tensor operator+(f32 scalar) const;

	tensor operator[](const vi32 idxs) const ;
	tensor operator-(f32 scalar) const;
	tensor pow(const f32 exponent) const; 
	void   pow_(const f32 exponent);
	friend std::ostream& operator<<(std::ostream& os, const tensor& t);

	// factory
	void   fill(f32 value);
	static tensor zeros(vi32 shape);
	static tensor ones(vi32 shape);
	static tensor rand_normal (vi32 shape, f32 mean, f32 std);
	static tensor randn(vi32 shape);
	static tensor randu(vi32 shape, f32 low = 0.0f, f32 high = 1.0f);
	static tensor randn_he 	  (vi32 shape, u32 fan_in);
	static tensor randn_xavier (vi32 shape, u32 fan_in, u32 fan_out);
	static tensor randu_he 	  (vi32 shape, u32 fan_in);
	static tensor randu_xavier (vi32 shape, u32 fan_in, u32 fan_out);
	static tensor eye(const i32 size);
	static tensor linspace(f32 start, f32 end, i32 num);
	static tensor arange(f32 start, f32 end, f32 step);

	//	transcendentals; inplace
	void log_();
	void exp_();
	void sin_();
	void cos_();
	void tanh_();
	void relu_(); 
	void sigmoid_();


	//	reductions
	tensor mean(u32 axis) const;
	tensor max (u32 axis) const;
	tensor min (u32 axis) const;
	tensor sum (u32 axis) const;
	tensor var () const;
	tensor std_dev () const;



	//	tensor tensor multiplications
	friend tensor matmul(const tensor& inp1, const tensor& inp2);
	friend tensor mm(const tensor& inp1, const tensor& inp2);
	friend tensor mv(const tensor& inp1, const tensor& inp2);
	friend tensor dot(const tensor& inp1, const tensor& inp2);



	//	bz namespace functions; all return new tensors. 

	friend tensor operator*(f32 scalar, const tensor& t);
	friend tensor relu(const tensor& t);
	friend tensor leakyrelu(const tensor& t, f32 negative_slope);
	friend tensor gelu(const tensor& t);
	friend tensor log(const tensor& t) ;
	friend tensor exp(const tensor& t) ;
	friend tensor sin(const tensor& t) ;
	friend tensor cos(const tensor& t) ;
	friend tensor tanh(const tensor& t); 
	friend tensor sigmoid(const tensor& t);
	friend tensor operator+(f32 scalar, const tensor& t);
	friend tensor softmax(const tensor& t, i32 dim);
	friend tensor rmsnorm(const tensor& t);
	friend tensor layernorm(const tensor& t);
	friend tensor unsqueeze(const tensor& t);

	// manipulations
	friend tensor transpose(const tensor& t, u32 dim0, u32 dim1);
	friend tensor reshape(const tensor& t, vi32 new_shape);


	// void T(u32 dim0, u32 dim1);
	void reshape(vi32 new_shape);

	tensor gather(const tensor& indices, i32 dim = 0) const ;
	void unsqueeze();










};





}

#endif // !BZ_TENSOR_HPP
