#pragma once
#include <cstdint>
#include <iostream>
#include <vector>
#include <initializer_list>

namespace bz{

typedef uint32_t u32; 
typedef uint64_t u64; 
typedef int64_t  i64; 
typedef int32_t  i32; 
typedef float  	 f32; 
typedef double   f64; 
typedef std::vector<uint32_t> vu32; 
typedef std::vector<uint64_t> vu64; 
typedef std::vector<int64_t>  vi64; 
typedef std::vector<int32_t>  vi32; 
typedef std::vector<float> 	  vf32; 
typedef std::vector<double>   vf64; 



class tensor{
	vf32 _data;
	vi32 _shape;
	vi32 _strides;
	u64  _numel;
	i32  _ndim;
	void compute_strides();

public: 
	tensor(vi32 shape); 
	tensor(vi32 shape, std::initializer_list<float> values); 

	//	accessors 
	f32 at(vi32 idxs) const;
	f32& at(vi32 idxs);
	std::vector<float> data()    const	{return _data;}
	vi32 shape()   const {return _shape;}
	vi32 strides() const {return _strides;}
	u64  numel()   const {return _numel; }
	i32  ndim()    const {return _ndim  ; }

	//	overloaded operators
	tensor operator+(const tensor& other) const;
	tensor operator-(const tensor& other) const;
	tensor operator*(const tensor& other) const;
	tensor operator*(f32 scalar) const;
	tensor operator/(f32 scalar) const;
	friend std::ostream& operator<<(std::ostream& os, const tensor& t);

	// factory
	void   fill(f32 value);
	static tensor zeros(vi32 shape);
	static tensor ones(vi32 shape);
	static tensor randn(vi32 shape);
	static tensor rand_uniform(vi32 shape, f32 low = 0.0f, f32 high = 1.0f);
	static tensor rand_normal (vi32 shape, f32 mean, f32 std);



	//	 TODO >:(
	//	reductions
	tensor mean(i32 axis) const;
	tensor std (i32 axis) const;
	tensor max (i32 axis) const;
	tensor min (i32 axis) const;
	tensor sum (i32 axis) const;
	static tensor rand_he 	  (vi32 shape, f32 fan_in);
	static tensor rand_xavier (vi32 shape, f32 fan_in, f32 fan_out);

	//	math on tensors for llm purposes
	static tensor matmul(const tensor& inp1, const tensor& inp2);
	tensor matmul(const tensor& other) const;
	tensor softmax(i32 dim) const;
	tensor layer_norm(const tensor& weight, const tensor& bias) const;
	tensor rmsnorm() const;


	//	transcendentals; new
	tensor log();
	tensor exp();
	tensor sin();
	tensor cos();
	tensor tanh();

	//	transcendentals; inplace
	void log_();
	void exp_();
	void sin_();
	void cos_();
	void tanh_();

	
};
}
