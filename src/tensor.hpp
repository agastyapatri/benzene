#pragma once
#include <cstdint>
#include <iostream>
#include <vector>
#include <initializer_list>
#include <numbers> 
#define DEGREES_TO_RADIANS(deg)	(deg * std::numbers::pi / 180)
#define RADIANS_TO_DEGREES(rad) (rad * 180 / std::numbers::pi)

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
	i32  ndim()    const {return _ndim;}

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
	static tensor rand_he 	  (vi32 shape, f32 fan_in);
	static tensor rand_xavier (vi32 shape, f32 fan_in, f32 fan_out);

	//	transcendentals; new
	tensor log()  const;
	tensor exp()  const;
	tensor sin()  const;
	tensor cos()  const;
	tensor tanh() const; 

	//	transcendentals; inplace
	void log_();
	void exp_();
	void sin_();
	void cos_();
	void tanh_();


	//	 TODO >:(
	//	reductions
	tensor mean(u32 axis) const;
	tensor std (u32 axis) const;
	tensor max (u32 axis) const;
	tensor min (u32 axis) const;
	tensor sum (u32 axis) const;

	//	math on tensors for llm purposes
	static tensor matmul(const tensor& inp1, const tensor& inp2);
	tensor matmul(const tensor& other) const;
	tensor softmax(i32 dim) const;
	tensor layer_norm(const tensor& weight, const tensor& bias) const;
	tensor rmsnorm() const;

};

}
