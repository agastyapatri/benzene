#pragma once
#include <iostream>
#include <vector>
#include <initializer_list>


namespace bz{
	typedef std::vector<int> vint;
	typedef std::vector<float> vfloat;


	class tensor{
		std::vector<float>  _data; 
		std::vector<int>   _shape;
		std::vector<int> _strides;
		size_t _numel;
		int   _ndim;
		void compute_strides();


	public: 
		tensor(std::vector<int> shape); 
		tensor(std::vector<int> shape, std::initializer_list<float> values); 

		//	accessors 
		float at(std::vector<int> idxs) const;
		float& at(std::vector<int> idxs);
		std::vector<float> data()    const	{return _data;}
		std::vector<int>   shape()   const	{return _shape;}
		std::vector<int>   strides() const  {return _strides;}
		size_t numel() const {return _numel; }
		int ndim() const {return _ndim  ; }



		//	overloaded operators
		tensor operator+(const tensor& other) const;
		tensor operator-(const tensor& other) const;
		tensor operator*(const tensor& other) const;
		tensor operator*(float scalar) const;
		tensor operator/(float scalar) const;
		friend std::ostream& operator<<(std::ostream& os, const tensor& t);

		// factory
		static tensor zeros(std::vector<int> shape);
		static tensor ones(std::vector<int> shape);
		static tensor randn(std::vector<int> shape);
		static tensor rand_uniform(std::vector<int> shape, float low = 0.0f, float high = 1.0f);
		static tensor rand_normal (std::vector<int> shape, float mean = 0.0f, float std = 1.0f);
		static tensor rand_he 	  (std::vector<int> shape, float fan_in);
		static tensor rand_xavier (std::vector<int> shape, float fan_in, float fan_out);



		void fill(float value);

		//	 TODO >:(
		//	reductions
		tensor mean(int axis) const;
		tensor std (int axis) const;
		tensor max (int axis) const;
		tensor min (int axis) const;
		tensor sum (int axis) const;

		//	math on tensors for llm purposes
		static tensor matmul(const tensor& inp1, const tensor& inp2);
		tensor matmul(const tensor& other) const;
		tensor softmax(int dim) const;
		tensor layer_norm(const tensor& weight, const tensor& bias) const;
		tensor rmsnorm() const;
	};
}
