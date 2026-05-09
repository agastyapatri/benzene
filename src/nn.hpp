/*
 *	Neural Network primitives for the Benzene LLM inferencing capabilities
 */ 
#ifndef BZ_NN_HPP
#define BZ_NN_HPP
#include "tensor.hpp"

#include <initializer_list>
#include <memory> 
#include <unordered_map>

namespace bz::nn{

class Module;
class Linear; 
class Embedding;
class Sequential;
class ReLU; 
class LeakyReLU; 
class Tanh;
class GELU; 
class Softmax; 
class LayerNorm;
class RMSNorm;
class Conv2d;



class Module{
public: 
	virtual ~Module() = default; 
	virtual tensor forward(const tensor& input) const = 0;
	virtual std::vector<tensor*> parameters() {return {};} 
	virtual std::unordered_map<std::string, const tensor*> state_dict() const {return {};}
	tensor operator()(const tensor& input) const {return forward(input);}
};


class ReLU : public Module {
public: 
	tensor forward(const tensor& input) const override {return bz::relu(input);} 
};

class LeakyReLU : public Module {
	f32 _negative_slope;
public: 
	LeakyReLU(f32 negative_slope) : _negative_slope(negative_slope){}
	tensor forward(const tensor& input) const override {return bz::leakyrelu(input, _negative_slope);} 
};

class Tanh : public Module {
public: 
	tensor forward(const tensor& input) const override {return bz::tanh(input);} 
};

class GELU : public Module {
public: 
	tensor forward(const tensor& input) const override {return bz::gelu(input);} 
};

class Softmax : public Module {
private: 
	i32 _dim;
public: 
	Softmax(i32 dim = -1) : _dim(dim) {}
	tensor forward(const tensor& input) const override {return bz::softmax(input, _dim);} 
};











class Linear: public Module{
private: 
	tensor _weight;
	std::optional<tensor> _bias = std::nullopt;
	i32	_in_shape;
	i32 _out_shape;
public: 
	Linear() = default;
	Linear(i32 in_features, i32 out_features, bool bias = true);
	tensor forward(const tensor& input) const override;
	std::vector<tensor*> parameters()  override; 
	std::unordered_map<std::string, const tensor*> state_dict() const override;
};


class Sequential: public Module{
private: 
	std::vector<std::unique_ptr<Module>> _layers;
public: 
	Sequential() = default; 
	void push_back(std::unique_ptr<Module> layer) {return _layers.push_back(std::move(layer));}
	tensor forward(const tensor& input) const override;
	std::vector<tensor*> parameters()  override; 
	std::unordered_map<std::string, const tensor*> state_dict() const override;
	i32 num_layers() const {return _layers.size();} 
};


class Embedding: public Module{
private: 
	i32 _num_embeddings; 
	i32 _embedding_dim; 
	tensor _weight;
public: 
	Embedding() = default; 
	Embedding(const i32 num_embeddings, const i32 embedding_dim);
	tensor forward(const tensor& input) const override;
	std::vector<tensor*> parameters()  override; 
	std::unordered_map<std::string, const tensor*> state_dict() const override;
};

class LayerNorm: public Module{
	tensor _weight;
	tensor _bias;
	vi32 _normalized_shape;
public: 
	LayerNorm() = default;
	LayerNorm(vi32 normalized_shape); 
	tensor forward(const tensor& input) const override;
	std::vector<tensor*> parameters()  override; 
	std::unordered_map<std::string, const tensor*> state_dict() const override;
};

class RMSNorm: public Module{
	tensor _weight;
	vi32 _normalized_shape;
public:
	RMSNorm() = default; 
	RMSNorm(vi32 normalized_shape);
	tensor forward(const tensor& input) const override;
	std::vector<tensor*> parameters()  override; 
	std::unordered_map<std::string, const tensor*> state_dict() const override;
};










//	TODO
//
//
//
//
// class Conv2d: public Module {
// 	i32 _in_channels;
// 	i32 _out_channels; 
// 	i32 _stride;
// 	i32 _kernel_size;
// 	i32 _padding;
// 	tensor _weight; 
// 	tensor _bias;
// public: 
// 	Conv2d() = default; 
// 	Conv2d(i32 in_channels, i32 out_channels, i32 kernel_size, i32 stride = 1, i32 padding = 0);
// 	tensor forward(const tensor& input) const override;
// 	std::vector<tensor*> parameters()  override; 
// 	std::unordered_map<std::string, const tensor*> state_dict() const override;
//
// };

// class MultiheadAttention : public Module {};
// class CausalAttention    : public Module {};
// class CrossAttention     : public Module {};






//	nn::Module factory

std::unique_ptr<Linear>    make_linear(i32 in_shape, i32 out_shape, bool bias=true);
std::unique_ptr<ReLU>      make_relu();
std::unique_ptr<GELU>      make_gelu();
std::unique_ptr<Tanh>      make_tanh();
std::unique_ptr<LeakyReLU> make_leakyrelu(f32 negative_slope);
std::unique_ptr<Softmax>   make_softmax(i32 dim = -1);
std::unique_ptr<Embedding> make_embedding(i32 num_embeddings, i32 embedding_dim);
std::unique_ptr<LayerNorm> make_layernorm(i32 normalized_shape);
std::unique_ptr<RMSNorm>   make_rmsnorm(i32 normalized_shape);


// TODO 
// std::unique_ptr<Conv2d>    make_conv2d(i32 in_channels, i32 out_channels, i32 kernel_size, i32 stride = 1, i32 padding = 0);
// std::unique_ptr<MultiheadAttention> make_multiheadattention();
// std::unique_ptr<CausalAttention>    make_causalattention();
// std::unique_ptr<CrossAttention>     make_crossattention();





}





#endif
