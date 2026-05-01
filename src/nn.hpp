/*
 *	Neural Network primitives for the Benzene LLM inferencing capabilities
 */ 
#ifndef BZ_NN_HPP
#define BZ_NN_HPP
#include "tensor.hpp"

#include <memory> 
#include <unordered_map>

namespace bz::nn{

class Module;
class Linear; 
class Embedding;
class Sequential;
class ReLU; 
class GELU; 
class Softmax; 
class LayerNorm;



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

// class LayerNorm: public Module{
//
// };







//	nn::Module factory

std::unique_ptr<Linear>    make_linear(i32 in_shape, i32 out_shape, bool bias=true);
std::unique_ptr<ReLU>      make_relu();
std::unique_ptr<GELU>      make_gelu();
std::unique_ptr<Softmax>   make_softmax(i32 dim = -1);
std::unique_ptr<Embedding> make_embedding(i32 num_embeddings, i32 embedding_dim);
std::unique_ptr<LayerNorm> make_layernorm();





}





#endif
