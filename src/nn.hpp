/*
 *	Neural Network primitives for the Benzene LLM inferencing capabilities
 */ 
#ifndef BZ_NN_HPP
#define BZ_NN_HPP

#include "tensor.hpp"

#include <memory> 
#include <unordered_map>
namespace bz::nn{

class module;
class Linear; 
class Embedding;
class Sequential;


class module{
public: 
	virtual ~module() = default; 
	virtual tensor forward(const tensor& input) const = 0;
	virtual std::vector<tensor*> parameters() {return {};} 
	virtual std::unordered_map<std::string, const tensor*> state_dict() const {return {};}
	tensor operator()(const tensor& input) const {return forward(input);}
};


class Linear: public module{
private: 
	tensor _weight;
	tensor _bias;
	u32	_in_shape;
	u32 _out_shape;
public: 
	Linear() = default;
	Linear(u32 in_features, u32 out_features, bool bias = true);
	tensor forward(const tensor& input) const override;
	std::vector<tensor*> parameters()  override; 
	std::unordered_map<std::string, const tensor*> state_dict() const override;
};


//	TODO 
class Embedding: public module{
private: 
	u32 _num_embeddings; 
	u32 _embedding_dim; 
public: 
	Embedding() = default; 
	Embedding(u32 num_embeddings, u32 embedding_dim) : _num_embeddings(num_embeddings), _embedding_dim(embedding_dim){}
	tensor forward(const tensor& input) const override;
	std::vector<tensor*> parameters()  override; 
	std::unordered_map<std::string, const tensor*> state_dict() const override;
};


//	TODO
class Sequential: public module{
private: 
	u32 _num_layers;
	std::vector<std::unique_ptr<module>> _layers;
public: 
	Sequential() = default; 
	Sequential(std::vector<module> layers);
	void push_back(const module& layer);
	void push_back(std::unique_ptr<module> layer);
	tensor forward(const tensor& input) const override;
	std::vector<tensor*> parameters()  override; 
	std::unordered_map<std::string, const tensor*> state_dict() const override;

	u32 num_layers() const {return _num_layers;}
};





}





#endif
