/*
 *	Neural Network primitives for the Benzene LLM inferencing capabilities
 */ 
#ifndef BZ_NN_HPP
#define BZ_NN_HPP

#include "tensor.hpp"
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






}





#endif
