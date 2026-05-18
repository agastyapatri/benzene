#include "gpt2.hpp"
#include "nn.hpp"
#include <memory>
#include <unordered_map>

namespace bz::gpt2{

tensor MLPBlock::forward(const tensor& input) const {
	tensor out = fc1(input);
	out = bz::gelu(out); 
	out = fc2(out);
	return out;
}

std::vector<tensor*> MLPBlock::parameters(){
    std::vector<tensor*> params;
	for(auto _param : fc1.parameters()){
		params.push_back(_param);
	}
	for(auto _param : fc2.parameters()){
		params.push_back(_param);
	}
    return params;

}

std::unordered_map<std::string, const tensor*> MLPBlock::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd ;
	sd["c_fc.weight"] = fc1.state_dict()["weight"];
	sd["c_fc.bias"] = fc1.state_dict()["bias"];
	sd["c_proj.weight"] = fc2.state_dict()["weight"];
	sd["c_proj.bias"] = fc2.state_dict()["bias"];
	return sd;

}


tensor TransformerBlock::forward(const tensor& input) const{
	tensor out = _ln(input);
	out = _mha(out);
	out = _mlp(out);
	out = out + input;
	tensor residual = out;
	out = _ln(out);
	out = _mlp(out);
	out = out + residual;
	return out;
}

std::vector<tensor*> TransformerBlock::parameters(){
    std::vector<tensor*> params;
	for(auto _param : _mlp.parameters()){
		params.push_back(_param);
	}
	for(auto _param : _mha.parameters()){
		params.push_back(_param);
	}
    return params;
}

std::unordered_map<std::string, const tensor*> TransformerBlock::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd; 
	for(const auto& [key, value] : _mlp.state_dict()){
		sd[key] = value;
	}
	for(const auto& [key, value] : _mha.state_dict()){
		sd[key] = value;
	}
	return sd;
}











std::unique_ptr<MLPBlock> make_mlpblock(u32 input_dimension, u32 projection_factor){
	return std::make_unique<MLPBlock>(input_dimension, projection_factor);
}





} 
