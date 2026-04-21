#include "nn.hpp"
#include "tensor.hpp"
#include <unordered_map>
namespace bz::nn{

Linear::Linear(u32 in_features, u32 out_features, bool bias){
	_in_shape = in_features; 
	_out_shape = out_features; 
	_weight = tensor::randu_he({(i32)in_features, (i32)out_features}, in_features);
	if(bias){
		_bias = tensor::randu_he({(i32)out_features}, in_features);
	}
}

tensor Linear::forward(const tensor& input) const {
	tensor out = matmul(input, _weight);
	if(_bias.numel() > 0)	out = out+_bias;
	return out;
}

std::vector<tensor*> Linear::parameters(){
    std::vector<tensor*> params;
    params.push_back(&_weight);
    if (_bias.numel() > 0) {
        params.push_back(&_bias);
    }
    return params;
}

std::unordered_map<std::string, const tensor*> Linear::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd ;
	sd["weight"] = &_weight;
	if(_bias.numel() > 0)	sd["bias"] = &_bias;
	return sd;
}







}
