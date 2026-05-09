#include "nn.hpp"
#include "tensor.hpp"
#include <memory>
#include <string>
#include <unordered_map>
namespace bz::nn{

Linear::Linear(i32 in_features, i32 out_features, bool bias){
	_in_shape = in_features; 
	_out_shape = out_features; 
	_weight = tensor::randu_he({(i32)in_features, (i32)out_features}, in_features);
	if(bias){
		_bias = tensor::randu_he({(i32)out_features}, in_features);
	}
}

tensor Linear::forward(const tensor& input) const {
	tensor out = matmul(input, _weight);
	if(_bias.has_value())	out = out+_bias.value();
	return out;
}

std::vector<tensor*> Linear::parameters(){
    std::vector<tensor*> params;
    params.push_back(&_weight);
	if(_bias.has_value())	params.push_back(&_bias.value());
    return params;
}

std::unordered_map<std::string, const tensor*> Linear::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd ;
	sd["weight"] = &_weight;
	if(_bias.has_value())	sd["bias"] = &_bias.value();
	return sd;
}





std::vector<tensor*> Sequential::parameters() {
    std::vector<tensor*> params;
	for(const auto& layer: _layers){
		auto curr_params = layer->parameters();
		params.insert(params.end(), curr_params.begin(), curr_params.end());
	}
	return params;
} 

std::unordered_map<std::string, const tensor*> Sequential::state_dict() const{
	std::unordered_map<std::string, const tensor*> sd ;
	for(u64 i = 0; i < _layers.size(); i++){
		auto layer_sd = _layers[i]->state_dict();
		for(const auto& [name, ptr] : layer_sd){
			sd[std::to_string(i) + "." + name] = ptr;
		}
	}
	return sd;
}

tensor Sequential::forward(const tensor& input) const {
	tensor out = input;
	for(const auto& layer: _layers){
		out = layer->forward(out);

	}
	return out;
}

Embedding::Embedding(const i32 num_embeddings, const i32 embedding_dim){
	_num_embeddings = num_embeddings; 
	_embedding_dim = embedding_dim; 
	_weight = tensor::rand_normal({num_embeddings, embedding_dim}, 0, 0.02);
}

tensor Embedding::forward(const tensor& input) const{
	tensor output = _weight.gather(input);
	return output;
}

std::unordered_map<std::string, const tensor*> Embedding::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd ;
	sd["weight"] = &_weight;
	return sd;
}

std::vector<tensor*> Embedding::parameters(){
    std::vector<tensor*> params;
    params.push_back(&_weight);
    return params;
}


LayerNorm::LayerNorm(vi32 normalized_shape){
	_normalized_shape = normalized_shape;
	_weight = tensor::ones(normalized_shape);
	_bias = tensor::zeros(normalized_shape);
}

tensor LayerNorm::forward(const tensor& input) const {
	tensor out = bz::layernorm(input);
	out = (out * _weight) + _bias; 
	return out;
}


std::vector<tensor*> LayerNorm::parameters(){
    std::vector<tensor*> params;
    params.push_back(&_weight);
    params.push_back(&_bias);
    return params;
}

std::unordered_map<std::string, const tensor*> LayerNorm::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd ;
	sd["weight"] = &_weight;
	sd["bias"] = &_bias;
	return sd;
}


RMSNorm::RMSNorm(vi32 normalized_shape){
	_normalized_shape = normalized_shape;
	_weight = tensor::ones(normalized_shape);
}

tensor RMSNorm::forward(const tensor& input) const {
	tensor out = bz::rmsnorm(input);
	out = out * _weight;
	return out;
}

std::vector<tensor*> RMSNorm::parameters(){
    std::vector<tensor*> params;
    params.push_back(&_weight);
    return params;
}

std::unordered_map<std::string, const tensor*> RMSNorm::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd ;
	sd["weight"] = &_weight;
	return sd;
}

//	TODO 
// Conv2d::Conv2d(i32 in_channels, i32 out_channels, i32 kernel_size, i32 stride, i32 padding){
// 	_in_channels = in_channels;
// 	_out_channels = out_channels; 
// 	_stride = stride;
// 	_kernel_size = kernel_size;
// 	_padding = padding;
// 	_weight = tensor::randn({_kernel_size, _kernel_size});
// }










/*****************
 *	Layer Factory
*****************/


std::unique_ptr<Linear> make_linear(i32 in_shape, i32 out_shape, bool bias){
	return std::make_unique<Linear>(in_shape, out_shape, bias);
}

std::unique_ptr<ReLU> make_relu(){
	return std::make_unique<ReLU>();
}

std::unique_ptr<LeakyReLU> make_leakyrelu(f32 negative_slope){
	return std::make_unique<LeakyReLU>(negative_slope);
}

std::unique_ptr<Tanh> make_tanh(){
	return std::make_unique<Tanh>();
}

std::unique_ptr<GELU> make_gelu(){
	return std::make_unique<GELU>();
}

std::unique_ptr<Softmax> make_softmax(i32 dim){
	return std::make_unique<Softmax>(dim);
}


std::unique_ptr<Embedding> make_embedding(i32 num_embeddings, i32 embedding_dim){
	return std::make_unique<Embedding>(num_embeddings, embedding_dim);
}

std::unique_ptr<LayerNorm> make_layernorm(i32 normalized_shape){
	vi32 norm_shape(1, normalized_shape);
	return std::make_unique<LayerNorm>(norm_shape);
}

std::unique_ptr<RMSNorm> make_rmsnorm(i32 normalized_shape){
	vi32 norm_shape(1, normalized_shape);
	return std::make_unique<RMSNorm>(norm_shape);
}







}
