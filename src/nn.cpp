#include "nn.hpp"
#include "tensor.hpp"
#include <cassert>
#include <limits>
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
	tensor output;
	if(input.ndim() == 1){
		output = _weight.gather(input);
		return output;
	}
	// input = [batch_size, sequence_length]
	else if(input.ndim() == 2){
		i32 batch_size 		= input.shape()[0];
		i32 sequence_length = input.shape()[1];
		tensor output({batch_size, sequence_length, _embedding_dim});
		for(i32 i = 0; i < batch_size; i++){
			for(i32 j = 0; j < sequence_length; j++){
				i32 query_idx = input.at({i, j});	// index which currently is building the embedding slice
				for(i32 k = 0; k < _embedding_dim; k++){
					output.at({i, j, k}) = _weight.at({query_idx, k});
				}
			}
		}
		return output;

	}
	else{
		throw std::runtime_error("singular and batched inputs can be handled now");
	}
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



SelfAttention::SelfAttention(i32 d_in, i32 d_kq, i32 d_v){
	_d_in = d_in;
	_d_kq = d_kq;
	_d_v  = d_v;
	_w_q = tensor::randu_he({_d_in, _d_kq}, _d_in);
	_w_k = tensor::randu_he({_d_in, _d_kq}, _d_in);
	_w_v = tensor::randu_he({_d_in, _d_v}, _d_in);
}

std::vector<tensor*> SelfAttention::parameters(){
    std::vector<tensor*> params;
    params.push_back(&_w_q);
    params.push_back(&_w_k);
    params.push_back(&_w_v);
    return params;
}

std::unordered_map<std::string, const tensor*> SelfAttention::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd ;
	sd["w_q.weight"] = &_w_q;
	sd["w_k.weight"] = &_w_k;
	sd["w_v.weight"] = &_w_v;
	return sd;
}

tensor SelfAttention::forward(const tensor& input) const {
	tensor keys = bz::matmul(input, _w_k);	 	//	batch_size x seq_len x d_kq
	tensor queries = bz::matmul(input, _w_q); 	//	batch_size x seq_len x d_kq
	tensor values = bz::matmul(input, _w_v);	//	batch_size x seq_len x d_v
	tensor attn_scores = bz::matmul(queries, bz::transpose(keys, keys.ndim() - 1, keys.ndim() - 2)); // batch_size x seq_len x seq_len
	tensor attn_weights = bz::softmax(attn_scores *std::sqrtf(1.0f / _d_kq) , -1); 					 //	batch_size x seq_len x seq_len
	tensor context_vec = bz::matmul(attn_weights, values); 											 // batch_size x seq_len x d_v
	return context_vec;
}


CausalSelfAttention::CausalSelfAttention(i32 d_in, i32 d_kq, i32 d_v){
	_d_in = d_in;
	_d_kq = d_kq;
	_d_v  = d_v;
	_w_q = tensor::randu_he({_d_in, _d_kq}, _d_in);
	_w_k = tensor::randu_he({_d_in, _d_kq}, _d_in);
	_w_v = tensor::randu_he({_d_in, _d_v}, _d_in);
}
std::vector<tensor*> CausalSelfAttention::parameters(){
    std::vector<tensor*> params;
    params.push_back(&_w_q);
    params.push_back(&_w_k);
    params.push_back(&_w_v);
    return params;
}

std::unordered_map<std::string, const tensor*> CausalSelfAttention::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd ;
	sd["w_q.weight"] = &_w_q;
	sd["w_k.weight"] = &_w_k;
	sd["w_v.weight"] = &_w_v;
	return sd;
}



static inline tensor _make_mask(i32 size){
	tensor out({size, size});
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			out.at({i, j}) = (i >=  j) ? 0 : -std::numeric_limits<float>::infinity();
		}
	}
	return out;
}

tensor CausalSelfAttention::forward(const tensor& input) const {
	tensor keys = bz::matmul(input, _w_k);	 	//	batch_size x seq_len x d_kq
	tensor queries = bz::matmul(input, _w_q); 	//	batch_size x seq_len x d_kq
	tensor values = bz::matmul(input, _w_v);	//	batch_size x seq_len x d_v
	tensor attn_scores = bz::matmul(queries, bz::transpose(keys, keys.ndim() - 1, keys.ndim() - 2)); // batch_size x seq_len x seq_len
	attn_scores = attn_scores + _make_mask(attn_scores.shape()[attn_scores.ndim() -1]); 			 // masking the attention scores
	tensor attn_weights = bz::softmax(attn_scores *std::sqrtf(1.0f / _d_kq) , -1); 					 //	batch_size x seq_len x seq_len
	tensor context_vec = bz::matmul(attn_weights, values); 											 // batch_size x seq_len x d_v
	return context_vec;
}


MultiheadAttention::MultiheadAttention(i32 num_heads, i32 d_in, i32 d_kq, i32 d_v){
	_num_heads = num_heads; 
	_d_in = d_in; 
	_d_kq = d_kq; 
	_d_v = d_v;
	_w_k = tensor::randu_he({d_in, num_heads*d_kq}, d_in);  
	_w_q = tensor::randu_he({d_in, num_heads*d_kq}, d_in);
	_w_v = tensor::randu_he({d_in, num_heads*d_v} , d_in);
	_w_o = tensor::randu_he({num_heads*d_v, d_in} , num_heads*d_v);
} 


std::vector<tensor*> MultiheadAttention::parameters(){
    std::vector<tensor*> params;
    params.push_back(&_w_q);
    params.push_back(&_w_k);
    params.push_back(&_w_v);
    params.push_back(&_w_o);
    return params;
}

std::unordered_map<std::string, const tensor*> MultiheadAttention::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd ;
	sd["w_q.weight"] = &_w_q;
	sd["w_k.weight"] = &_w_k;
	sd["w_v.weight"] = &_w_v;
	sd["w_o.weight"] = &_w_o;
	return sd;
}

//	TODO: fix this function; fix tesnor::T 
tensor MultiheadAttention::forward(const tensor& input) const{
	i32 bs = input.shape()[0];
	i32 sl = input.shape()[1];
	tensor queries = bz::matmul(input, _w_q); 		//	batch_size x seq_len x num_heads*d_kq
	tensor keys = bz::matmul(input, _w_k);	 		//	batch_size x seq_len x num_heads*d_kq
	tensor values = bz::matmul(input, _w_v);		//	batch_size x seq_len x num_heads*d_v
	queries.reshape({bs, sl, _num_heads, _d_kq});   //	batch_size x seq_len x num_heads x d_kq
	keys.reshape({bs, sl, _num_heads, _d_kq});	    //	batch_size x seq_len x num_heads x d_kq 	
	values.reshape({bs, sl, _num_heads, _d_v});     //	batch_size x seq_len x num_heads x d_v
	queries = bz::transpose(queries, 1, 2);
	keys = bz::transpose(keys, 1, 2);
	values = bz::transpose(values, 1, 2);
	keys = bz::transpose(keys, keys.ndim() - 1, keys.ndim() - 2);
	tensor attn_scores = bz::matmul(queries, keys);	//	batch_size x num_heads x seq_len x seq_len
	attn_scores = attn_scores + _make_mask(attn_scores.shape()[attn_scores.ndim() -1]);
	tensor attn_weights = bz::softmax(attn_scores *std::sqrtf(1.0f / _d_kq) , -1); 					 
	tensor context_vec = bz::matmul(attn_weights, values);		//   batch, num_heads, seq_len, d_v  											 
	context_vec = bz::transpose(context_vec, 1, 2);
	context_vec.reshape({bs, sl, _num_heads*_d_v});
	return bz::matmul(context_vec, _w_o);
}




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

std::unique_ptr<SelfAttention>   make_SA(i32 d_in, i32 d_kq, i32 d_v){
	return std::make_unique<SelfAttention>(d_in, d_kq, d_v);
}

std::unique_ptr<CausalSelfAttention>   make_CSA(i32 d_in, i32 d_kq, i32 d_v){
	return std::make_unique<CausalSelfAttention>(d_in, d_kq, d_v);
}

std::unique_ptr<MultiheadAttention>  make_MHA(i32 num_heads, i32 d_in, i32 d_kq, i32 d_v){
	return std::make_unique<MultiheadAttention>(num_heads,d_in,  d_kq,  d_v);
}





}
