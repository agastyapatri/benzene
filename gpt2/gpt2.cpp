#include "gpt2.hpp"
#include "nn.hpp"
#include <memory>
#include <string>
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
	tensor out = _ln1(input);
	out = _mha(out);
	out = out + input; 
	tensor residual = out;
	out = _ln2(out);
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
	for(auto _param : _ln1.parameters()){
		params.push_back(_param);
	}
	for(auto _param : _ln2.parameters()){
		params.push_back(_param);
	}
    return params;
}

std::unordered_map<std::string, const tensor*> TransformerBlock::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd; 
	for(const auto& [key, value] : _mlp.state_dict()){
		sd["mlp." + key] = value;
	}
	for(const auto& [key, value] : _mha.state_dict()){
		sd["attn." + key] = value;
	}
	for(const auto& [key, value] : _ln1.state_dict()){
		sd["ln_1." + key] = value;
	}
	for(const auto& [key, value] : _ln2.state_dict()){
		sd["ln_2." + key] = value;
	}
	return sd;
}


GPT2::GPT2(u32 embedding_dim, u32 num_heads, u32 num_transformer_blocks, u32 vocab_size, u32 context_length){
	_embd_dim             = embedding_dim; 
	_num_trans_blocks     = num_transformer_blocks; 
	_num_heads            = num_heads;
	_vocab_size           = vocab_size;
	_context_len          = context_length;
	_token_embedding 	  = nn::Embedding(vocab_size, embedding_dim); 
	_positional_embedding = nn::Embedding(context_length, embedding_dim);
	_final_layer_norm     = nn::LayerNorm({static_cast<i32>(embedding_dim)});
	_final_projection     = nn::Linear(embedding_dim, vocab_size, false);
	for(u32 i = 0; i < num_transformer_blocks; i++){
		_transformers.push_back(make_transformer_block(_num_heads, _embd_dim));
	}
}


std::vector<tensor*> GPT2::parameters(){
	std::vector<tensor*> params; 
	for(auto param : _token_embedding.parameters())
		params.push_back(param);
	for(auto param : _positional_embedding.parameters())
		params.push_back(param);
	for(u32 i = 0; i < _num_trans_blocks; i++){
		for(auto param: _transformers[i]->parameters())
			params.push_back(param);
	}
	for(auto param : _final_layer_norm.parameters())
		params.push_back(param);
	for(auto param : _final_projection.parameters())
		params.push_back(param);
	return params;
}

std::unordered_map<std::string, const tensor*> GPT2::state_dict() const {
	std::unordered_map<std::string, const tensor*> sd; 
	sd["wte.weight"] = _token_embedding.state_dict()["weight"];
	sd["wpe.weight"] = _positional_embedding.state_dict()["weight"];
	for(u32 i = 0; i < _num_trans_blocks; i++){
		for(const auto& [key, value] : _transformers[i]->state_dict()){
			sd["h." + std::to_string(i) + "." + key] = value;
		}
	}
	sd["ln_f.weight"] = _final_layer_norm.state_dict()["weight"];
	sd["ln_f.bias"] = _final_layer_norm.state_dict()["bias"];
	return sd;
} 
	



tensor GPT2::forward(const tensor& input) const {
	tensor out = input;
	i32 _seq_len = input.shape()[input.ndim() - 1];
	tensor token_embeddings = _token_embedding(input);
	tensor positions = tensor::arange(0, _seq_len, 1); 
	tensor position_embeddings = _positional_embedding(positions);
	out = token_embeddings + position_embeddings;
	// if(out.ndim() == 2)	out.unsqueeze();	// if it is not a batched input, introduce a batch size of 1
	for(u32 i = 0; i < _num_trans_blocks; i++){
		out = _transformers[i]->forward(out);
	}
	out = _final_layer_norm(out);
	out = _final_projection(out);
	out = bz::softmax(out, -1);
	return out;
}




std::unique_ptr<MLPBlock> make_mlpblock(u32 input_dimension, u32 projection_factor){
	return std::make_unique<MLPBlock>(input_dimension, projection_factor);
}
std::unique_ptr<TransformerBlock> make_transformer_block(u32 num_heads, u32 embedding_dim){
	return std::make_unique<TransformerBlock>(num_heads, embedding_dim);
}





} 
