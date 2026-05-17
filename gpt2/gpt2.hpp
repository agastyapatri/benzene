#ifndef BZ_GPT2_HPP
#define BZ_GPT2_HPP
#include "tensor.hpp"
#include "nn.hpp"
#include <memory>

namespace bz::gpt2{
constexpr u32 gpt2_small_embd_dim     = 768; 		
constexpr u32 gpt2_small_trans_blocks = 12; 	
constexpr u32 gpt2_small_vocab_size   = 50257;	
constexpr u32 gpt2_small_context_len  = 1024;	
constexpr u32 gpt2_small_ffwd_dim     = 3072;


class MLPBlock;
class TransformerBlock;
class GPT2Small;


class MLPBlock: public nn::Module{
	u32 _input_dim;			//	embedding dimension of the input sequence 
	u32 _projection_factor; //	projection from embedding_dim -> 4*embedding_dim
	nn::Linear fc1; 
	nn::Linear fc2; 
public: 
	MLPBlock() = default; 
	MLPBlock(u32 input_dimension, u32 projection_factor = 4) 
		: _input_dim(input_dimension), 
		_projection_factor(projection_factor), 
		fc1(input_dimension, input_dimension*projection_factor), 
		fc2(input_dimension*projection_factor, input_dimension)
	{};
	tensor forward(const tensor& input) const override;
	std::vector<tensor*> parameters()  override; 
	std::unordered_map<std::string, const tensor*> state_dict() const override;
};














std::unique_ptr<MLPBlock> make_mlpblock(u32 input_dimension, u32 projection_factor = 4);



}








#endif // !GPT_2_HPP
