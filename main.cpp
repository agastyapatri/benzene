#include "gpt2.hpp"
#include "tensor.hpp"

namespace nn = bz::nn;
namespace gpt2 = bz::gpt2;
using tensor = bz::tensor;

constexpr bz::i32 batch_size = 1;
constexpr bz::i32 seq_len =  4;
constexpr bz::i32 embd_dim = 768;
constexpr bz::i32 d_kq =  8;
constexpr bz::i32 d_v =  8;
constexpr bz::i32 num_heads =  2;

int main(){
	tensor x = tensor::randn({batch_size, seq_len, gpt2::gpt2_small_embd_dim});
	nn::Sequential seq;
	seq.push_back(std::make_unique<gpt2::MLPBlock>(gpt2::gpt2_small_embd_dim));
	tensor out = seq(x);
	for(auto i : out.shape())
		std::cout << i << " ";
	std::cout << std::endl;


	


	
	return EXIT_SUCCESS;
}
