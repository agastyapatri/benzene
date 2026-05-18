#include "tensor.hpp"
#include "nn.hpp"
#include "gpt2.hpp"
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
	tensor t1 = tensor::randn({batch_size, seq_len, embd_dim}); 
	gpt2::TransformerBlock transblock(num_heads, embd_dim, d_kq, d_v); 
	tensor out = transblock(t1);
	for(auto i : out.shape())
		std::cout << i << " ";
	std::cout << std::endl;

	





	return EXIT_SUCCESS;
}
