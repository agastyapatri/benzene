#include "nn.hpp"
#include "tensor.hpp"
constexpr bz::i32 batch_size = 1;
constexpr bz::i32 seq_len =  4;
constexpr bz::i32 d_in =  8;
constexpr bz::i32 d_kq =  8;
constexpr bz::i32 d_v =  8;
constexpr bz::i32 num_heads =  2;
using tensor = bz::tensor;
namespace nn = bz::nn;


int main(void){
	tensor::manual_seed(42);
	tensor t1 = bz::tensor::randn({batch_size, seq_len, d_in});
	nn::MultiheadAttention mha(num_heads, d_in, d_kq, d_v);
	tensor out = mha(t1);
	// std::cout << out << std::endl;

	









	return EXIT_SUCCESS;

}


