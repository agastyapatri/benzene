#include "nn.hpp"
#include "tensor.hpp"
constexpr bz::i32 batch_size = 2;
constexpr bz::i32 seq_len =  4;
constexpr bz::i32 d_in =  8;
constexpr bz::i32 d_kq =  8;
constexpr bz::i32 d_v =  8;
using tensor = bz::tensor;
namespace nn = bz::nn;


int main(void){
	tensor::manual_seed(42);
	tensor input = bz::tensor::randn({batch_size, seq_len, d_in});
	nn::CausalSelfAttention csa(d_in, d_kq, d_v);
	tensor out = csa(input);
	std::cout << out << std::endl;







	return EXIT_SUCCESS;

}


