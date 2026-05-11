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
	tensor t1 = bz::tensor::randn({batch_size, seq_len, seq_len});
	tensor t2 = bz::tensor::randn({seq_len, seq_len});
	tensor t3 = t1 / t2;
	std::cout << t3 << std::endl;








	return EXIT_SUCCESS;

}


