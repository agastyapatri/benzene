#include "tensor.hpp"
#include "nn.hpp"
#include "gpt2.hpp"
#include <chrono> 
namespace gpt2 = bz::gpt2;
using tensor   = bz::tensor;
namespace nn   = bz::nn;

constexpr bz::i32 batch_size = 2;
constexpr bz::i32 seq_len =  4;
constexpr bz::i32 embd_dim = 768;
constexpr bz::i32 num_heads =  2;












int main(){
	tensor t1 = tensor::randn({5,5});
	tensor::save(t1, "test.npy");
	std::cout << t1 << std::endl;



}
