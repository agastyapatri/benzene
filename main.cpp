#include "gpt2.hpp"
#include "tensor.hpp"

namespace nn = bz::nn;
namespace gpt2 = bz::gpt2;
using tensor = bz::tensor;

constexpr bz::i32 batch_size = 1;
constexpr bz::i32 seq_len =  4;
constexpr bz::i32 embd_dim = 10;

// constexpr bz::i32 embd_dim = 768;
// constexpr bz::i32 d_kq =  8;
// constexpr bz::i32 d_v =  8;
// constexpr bz::i32 num_heads =  2;

int main(){
	tensor t1 = tensor::randn({batch_size, seq_len, embd_dim});





	return EXIT_SUCCESS;
}
