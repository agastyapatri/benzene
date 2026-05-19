#include "tensor.hpp"
#include "nn.hpp"
#include "gpt2.hpp"
#include <chrono> 
namespace gpt2 = bz::gpt2;
using tensor = bz::tensor;

constexpr bz::i32 batch_size = 1;
constexpr bz::i32 seq_len =  4;
constexpr bz::i32 embd_dim = 768;
constexpr bz::i32 num_heads =  2;










int main(){
	tensor t1({1, 4}, {0.0f, 1.0f, 2.0f, 3.0f});
	auto start = std::chrono::high_resolution_clock::now();
	gpt2::GPT2 gpt2small(
			gpt2::gpt2_small_embd_dim, 
			gpt2::gpt2_small_num_heads, 
			gpt2::gpt2_small_trans_blocks, 
			gpt2::gpt2_small_vocab_size, 
			gpt2::gpt2_small_context_len
			);
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << elapsed << std::endl;
	
	// for(const auto& [key, value] : gpt2small.state_dict())
	// 	std::cout << key << "\n";

	





	return EXIT_SUCCESS;
}
