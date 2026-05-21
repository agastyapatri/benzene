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


	// tensor t1({2, 4}, {1,2,3,4,5,6,7,8});
	// tensor t2({2, 4}, {9,10,11,12,13,14,15,16});
	// tensor t3({2, 4}, {17, 18, 19, 20, 21, 22, 23, 24});
	tensor t1 = tensor::randn({2,3, 4});
	tensor t2 = tensor::randn({2,3, 4});
	// tensor t3 = tensor::randn({2,3, 4});


	


	// tensor out = bz::concat({t1, t2}, 0);
	// for(auto i : out.shape())	
	// 	std::cout << i << " ";
	// std::cout << std::endl << std::endl;
	//
	// std::cout << out << std::endl;


	tensor out = bz::concat({t1, t2}, 2);
	for(auto i : out.shape())	
		std::cout << i << " ";
	std::cout << std::endl << std::endl;

	std::cout << out << std::endl;

	// out = bz::concat({t1, t2}, 2);
	// for(auto i : out.shape())	
	// 	std::cout << i << " ";
	// std::cout << std::endl << std::endl;
	//
	// std::cout << out << std::endl;
	//



















	
	// tensor tokens({1, 4}, {1,2,3,4});
	// gpt2::GPT2 gpt2small(
	// 		gpt2::gpt2_small_embd_dim, 
	// 		gpt2::gpt2_small_num_heads, 
	// 		gpt2::gpt2_small_trans_blocks, 
	// 		gpt2::gpt2_small_vocab_size, 
	// 		gpt2::gpt2_small_context_len
	// 		);
	// auto start = std::chrono::high_resolution_clock::now();
	// tensor out = gpt2small(tokens);
	// auto end = std::chrono::high_resolution_clock::now();
	// auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	//
	//
	// for(auto i : out.shape())
	// 	std::cout << i << " ";
	// std::cout << std::endl;
	// std::cout << elapsed.count()*1e-6 << " seconds." << std::endl;




	

	





	return EXIT_SUCCESS;
}
