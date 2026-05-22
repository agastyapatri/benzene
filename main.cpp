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









#include <fstream> 
#include <vector> 
std::vector<char> readnpy(const std::string& path){
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if(!file) return{};
	std::streamsize size = file.tellg(); 
	file.seekg(0, std::ios::beg);
	std::vector<char> buffer(size);
	if(file.read(buffer.data(), size)){
		return buffer;
	}
	return buffer;
}






int main(){
	std::cout << 33 % 64 << std::endl;
	std::cout << 119 % 64 << std::endl;


}
