#include "nn.hpp"
#include "tensor.hpp"
#define ROWS 1
#define COLS 784
#define EMBEDDING_DIM 768
#define ITER 10000
#define DIM0 0
#define DIM1 1
using tensor = bz::tensor;
namespace nn = bz::nn;


int main(void){
	tensor::manual_seed(0);
	tensor x = tensor::randn({10, EMBEDDING_DIM});
	nn::Sequential seq;
	seq.push_back(nn::make_linear(768, 768));
	seq.push_back(nn::make_layernorm(EMBEDDING_DIM));
	seq.push_back(nn::make_gelu());
	tensor out = seq(x);
	for(auto i : out.shape()){
		std::cout << i << " ";
	}












}


