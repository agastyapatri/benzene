#include "nn.hpp"
#include "tensor.hpp"
#define ROWS 1
#define COLS 784
#define EMBEDDING_DIM 768
#define ITER 10000
#define DIM0 0
#define DIM1 1


int main(void){
	bz::tensor::manual_seed(0);
	bz::tensor x = bz::tensor::randn({10, EMBEDDING_DIM});
	bz::nn::Sequential seq;
	seq.push_back(bz::nn::make_linear(768, 768));
	seq.push_back(bz::nn::make_layernorm(EMBEDDING_DIM));
	seq.push_back(bz::nn::make_gelu());
	bz::tensor out = seq(x);
	for(auto i : out.shape()){
		std::cout << i << " ";
	}












}


