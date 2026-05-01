#include "tensor.hpp"
#define ROWS 1
#define COLS 784
#define ITER 10000
using tensor = bz::tensor;
#define DIM0 0
#define DIM1 1

bz::vi32 coords(bz::i32 idx, tensor t){ 
	bz::vi32 coords(t.ndim(), 0);
	for(bz::i32 j = 0; j < t.ndim(); j++){
		coords[j] = idx / t.strides()[j];
		idx  = idx % t.strides()[j];
	}
	return coords;
}


int main(){
	bz::tensor::manual_seed(0);


	//	REORDERING THE ELEMENTS IN MEMORY

	tensor t1 = tensor::randn({2,3,4});
	tensor t2 = tensor::rand_normal(t1.shape(), -100, 100);



	tensor t3 = t1 - t2;
	for(bz::u64 i = 0; i < t1.numel(); i++){
		std::cout << t1.data()[i] << " + " << t2.data()[i] << ": " << t3.data()[i] << "\n"; 
	}







}


