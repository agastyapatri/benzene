#include "tensor.hpp"
#include "linalg.hpp"
using tensor = bz::tensor;
namespace linalg = bz::linalg;


int main(){
	bz::tensor::manual_seed(0);
	tensor a = tensor::ones({5,5});
	tensor b = linalg::trace(a, 1);
	return 0;
}
