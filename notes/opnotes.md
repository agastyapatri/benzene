#  OPTIMIZING PARTS OF THE BENZENE PRIMITIVES
It is imperative that parts of this project -  scaffolding on which actual work is done - are as optimized as I can get them. Unnecessary copies, moves etc. are just a cop out to get things done quickly, but have deficiencies in speed and efficiency which propagate into the code written on top of the compromised code, and get compounded in their effect. 

As of 2026-05-17, this is what needs to be done to make things better: 

1.  Replace `bz::transpose` from a copy-move approach to a "view based approach". Swap the strides and materialize the physical move of data only when absolutely necessary.
2.  Change how `flat_idx_to_coords` is being calculated - divisions and modulos are expensive operations. 
3.  In `bz::nn::MultiheadAttention` and other functions, every operation allocates a new tensor. A memory pool or arena allocator would significantly reduce this overhead.
4.  `_make_mask` is being recomputed for every forward pass. Find a way to cache it.
5.  `bz::tensor::sum`, `bz::tensor::mean` and other reductions are single threaded. They can be parallelized with OpenMP reductions.
6.  `operator==` uses exact float comparison. Consider replacing with a margin of `1e-9`
7.  Elementwise loops could use AVX2 intrinsics to process 8 floats at a time instead of 1. 
8.  Loop fusions whenever possible???



##  1.  Wrestling with `bz::transpose`
I'm in a bit of a conundrum. `bz::transpose` currently is this: 

```cpp 
tensor transpose(const tensor& t, u32 dim0, u32 dim1){
	tensor out = t;
	std::swap(out._shape[dim0], out._shape[dim1]);
	out.compute_strides();

	for(bz::u64 i = 0; i < t._numel; i++){
		bz::vi32 original_loc = t.flat_idx_to_coord(i);
		bz::vi32 new_loc = original_loc;
		std::swap(new_loc[dim0], new_loc[dim1]);
		i32 offset = 0;
		for(bz::i32 j = 0; j < out._ndim; j++){
			offset += new_loc[j] * out._strides[j];
		}
		out._data[offset] = t._data[i];
	}

	return out;
}
```
The new tensor created is a copy of the original with strides swapped and all the data being physically reordered. This a _ridiculously_ slow and inefficient program. 
One fixes this by realizing that data does not need to be moved, really. A transposed tensor is the same tensor as the original but with swapped strides.

`bz::tensor::at({i, j})` for the transposed tensor really just becomes `bz::tensor::at({j, i})}` on the original. This apparently is what NumPy does as well, its transpose is free because it is just a metadata change. 

The issue crops up when the tensor needs to be iterated through. All loops which are of the form 
```cpp 
for(i32 i = 0; i < _numel; i++)
    _data[i] += ....
```
Are now invalid because for a view-based transpose, the data becomes non-contiguous. Meaning, the data returned by the strided offset calculation and the data actually in the memory are not the same. To fix this, what needs to be done in every flat loop is the following: 

SO, heres what needs to be done: 

1. Refactor bz::transpose — swap shape and strides only, no data movement.
2. Implement tensor::contiguous() — physically reorders data into row-major layout, returns a new tensor.
3. Elementwise unary ops (exp, log, relu, etc.) — if contiguous, flat loop as is. If not, strided iterator.
4. Elementwise binary ops (operator+, operator-, operator*, operator/) — if both tensors contiguous, existing logic unchanged. If not contiguous, strided iterator for the fallback.
5. Matrix multiplication specifically — detect if an input was transposed via is_contiguous() and pass CblasTrans to cblas_sgemm accordingly, avoiding any data movement entirely.


























