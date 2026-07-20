```
░▒▓███████▓▒░░▒▓████████▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓████████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░    ░▒▓██▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓███████▓▒░░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░  ░▒▓██▓▒░  ░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░   
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓██▓▒░    ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        
░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░ 
```
Benzene is a very tiny LLM inference engine, written in C++20. This began as an educational project, and my intent is to slowly increase the complexity of the project as my skill improves.
I've tried to keep the dependencies minimal, but BLAS/LAPACK were used to make the core math kernels reasonably performant.

##  Roadmap
1.  `bz::tensor` foundation + linear algebra
-   ~Tensor class~
-   ~Factory methods + Random initialization~
-   ~Reductions~ 
-   ~Arithmetic; operator overloading~
-   ~matmul (BLAS)~ 
-   ~transpose~ 
-   ~Broadcasting~ 
-   ~softmax~ 
-   ~layer_norm~
-   ~rmsnorm~ 
-   ~reshape~ 
-   slice 
-   ~concat~ 
-   ~Tensor I/O: Supporting NumPy's `.npy` format.~

2.  Neural Network Primitives 
-   ~Linear Layer~
-   ~GELU~
-   ~Causal Self Attention~
-   ~Multi Head Self Attention~
-   ~Feed Forward Networks~
-   ~Transformer Block - attention + ffn + layernorm~

3.  Load Weights 
-   Loading GPT-2 weights; support GGUF? safetensor? 
-   Weight map : name -> tensor dictionary 

4.  GPT-2 
-   ~Token Embedding lookup~
-   ~Positional Embedding lookup~
-   ~Stacking transformer blocks - aim to replicate GPT-2 small (124 M)~
-   ~Final projection + logits~
-   Greedy sampling 
-   Temperature + top-k sampling 

5.  Tokenizer 
-   BPE tokenizer? 
-   Load GPT-2 vocab and merge files 
-   Encoding: text -> token ids
-   Decoding: token ids -> text 

6.  End to End
-   Load model + tokenizer > accept a prompt > generate tokens in a loop > print the decoded output as they are generated. 

7.  Optimization 
-   KV Cache 
-   PCG random generation instead of mt19937
-   OpenMP parallelization
-   INT8 quantization 
-   Memory Mapped weight loading 
-   Change the way bz reductions are calculated; move to a recursive solution / hardcoded loops.
-   `bz::transpose` needs to go from returning a whole copy of the source tensor to returning a view. This is a major refactor of the entire base layer of code, it requires every single flat loop to become stride aware, and to be ableto handle non-contiguous tensors.





##  The NumPy `.npy` format.
`.npy` is a simple format for saving numpy arrays to disk with the full information about them. It is the standard binary format for persisting single arbitrary NumPy arrays on disk. The format stores all of the shape and dtype information necessary to reconstruct the array correctly even on another machine with a different architecture. 
`.npz` is the standard format for saving multiple numpy arrays on disk.


*From Claude* 
**the `.npy` format has three parts:**
1.  **Magic string and version**: every .npy file starts with a fixed sequence of bytes - a magic string `\x93NUMPY` folowed by two bytes for the major and minor version number. This is how numpy identifies the file as a valid .npy file. 
2.  **Header:** The header is a python dictionary literal stored as a string. It contains three bits of information: 
-   `descr`: the data type, `<f4` means little endian 32 bit float; what benzene uses.
-   `fortran_order`: Whether the data is stored in fortran / column major order. For bezene, `False.`
-   `shape`:    the shape of the tensor as a python tuple
The header string is padded with spaces to make the total length up to that point a multiple of 64 bytes. 
3.  **Raw Data**: Immediately after the header, the raw float bytes are written sequentially. 






##  The GGUF format. 
GGUF was developed by the llama.cpp project to store LLM weights. It was designed to be a self contained file that hold everything needed to run a model - the weights, architecture confguration and the tokenizer. 
There are four sections to a GGUF file: 

1.  Header: A fixed set of fields: A magic number GGUF, a version number, the number of tensors in the file, the number of metadata key-value pairs. 
2.  Metadata: a flat list of typed key-value pairs. Each entry has a key string, a type identifier and a value. THis is where the tokenizer vocabulary, merge rules and model configuration live. 
3.  Tensor info: a list of entries, one per tensor, each containing the tensor's name, shape, datatype, offset into the data section. 
4.  Tensor data: the raw weight bytes, each tensor aligned to 32 bytes. 










































