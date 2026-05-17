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
-   concat 
-   Tensor I/O 

2.  Neural Network Primitives 
-   ~Linear Layer~
-   ~GELU~
-   ~Causal Self Attention~
-   ~Multi Head Self Attention~
-   Feed Forward Networks 
-   Transformer Block - attention + ffn + layernorm

3.  Load Weights 
-   Loading GPT-2 weights; support GGUF? safetensor? 
-   Weight map : name -> tensor dictionary 

4.  GPT-2 
-   Token Embedding lookup 
-   Positional Embedding lookup 
-   Stacking transformer blocks - aim to replicate GPT-2 small (~120 M)
-   Final projection + logits 
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



##  Immediate Concerns 
1.  ~Learn about and implement the `bz::nn::MultiheadAttention` module~
2.  ~MLP Block: `bz::nn::Linear` and `bz::nn::GELU` stacked together (trivial)~
3.  Implement the transformer block; test how well `bz::nn::Sequential` works with all modules created up to this point.
4.  Build `GPT-2 Small` architecture while making sure to understand the decisions made during the building of this model. Testing of the forward pass will be done with dummy data taken from microgpt-c.
5.  Loading GPT-2 weights from GGUF. Have to figure out the deserialization process. This will be a bit of fun.



### Concerns after that 
6.  Tokenizing. 
7.  Actual inference 


