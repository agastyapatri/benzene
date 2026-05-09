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



NOTES: 

1.  `bz::layernorm` and `bz::nn::LayerNorm` currently only handle normzalizing over the last dimension




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
-   Causal Self Attention
-   Multi Head Self Attention
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
1.  Learn about and implement the `bz::nn::MultiheadAttention` module 
2.  MLP Block: `bz::nn::Linear` and `bz::nn::GELU` stacked together (trivial)
3.  Implement the transformer block; test how well `bz::nn::Sequential` works with all modules created up to this point.
4.  Build `GPT-2 Small` architecture while making sure to understand the decisions made during the building of this model. Testing of the forward pass will be done with dummy data taken from microgpt-c.
5.  Loading GPT-2 weights from GGUF. Have to figure out the deserialization process. This will be a bit of fun.


### Concerns after that 
6.  Tokenizing. 
7.  Actual inference 











##  GPT-2 Layers and Structure 
-   Vocabulary: 50257 
-   Embedding dimension: 768 
-   Maximum sequence length: 1024 tokens 
-   Transformer Layers: 12

GPT-2 is built upon a stack of 12 identical transformer decoder blocks. Unlike the original "Attention is all you need" transformer architecture, which uses both and encoder and a decoder, GPT-2 uses only the decoder stack. 

1. **Input Processing Layers**
Before reaching the transformer blocks, the input goes through two primary layers: 

- **Token Embedding Layer:** maps the token IDs into dense vectors of dimension 768 
- **Positional Embedding Layer:** Adsd a learnable vector to the token embedding, providing the model with information about the position of each token in the sequence, since self attention is position agnostic.

2.  **12 Transformer Blocks**
Each of the 12 blocks contains several sub-layers, structured using pre-normalization (LayerNorm is applied before the sub layers rather than afer)
-   **Layer Norm**: Ensures stable gradients by normalizing the hidden states 
-   **Masked Multi Head Self Attention**: this mechanism allows each token to attend to previous tokens in the sequence. It is masked to prevent the model from looking into the future tokens during training, which preserves the auto regressive quality. It projects the input into Query, Key and Value matrices.
-   **Feed Forward Network**: A two layer nerual net that processes each token independently. It typically consists of a linear projection that expands the dimension (usually to 4x768 = 3072), A GeLU activation function, A linear projection that projects back to the original dimension
-   **Residual Connections**: Add operations that bypass sub layers.


3.  **Output Layers**: After passing through 12 transformer blocks, the output is processed by the final layers to produce predictions.
-   The final layer norm is applied after the last transformer block. 
-   Language Modeling linear layer: the final 768 dim hidden state is projected back into the vocabulary size to produce logits, which are then passed through a softmax function to generate the probability distribution for the next token.



