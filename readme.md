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
-   ~Causal Self Attention~
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


------------------------------------------
#   Theory
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




### Self Attention
From [Understanding and coding Multi-Head Attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention)


*   **Introduction** 

Self-Attention is the cornerstone of many SOTA deep learning models, particularly in the field of natural language processing. It has it's roots in the effort to improve Recurrent Neural Network Architectures in the handling of longer sequences and sentences. 

RNNs would have to process each sequence word-by-word or token-by-token. Attention mechanisms were introduced to operate  on  all sequence elements at each time step. 
The key is to be selective and determine which words are the most important in a specific context. The "Attention is all you need" paper proved that the standalone self-attention mechanism alone was enough for machine translation tasks. 

SA can be thought of as a mechanism that enhances the information of an input embedding by including information about the input's context. In other words, the **self-attention mechanism enables the model to weigh the importance of different elements in an input sequence and dynamically adjust their influence on the output.**
This is essential for language tasks, where the meaning of the word can change depending on the context it belongs in. 

The original paper introduced attention as "Scaled Dot Product Attention" or simply self-attention. There have been many variants of this since then.


**The step-by-step process for computing the self-attention:**

1.  Tokenize the sentence into an input sequence of tokens.
2.  Embed the input sequence, convert to a matrix of size `[sequence_length, embedding_dimension]`
3.  Define the weight matrices - `W_q, W_k, W_v`
4.  Calculate the unnormalized attention weights `omega_i = matmul(q_i, keys.Transpose)`
5.  Calculate the attention weights `alpha_i = softmax(sqrt(1/d_k)*omega_i)`
6.  Compute the context vector `z_i = matmul(alpha_i, values)`


*   **Embedding an Input Sequence** 

Any natural language sequence (a sentence of words) needs to be transformed into a sentence embedding before it is ingested by a neural net.

1.  The input sequence is represented as an integer-vector of tokens: `"Life is short, eat dessert first" -> [0, 4, 5, 2, 1, 3]`. 
2.  Using an embedding layer, the integer-vector is represented as a real-vector using the token-embedding lookup. `bz::nn::Embedding` provides. `[0, 4, 5, 2, 1, 3] -> [0.1234, 0.2345, 0.6352]`
    Embedding sizes typically range from hundreds to thousands of dimensions. Llama-2 uses embedding sizes of 4096. GPT-2 uses 768 dimensional vectors to represent its embeddings. 
3.  The embedding of the sequence is now a matrix of dimension `[sequence_len , embedding_dim]`. In this case, `[6, 3]`.



*   **Defining The Weight Matrices** 
Scaled Dot Product Attention (Self-Attention) utilizes 3 weight matrices - `W_q, W_k, W_v` - the query, key, value matrices. These weight matrices are adjusted during training. 
They serve to project the input into Q, K, V components of the sequence resepectively. 

The Q, K, V sequences are obtained via matmul between the weight matrices `W_i` and the embedded inputs `x`. 

The shapes of the weight matrices are: 

1.  $W_q$ : $[d, d_k]$
2.  $W_k$ : $[d, d_k]$
3.  $W_v$ : $[d, d_v]$


Then, calculate the queries, keys, values: 
1.  Query sequence: $q^{i} = x^{i}W_{q} \forall i \in [1, T]$
2.  Key sequence: $k^{i} = x^{i}W_{k} \forall i \in [1, T]$
3.  Value sequence: $v^{i} = x^{i}W_{v} \forall i \in [1, T]$

This is done for each token in the embedding sequence. 

In the above calculations, 
1.  $q^i$ is of a shape $[1, d_k]$
2.  $k^i$ is of a shape $[1, d_k]$
2.  $v^i$ is of a shape $[1, d_v]$

**d is the embedding dimension; or the size of each word vector x_i** 

-   Since the dot product between the query and the key vectors is being computed, these two vectors have to contain the same number of elements (`d_q == d_k`). 
-   In many LLMs, `d_q == d_k == d_v`. The number of elements in the value vector `v_i`, which determines the size of the resulting context vector can be arbitrary. 



*   **Computing the Unnormalized Attention Weights**
Now, let's suppose we are interested in computing the attention vector for the second input element, **the second input element acts as the query here.**  
The unnormalized attention weights for the query `q_i` and the jth input element `k_j` is given by  
        
$$\omega_{i}{j} = q^{i}k^{j}$$

The unnormalized attention weights for the input sequence as a whole is given by
`omega_i = matmul(q_i, keys.Transpose)`. The resultant shape is `omega_i: [1, sequence_length]`


*   **Computing the Attention Weights**
Normalizing the unnnormalized attention weights is done by scaling with `sqrt(1 / dk)` and applying softmax: 
`alpha_i = softmax(sqrt(1/d_k)*omega_i)`. The resultant shape is `alpha_i: [1, sequence_length]` 

The scaling factor ensures that the euclidean length of the weight vectors will approximately be the same magnitude. This prevents the attention weights from becoming too small or too large. 


*   **Computing the Context Vector**
The context vector for the ith element of the input sequence is: 

`z_i = matmul(alpha_i, values)`. The resultant shape is `z_i: [1, d_v]`



### Multi-Head Attention
Transformers use a module called MHA. 
In SA, the input sequence was transformedusing three matrices representing the query, key and value. These three matrices can be considered as a single attention head in the context of MHA. The MHA module contains multiple heads, each consisting of query, key, value matrices, similar to how a convolution layer has multiple kernel matrices, producing feature maps with multiple output channels. 

Self attention is initialized `num_heads` times with the same input shapes discussed above. The forward pass involves applying each self attention head to the input independently and the result from each head is concatentated along the last dimension (dv). The output of a MHA module is of the shape `[seq_len, num_heads]`


The advantage of MHA over SA with `dv > 1` is that each head in MHA can potentially learn to focus on different parts of the input sequence, capturing various aspects or relationships within the data. 


**Procedure to Compute the Multi-Head Attention of an input sequence:** 

1.  Tokenize the sentence into an input sequence of tokens.
2.  Embed the input sequence, convert to a matrix of size `[sequence_length, embedding_dimension]`
3.  Slice the embedded matrix into `num_heads` parts. Each Slice is now a matrix of size `[sequence_length, embedding_dimension / num_heads]`

4.  for(int i = 0; i < num_heads; i++)
    4.1 Define the input as - `input_i = input[:, i*(embedding_dimension/num_heads) : (i+i)*(embedding_dimension/num_heads)]`
    4.1 Define the weight matrices - `W_iq, W_ik, W_iv`
    4.2 Find the queries, keys, values by matmul - `queries_i = matmul(input_i, W_qi)`
    4.2 Calculate the unnormalized attention weights `omega = matmul(q, keys.Transpose)`
    4.3 Calculate the attention weights `alpha = softmax(sqrt(1/d_k)*omega_i)`
    4.4 Compute the context vector `z = matmul(alpha, values)`


Implementing efficient MHA: 

1.  Tokenize the sentence into an input sequence of tokens.
2.  Embed the input sequence, convert to a matrix of size `[sequence_length, embedding_dimension]`
3.  Define `w_k, w_q, w_v` with the the shapes: 
    ```
    w_k: [d_in, num_heads*d_kq]
    w_q: [d_in, num_heads*d_kq]
    w_v: [d_in, num_heads*d_kv]
    w_o: [num_heads*d_kv, d_in]
    ```
4.  Find the queries, keys, values by matmul:
    ```
    queries = matmul(input, w_q)    [batch_size, sequence_length, num_heads*d_kq]
    keys = matmul(input, w_k)       [batch_size, sequence_length, num_heads*d_kq]
    values = matmul(input, w_v)     [batch_size, sequence_length, num_heads*d_v] 
    ```

5.  Reshape the queries, keys, values: 
    ```
    queries-> [batch_size, sequence_length, num_heads, d_kq] 
    keys   -> [batch_size, sequence_length, num_heads, d_kq] 
    values -> [batch_size, sequence_length, num_heads, d_v]  
    ```
6.  Transpose the queries, keys, values: 
    ```
    queries-> [batch_size, num_heads, sequence_length, d_kq] 
    keys   -> [batch_size, num_heads, sequence_length, d_kq] 
    values -> [batch_size, num_heads, sequence_length, d_v]  
    ```

7.  Compute attention across all heads: 
    ```
    attn_scores = matmul(queries, keys.T) → [batch, num_heads, seq_len, seq_len]
    add causal mask
    attn_weights = softmax(attn_scores / sqrt(d_kq))
    context = matmul(attn_weights, values) → [batch, num_heads, seq_len, d_v]
    ```
8.  Transpose back to `[batch_size, sequence_length, num_heads, d_v]`
9.  Reshape to `[batch_size, sequence_length, num_heads*d_v]`
10. Output = `matmul(context, w_o) -> [batch_size, sequence_length, d_in]`


### Causal Self Attention
CSA is used by GPT style decoder only LLMs to generate text. CSA is also called Masked Self Attention. 
CSA ensures that the outputs for a certain position in a sequence is based only on the known outputs at previous positions and not on future positions. 
**It ensures that the prediction for each next word should only depend on the words that came before it**.


**Procedure to Compute the Causal Self Attention of an input sequence:** 

1.  Tokenize the sentence into an input sequence of tokens.
2.  Embed the input sequence, convert to a matrix of size `[sequence_length, embedding_dimension]`
3.  Define the weight matrices - `W_q, W_k, W_v`
4.  Calculate `queries = matmul(input, W_q)`
5.  Calculate the unnormalized attention weights `omega = matmul(queries, keys.Transpose)`
6.  Calculate the masked attention scores: `omega_masked = omega * mask`. The mask is a matrix which is zeroed out along a diagonal depending on the index of the element in the input sequence.
7.  Calculate the attention weights `alpha = softmax(sqrt(1/d_k)*omega)`
8.  Compute the context vector `z_i = matmul(alpha_i, values)`
























