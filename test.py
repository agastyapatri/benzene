import numpy as np 
import torch 

# x = np.random.randn(2,3,4,5)
if __name__ == "__main__":
    x = torch.ones(2,2,3,4)
    softmax = torch.nn.Softmax(dim = 0)
    print(softmax(x))

