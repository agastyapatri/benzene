import numpy as np 
import torch 

# x = np.random.randn(2,3,4,5)
if __name__ == "__main__":
    x = torch.randn(2,3,4)
    y = torch.randn(4,3,2)
    z = torch.matmul(x, y)
    print(z.shape)

