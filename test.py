import numpy as np 
import torch 

# x = np.random.randn(2,3,4,5)
if __name__ == "__main__":
    x = torch.ones(2,10,5)
    y = torch.ones(10,5)
    z = x + y
    print(z)

