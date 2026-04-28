import numpy as np 
import torch 

rows = 1024
cols = 784
# x = np.random.randn(2,3,4,5)
if __name__ == "__main__":
    x = torch.randn(5,5,5)
    x = x.unsqueeze(1)
    x = x.unsqueeze(1)
    print(x.shape);
