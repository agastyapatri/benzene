import numpy as np 
import torch 

# x = np.random.randn(2,3,4,5)
if __name__ == "__main__":
    x = torch.nn.Linear(in_features=5, out_features=10)
    relu = torch.nn.Tanh();
    print(relu.state_dict())

