import numpy as np 
import torch 

rows = 1
cols = 784
# x = np.random.randn(2,3,4,5)
if __name__ == "__main__":
    l1 = torch.nn.Linear(784, 394)
    l2 = torch.nn.Linear(394, 192)
    l3 = torch.nn.Linear(192, 96)
    l4 = torch.nn.Linear(96, 10)

    net = torch.nn.Sequential(l1, l2, l3, l4)

    samples: int = 0 
    import timeit
    start = timeit.default_timer()
    while(samples < 60000): 
        x = torch.randn(rows, cols)
        out = net(x)
        samples += rows 
    end = timeit.default_timer()
    print(f"time taken to process the mnist dataset in python: {end - start} seconds")
