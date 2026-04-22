import numpy as np 
import torch 

rows = 128
cols = 768
# x = np.random.randn(2,3,4,5)
if __name__ == "__main__":
    x = torch.randn(rows, cols)
    l1 = torch.nn.Linear(768, 394)
    l2 = torch.nn.Linear(394, 192)
    l3 = torch.nn.Linear(192, 96)
    l4 = torch.nn.Linear(96, 10)
    out = x 

    import timeit
    start = timeit.default_timer()
    out = l1(out)
    out = l2(out)
    out = l3(out)
    out = l4(out)
    end = timeit.default_timer()
    print(end - start)
    print(out.shape)

