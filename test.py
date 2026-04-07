import numpy as np 

# x = np.random.randn(2,3,4,5)
x = np.random.randn(2,3)
print(x.mean(axis = 0).shape)
print(x.mean(axis = 1).shape)
print(x.mean(axis = 2).shape)
print(x.mean(axis = 3).shape)

