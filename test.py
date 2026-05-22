import numpy as np 
if __name__ == "__main__":
    x = np.random.randn(10, 10)
    print(x)
    np.save("test.npy", x)
