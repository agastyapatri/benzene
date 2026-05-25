import numpy as np 
if __name__ == "__main__":
    with open("test.npy", "rb") as file: 
        x = np.load(file)
    print(x.shape)
