import numpy as np

if __name__ == "__main__":
    x = np.ones((5, 5))
    y = np.linalg.trace(x, offset=2)
    print(y)
