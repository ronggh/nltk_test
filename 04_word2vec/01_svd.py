import numpy as np

if __name__ == "__main__":
    la = np.linalg
    words = ["I","like","enjoy","deep","learning","NLP","flying","."]
    x = np.array([[0,2,1,0,0,0,0,0],
                  [2,0,0,0,1,1,0,0],
                  [1,0,0,0,0,0,1,0],
                  [0,1,0,0,1,0,0,0],
                  [0,0,0,1,0,0,0,1],
                  [0,1,0,0,0,0,0,1],
                  [0,0,1,0,1,0,0,1],
                  [0,0,0,0,1,1,1,0] ])
    u,s,vh = la.svd(x,full_matrices=False)
    print("u:")
    print(u)
    print("s:")
    print(s)
    print("vh:")
    print(vh)