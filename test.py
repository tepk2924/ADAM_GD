import numpy as np

A = np.array([[1, 1, 1],
              [5, 6, 7],
              [8, 7, 10]])

B = np.diff(A, axis=0)

C = np.linalg.norm(B, axis=1)

print(B)
print(C)