import numpy as np
import copy

a = np.zeros(5)
a = np.array([1 ,2 ,3 , 4, 5])
b = np.zeros(5)
b = a
a[0] = 100
print(a)
print(b)