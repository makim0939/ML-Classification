import numpy as np
import matplotlib as plt
from common.functions import gradient_descent

# f(x1, x2) = x1^2 + x2^2
def test_f(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
min = gradient_descent(test_f, init_x, lr=0.1, step_num=100)
print(min)

