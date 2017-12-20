import numpy as np
import math

a = np.array([1, 2, 3])


def base_sigmoid(x):
    return 1 / (1 + math.exp(-x))


print(base_sigmoid(3))


# 0 - 1之间的函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


print(sigmoid(a))


def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds


print("sigmoid_derivative(x) = " + str(sigmoid_derivative(a)))


def image2vector(x):
    return x.reshape(x.shape[0] * x.shape[1] * x.shape[2], 1)


image = np.array([[[0.67826139, 0.29380381],
                   [0.90714982, 0.52835647],
                   [0.4215251, 0.45017551]],

                  [[0.92814219, 0.96677647],
                   [0.85304703, 0.52351845],
                   [0.19981397, 0.27417313]],

                  [[0.60659855, 0.00533165],
                   [0.10820313, 0.49978937],
                   [0.34144279, 0.94630077]]])

print(image2vector(image))


def normalizeRows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norm
    return x


b = np.array([
    [0, 3, 4],
    [1, 6, 4],
])
print(normalizeRows(b))


def L1(y_hat, y):
    loss = np.sum(np.abs(np.subtract(y, y_hat)))
    return loss


def L2(y_hat, y):
    loss = np.sum(np.square(np.subtract(y, y_hat)))
    return loss
