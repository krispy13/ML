import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([2, 4, 6, 8]).reshape((-1, 1))
y_actual = np.array([3, 7, 5, 10]).reshape((-1, 1))
slope = np.zeros(10)
error = np.zeros(10)
alpha = 0.01

mean_x = np.mean(x)
mean_y = np.mean(y_actual)

n = len(x)

numer = 0
denom = 0
for i in range(n):
    numer += (x[i] - mean_x) * (y_actual[i] - mean_y)
    denom += (x[i] - mean_x) ** 2

m = numer / denom

for k in range(10):
    c = mean_y - (m * mean_x)
    y_new = m * x + c
    sum = 0
    for i in range(1, n):
        sum += x[i] * (y_new[i] - y_actual[i])

    slope[k] = m
    error[k] = 0
    for i in range(n):
        error[k] += (y_new[i] - y_actual[i])**2
    # print(y_new, y_actual, error)
    print(f'At slope = {m}, the error is {error[k]}')
    m = m + alpha * sum

plt.plot(slope, error)
plt.title("Slope M vs Error E graph")
plt.ylabel("E")
plt.xlabel("M")
plt.show()
