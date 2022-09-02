import numpy as np
import matplotlib.pyplot as plt


def draw(x1, x2):
    ln = plt.plot(x1, x2)
    plt.pause(0.0001)
    ln[0].remove()

def sigmoid(z):
    return 1/(1+np.exp(-z))


def calculate_error(line_param, points, y):
    m = points.shape[0]
    p = sigmoid(points*line_param)
    cross_entropy = -(1/m)*(np.log(p).T*y + (1-p).T*(1-y))
    return cross_entropy


def gradient_descent(line_param, pointes, y, alpha, iterations):
    m = pointes.shape[0]
    history = []
    for i in range(iterations):
        p = sigmoid(pointes*line_param)
        gradient = (pointes.T*(p-y))*(alpha/m)
        line_param = line_param-gradient
        w1 = line_param.item(0)
        w2 = line_param.item(1)
        b = line_param.item(2)
        x1 = np.array([pointes[:, 0].min(), pointes[:, 0].max()])
        x2 = -b / w2+x1*(-w1 / w2)
        history.append(float(calculate_error(line_param, pointes, y)))
        draw(x1, x2)
    return history


points=100
np.random.seed(2222)
bias = np.ones(points)
top_region=np.array([np.random.normal(10, 2, points), np.random.normal(12, 2, points), bias]).T
bottom_region=np.array([np.random.normal(5, 2, points), np.random.normal(6, 2, points), bias]).T
all_points=np.vstack((top_region, bottom_region))
line_parameters=np.matrix(np.zeros(3)).T
# x1 = np.array([bottom_region[:, 0].min(), top_region[:, 0].max()])
# x2 = -b / w2+x1*(-w1 / w2)
linear_compination=all_points*line_parameters
probabilities=sigmoid(linear_compination)
y=np.array([np.zeros(points), np.ones(points)]).reshape(points*2, 1)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(top_region[:, 0], top_region[:, 1], c='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], c='b')
history = gradient_descent(line_parameters, all_points, y, 0.09, 1000)
plt.show()

plt.plot(range(0,len(history)),history)