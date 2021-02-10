import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# x is an array
def ellipsoid(x, alpha=1000):
    d = len(x)
    res = 0
    for i in range(d):
        res += alpha**(i/(d-1)) * x[i]**2
    return res


def ellipsoid_d1(x, alpha=1000):
    d = len(x)
    res = []
    for i in range(d):
        res.append(2 * alpha**(i/(d-1)) * x[i])
    return res


def ellipsoid_d2(x, alpha=1000):
    d = len(x)
    res = []
    for i in range(d):
        tmp = []
        for j in range(d):
            if i == j:
                tmp.append(2 * alpha ** (i / (d - 1)))
            else:
                tmp.append(0)
        res.append(tmp)
    return res


def log_ellipsoid(x, alpha=1000, epsilon=1e-16):
    elli = ellipsoid(x, alpha)
    return np.log(epsilon+elli)


def log_ellipsoid_d1(x, alpha=1000, epsilon=1e-16):
    d = len(x)
    elli = ellipsoid(x, alpha)
    res = []
    for i in range(d):
        res.append((2*alpha**(i/(d-1))*x[i])/(epsilon+elli))
    return res


def log_ellipsoid_d2(x, alpha=1000, epsilon=1e-16):
    d = len(x)
    res = []
    elli = ellipsoid(x, alpha)
    for i in range(d):
        tmp = []
        for j in range(d):
            if i == j:
                tmp.append((2*alpha**(i/(d-1))*(epsilon+elli)
                           - (2*alpha**(i/(d-1))*x[i])**2)/(epsilon+elli)**2)
            else:
                tmp.append((-2*alpha**(i/(d-1))*x[i] * 2*alpha**(j/(d-1))*x[j])
                           / (epsilon+elli)**2)
        res.append(tmp)
    return res


def plotting(x, y, z, name, st=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xx = np.arange(-x, x, step=st)
    yy = np.arange(-y, y, step=st)
    Y, X = np.meshgrid(xx, yy)
    # print(X, Y)
    Z = np.array(z)
    Z = np.reshape(Z, [2*int(x/st), 2*int(y/st)])
    print(Z)
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel(name)
    # plt.show()


def test1():
    x = []
    for x1 in range(-50, 50):
        for x2 in range(-50, 50):
            x.append([x1/10.0, x2/10.0])
    x = np.array(x)
    res = list(map(ellipsoid, x))
    # print(ellipsoid([-5, -4.9]))
    plotting(5, 5, res, "ellipsoid")


def test2():
    x = []
    for x1 in range(-50, 50):
        for x2 in range(-50, 50):
            x.append([x1/10.0, x2/10.0])
    x = np.array(x)
    res = list(map(log_ellipsoid, x))
    plotting(5, 5, res, "log_ellipsoid")


test1()
test2()

plt.show()

print(log_ellipsoid_d1([1, 2, 3]))
print(log_ellipsoid_d2([1, 2, 3]))
