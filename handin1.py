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
    return np.array(res)


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
    return np.array(res)


def log_ellipsoid(x, alpha=1000, epsilon=1e-16):
    elli = ellipsoid(x, alpha)
    return np.log(epsilon+elli)


def log_ellipsoid_d1(x, alpha=1000, epsilon=1e-16):
    d = len(x)
    elli = ellipsoid(x, alpha)
    res = []
    for i in range(d):
        res.append((2*alpha**(i/(d-1))*x[i])/(epsilon+elli))
    return np.array(res)


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
    return np.array(res)


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
    ax.contourf(X, Y, Z, 100)
    ax.set_title(name+" function")
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


# test1()
# test2()
#
# plt.show()

# print(log_ellipsoid_d2([1, 2, 3]))


# by Josefine
# The Rosenbrock Banana Function
def Rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# The Gradient for the Rosenbrock Banana Function
def Grad_Ros(x):
    return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])


# The Hessian for the Rosenbrock Banana Function
def Hessian_Ros(x):
    return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])


def test3():
    n = 100  # length of x1,x2
    x1 = np.linspace(-1.5, 1.5, num=n)
    x2 = np.linspace(-1, 3, num=n)
    x = np.meshgrid(x1, x2, sparse=True)
    ax = plt.axes(projection='3d')
    ax.contourf(x1, x2, Rosenbrock(x), 100)
    ax.set_title('Rosenbrock Banana Function')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Rosenbrock')
    plt.show()


# test3()
# by Niels

q = 10**8


def h(x):
    return np.log(1+np.exp(q*x)) / q


def h_safe(x):
    '''
    Overflow safe version of the h function
    '''
    return (np.log(1+np.exp(-np.abs(q*x))) + max(q*x, 0)) / q


def f_4(x):
    if isinstance(x, int) or isinstance(x, float):
        return h_safe(x) + 100*h_safe(-x)
    else:
        d = len(x)
        result = 0
        for i in range(d):
            result += h_safe(x[i]) + 100*h_safe(-x[i])
        return result


def f_5(x):
    if isinstance(x, int) or isinstance(x, float):
        return h_safe(x)**2 + 100*h_safe(-x)**2
    else:
        d = len(x)
        result = 0
        for i in range(d):
            result += h_safe(x[i])**2 + 100*h_safe(-x[i])**2
        return result


# xs4 = np.linspace(-0.5,0.5)
# ys4 = [f_4(x) for x in xs4]
#
# xs5 = np.linspace(-0.5,0.5)
# ys5 = [f_5(x) for x in xs5]
#
# plt.plot(xs4, ys4,c='r',label=r'$f_4(x)$')
# plt.xlabel('x')
# plt.ylabel(r'$f_4(x)$')
# plt.legend()
# plt.show()
#
# plt.plot(xs5, ys5,c='r',label=r'$f_5(x)$')
# plt.xlabel('x')
# plt.ylabel(r'$f_5(x)$')
# plt.legend()
# plt.show()


def plotting3D(x, y, z, name, st=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xx = np.arange(-x, x, step=st)
    yy = np.arange(-y, y, step=st)
    Y, X = np.meshgrid(xx, yy)
    # print(X, Y)
    Z = np.array(z)
    Z = np.reshape(Z, [2*int(x/st), 2*int(y/st)])
    # print(Z)
    ax.plot_surface(X, Y, Z)
    ax.set_title(name+" function")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel(name)


def f_4_plot():
    x = []
    for x1 in range(-50, 50):
        for x2 in range(-50, 50):
            x.append([x1/10.0, x2/10.0])
    x = np.array(x)
    res = list(map(f_4, x))
    plotting3D(5, 5, res, r'$f_4(x)$')


# f_4_plot()
# plt.show()


def f_5_plot():
    x = []
    for x1 in range(-50, 50):
        for x2 in range(-50, 50):
            x.append([x1/10.0, x2/10.0])
    x = np.array(x)
    res = list(map(f_5, x))
    plotting3D(5, 5, res, r'$f_5(x)$')


# f_5_plot()
# plt.show()

# q = 20


# def h(x):
#     return np.log(1 + np.exp(q * x)) / q
#
#
# def h_safe(x):
#     '''
#     Overflow safe version of the h function
#     '''
#     return (np.log(1 + np.exp(-np.abs(q * x))) + max(q * x, 0)) / q
#

def h_d1(x):
    return np.exp(x * q) / (1 + np.exp(x * q))


def h_d2(x):
    return (q * np.exp(x * q)) / (1 + np.exp(x * q)) ** 2


def f_4grad(x):
    if isinstance(x, int) or isinstance(x, float):
        return h_d1(x) - 100 * h_d1(-x)
    else:
        d = len(x)
        grad = np.zeros(d)
        for i in range(d):
            grad[i] = h_d1(x[i]) - 100 * h_d1(-x[i])
        return grad


def f_4hessian(x):
    if isinstance(x, int) or isinstance(x, float):
        return h_d2(x) + 100 * h_d2(-x)
    else:
        d = len(x)
        hessian = np.zeros((d, d))
        for i in range(d):
            hessian[i, i] = h_d2(x[i]) + 100 * h_d2(-x[i])
        return hessian


def f_5grad(x):
    if isinstance(x, int) or isinstance(x, float):
        return 2 * h(x) * h_d1(x) - 100 * 2 * h(-x) * h_d1(-x)
    else:
        d = len(x)
        grad = np.zeros(d)
        for i in range(d):
            grad[i] = 2 * h(x[i]) * h_d1(x[i]) - 100 * 2 * h(-x[i]) * h_d1(-x[i])
        return grad


def f_5hessian(x):
    if isinstance(x, int) or isinstance(x, float):
        return 2 * np.exp(q * x) * (np.exp(q * x) + np.log(np.exp(q * x) + 1)) / (
                    np.exp(q * x) + 1) ** 2 + 200 * np.exp(-2 * q * x) * (
                           np.exp(q * x) * np.log(np.exp(-q * x) + 1) + 1) / (np.exp(-q * x) + 1) ** 2
    else:
        d = len(x)
        hessian = np.zeros((d, d))
        for i in range(d):
            hessian[i, i] = 2 * np.exp(q * x[i]) * (np.exp(q * x[i]) + np.log(np.exp(q * x[i]) + 1)) / (
                    np.exp(q * x[i]) + 1) ** 2 + 200 * np.exp(-2 * q * x[i]) * (
                           np.exp(q * x[i]) * np.log(np.exp(-q * x[i]) + 1) + 1) / (np.exp(-q * x[i]) + 1) ** 2
        return hessian


# ep = 1e-10
# print("-----f1-----")
# print((ellipsoid([1, 2])-ellipsoid([1-ep, 2]))/ep)
# print((ellipsoid([1, 2])-ellipsoid([1, 2-ep]))/ep)
# print(ellipsoid_d1([1, 2]))
# print('\n')
# print((ellipsoid_d1([1, 2])-ellipsoid_d1([1-ep, 2]))/ep)
# print((ellipsoid_d1([1, 2])-ellipsoid_d1([1, 2-ep]))/ep)
# print(ellipsoid_d2([1, 2]))
#
# print("-----f2-----")
# print((Rosenbrock([1, 2]) - Rosenbrock([1-ep, 2]))/ep)
# print((Rosenbrock([1, 2]) - Rosenbrock([1, 2-ep]))/ep)
# print(Grad_Ros([1, 2]))
# print('\n')
# print((Grad_Ros([1, 2])-Grad_Ros([1-ep, 2]))/ep)
# print((Grad_Ros([1, 2])-Grad_Ros([1, 2-ep]))/ep)
# print(Hessian_Ros([1, 2]))
#
# print("-----f3-----")
# print((log_ellipsoid([1, 2])-log_ellipsoid([1-ep, 2]))/ep)
# print((log_ellipsoid([1, 2])-log_ellipsoid([1, 2-ep]))/ep)
# print(log_ellipsoid_d1([1, 2]))
# print('\n')
# print((log_ellipsoid_d1([1, 2])-log_ellipsoid_d1([1-ep, 2]))/ep)
# print((log_ellipsoid_d1([1, 2])-log_ellipsoid_d1([1, 2-ep]))/ep)
# print(log_ellipsoid_d2([1, 2]))
#
# ep1 = 1e-15
# print("-----f4-----")
# print((f_4([1e-8, 2e-8])-f_4([1e-8-ep1, 2e-8]))/ep1)
# print((f_4([1e-8, 2e-8])-f_4([1e-8, 2e-8-ep1]))/ep1)
# print(f_4grad([1e-8, 2e-8]))
# print('\n')
# print((f_4grad([1e-8, 2e-8])-f_4grad([1e-8-ep1, 2e-8]))/ep1)
# print((f_4grad([1e-8, 2e-8])-f_4grad([1e-8, 2e-8-ep1]))/ep1)
# print(f_4hessian([1e-8, 2e-8]))
#
# print("-----f5-----")
# print((f_5([1e-8, 2e-8])-f_5([1e-8-ep1, 2e-8]))/ep1)
# print((f_5([1e-8, 2e-8])-f_5([1e-8, 2e-8-ep1]))/ep1)
# print(f_5grad([1e-8, 2e-8]))
# print('\n')
# print((f_5grad([1e-8, 2e-8])-f_5grad([1e-8-ep1, 2e-8]))/ep1)
# print((f_5grad([1e-8, 2e-8])-f_5grad([1e-8, 2e-8-ep1]))/ep1)
# print(f_5hessian([1e-8, 2e-8]))
