import numpy as np
from numpy.core.fromnumeric import argmin
import handin1 as func
import matplotlib.pyplot as plt


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def trust_region_subproblem(gradient, hessian, lambda_0, delta, max_iter=1000):
    d = hessian.shape[0]
    lambda_j, vector = np.linalg.eig(hessian)
    lambda_1 = min(lambda_j)
    min_index = argmin(lambda_j)
    lam = lambda_0
    p = 0
    count = 0
    while lam >= -lambda_1:
        count += 1
        B_ = hessian + lam * np.identity(d)
        # try:
        if vector[:, min_index].T @ gradient == 0:
            p = hard_case(gradient, hessian, delta, lambda_j, vector, min_index)
        else:
            p = np.linalg.inv(B_) @ (-gradient)

        R = np.linalg.cholesky(B_)
        q = np.linalg.inv(R.T) @ p

        lam_new = lam + (np.linalg.norm(p) / np.linalg.norm(q)) ** 2 * (
            (np.linalg.norm(p) - delta) / delta
        )
        if lam_new == lam or (count >= max_iter and np.linalg.norm(p) <= delta):
            # return p
            break
        else:
            lam = lam_new
    return p


def hard_case(gradient, hessian, delta, lambda_j, vector, min_index):
    temp = gradient.T @ hessian @ gradient
    tau = min(1, np.linalg.norm(gradient) ** 3 / (delta * temp))
    z = vector[:, min_index]  # eigenvector of B corresponding to lamda_1
    lamda_1 = lambda_j[min_index]
    p = 0
    for i in range(np.shape(lambda_j)[0]):
        if lamda_1 != lambda_j[i]:
            p += -((vector[:, i].T @ gradient) / (lambda_j[i] - lamda_1) * vector[:, i])
    p += tau * z
    return p


def cauchy_point(gradient, hessian, delta):
    temp = gradient.T @ hessian @ gradient
    if temp <= 0:
        tau = 1
    else:
        tau = min(1, np.linalg.norm(gradient) ** 3 / (delta * temp))
    return -tau * delta / np.linalg.norm(gradient) * gradient


def trust_region(_x, fun, fun_g, fun_h, _delta, delta_max, eta=0.2, max_iter=1000):
    xs = [_x]
    deltas = [_delta]
    x = _x
    delta = _delta
    count = 0
    while count < max_iter:
        count += 1
        g = fun_g(x)
        B = fun_h(x)
        if is_pos_def(B):
            if np.linalg.norm(np.linalg.inv(B) @ g) <= delta:
                p = -np.linalg.inv(B) @ g
            else:
                p = trust_region_subproblem(g, B, 0, delta)
        else:
            p = trust_region_subproblem(g, B, -min(np.linalg.eigvals(B))+1e-5, delta)  # set the first lambda to -lambda_1

        rho = (fun(x) - fun(x + p)) / (- g.T @ p - 0.5 * p.T @ B @ p)  # should we use (4.1)? t?

        if rho < 1 / 4:
            delta = 1 / 4 * delta
        else:
            if rho > 3 / 4 and np.linalg.norm(p) == delta:
                delta = min(2 * delta, delta_max)

        if rho > eta:
            x = x + p
        else:
            break  # x_k = x_k+1

        xs.append(x)
        deltas.append(delta)

    return np.array(xs), fun(x), count, np.array(deltas)


def cal_rate_convergence(xs, fun):
    x_ = xs[-1]
    x1 = xs[:-1]
    x2 = xs[1:]
    res = []
    for i in range(len(x1)):
        if fun(x1[i])-fun(x_) == 0:
            continue
        res.append((fun(x2[i])-fun(x_))/(fun(x1[i])-fun(x_)))
    return np.array(res)


def plot_convergence(xs, fun, label="", color='b', name=""):
    # _x = xs[-1]
    # xx = cal_convergence(xs)
    xx = cal_rate_convergence(xs, fun)
    it = np.arange(xx.shape[0]-1)
    plt.plot(it, xx[:-1], c=color, label=label)
    # plt.legend()
    plt.yscale("log")
    plt.title(name)
    plt.xlabel("iteration")
    plt.ylabel("convergence rate of x")


def plot_deltas(dels, label="", name="", color="b"):
    it = np.arange(dels.shape[0])
    plt.plot(it, dels, c=color, label=label)
    plt.title(name)
    plt.xlabel("iteration")
    plt.ylabel("delta")


def test_1(x):
    print("Ellipsoid test")
    xs, val, count, dels = trust_region(np.array(x), func.ellipsoid, func.ellipsoid_d1, func.ellipsoid_d2, 0.5, 5)
    # plot_convergence(xs, func.ellipsoid, name="convergence rate of f1")
    plot_deltas(dels, name="delta changes in f1")
    plt.show()
    print(count)


def test_2(x):
    print("F2 test")
    xs, val, count, dels = trust_region(np.array(x), func.Rosenbrock, func.Grad_Ros, func.Hessian_Ros, 0.5, 5)
    # plot_convergence(xs, func.ellipsoid, name="convergence rate of f2", label="delta=0.5, max_delta=1")
    plot_deltas(dels, name="delta changes in f2")
    plt.show()
    print(count)


def test_3(x):
    print("Log-ellipsoid test")
    xs, val, count, dels = trust_region(np.array(x), func.log_ellipsoid, func.log_ellipsoid_d1, func.log_ellipsoid_d2, 0.5, 5)
    # plot_convergence(xs, func.ellipsoid, name="convergence rate of f3")
    plot_deltas(dels, name="delta changes in f3")
    plt.show()
    print(count)


def test_4(x):
    print("F4 test")
    xs, val, count, dels = trust_region(np.array(x), func.f_4, func.f_4grad, func.f_4hessian, 0.5, 5)
    # plot_convergence(xs, func.ellipsoid, name="convergence rate of f4")
    plot_deltas(dels, name="delta changes in f4")
    plt.show()
    print(count)


def test_5(x):
    print("F5 test")
    xs, val, count, dels = trust_region(np.array(x), func.f_5, func.f_5grad, func.f_5hessian, 0.5, 5)
    # plot_convergence(xs, func.ellipsoid, name="convergence rate of f5")
    plot_deltas(dels, name="delta changes in f5")
    plt.show()
    print(count)


x0 = [2.0, 5.0, 10.0, 4.0, 8.0]
x1 = [2.0, 4.0]
test_1(x0)
test_2(x1)
test_3(x0)
test_4(x0)
test_5(x0)


