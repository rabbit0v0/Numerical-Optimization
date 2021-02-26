import numpy as np
import handin1 as func
import matplotlib.pyplot as plt
import time


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def backtracking(p, x, fun, fun_g, alpha=1.0, rho=0.6, c=1e-4):
    while fun(x+alpha*p) > fun(x) + c * alpha * np.dot(fun_g(x), p):
        alpha = rho * alpha
    return alpha


def plot_distance(xs, name):
    _x = xs[-1]
    it = np.arange(xs.shape[0])
    plt.plot(it, np.linalg.norm(xs-_x, axis=1), 'b')
    plt.yscale("log")
    plt.title(name)
    plt.xlabel("iteration")
    plt.ylabel("distance of x & x*")
    plt.show()


# def cal_convergence(xs, alpha=1.0):
#     d = xs.shape[1]
#     res = []
#     for x in xs:
#         res.append(x.T @ np.diag(alpha ** (np.arange(d)/4)) @ x)
#     return np.array(res)

def cal_rate_convergence(xs, fun):
    x_ = xs[-1]
    x1 = xs[:-1]
    x2 = xs[1:]
    res = []
    for i in range(len(x1)):
        res.append((fun(x2[i])-fun(x_))/(fun(x1[i])-fun(x_)))
    return np.array(res)


def plot_convergence(xs, fun, label, color='b'):
    # _x = xs[-1]
    # xx = cal_convergence(xs)
    xx = cal_rate_convergence(xs, fun)
    it = np.arange(xs.shape[0]-2)
    plt.plot(it, xx[:-1], c=color, label=label)
    plt.legend()
    plt.yscale("log")
    # plt.title(name)
    plt.xlabel("iteration")
    plt.ylabel("convergence rate of x")
    # plt.show()


def plot_gradient(g, name):
    it = np.arange(g.shape[0])
    plt.plot(it, np.linalg.norm(g, axis=1), 'b')
    plt.yscale("log")
    plt.title(name)
    plt.xlabel("iteration")
    plt.ylabel("gradient")
    plt.show()


def steepest_descent(_x, fun, fun_g, max_iter=1000, name=""):
    x = _x
    count = 0
    gradient = []
    xs = []
    while np.linalg.norm(fun_g(x)) > 1e-6 and count < max_iter:
        p = -fun_g(x)
        alpha = backtracking(p, x, fun, fun_g)
        # alpha = backtracking_LS(x, fun, fun_g, p)
        x += alpha * p
        xx = list(x)
        gradient.append(fun_g(x))
        xs.append(xx)
        count += 1
    # print("steepest steps:", count)
    plot_gradient(np.array(gradient), "Steepest Descent - "+name)
    plot_distance(np.array(xs), "Steepest Descent - "+name)
    # plot_convergence(np.array(xs), fun, "Steepest Descent", 'c')
    # return x, fun(x), count
    return x


def newton(_x, fun, fun_g, fun_h, beta=1e-3, max_iter=1000, name=""):
    x = _x
    count = 0
    gradient = []
    xs = []
    while np.linalg.norm(fun_g(x)) > 1e-8 and count < max_iter:
        hessian = fun_h(x)
        if not is_pos_def(hessian):
            diag = hessian.diagonal()
            min_diag = np.amin(diag)
            tol = 0
            if min_diag <= 0:
                tol = -min_diag + beta
            d = hessian.shape[0]
            hessian_modi = hessian + tol * np.identity(d)
            while not is_pos_def(hessian_modi):
                tol = max(2*tol, beta)
                hessian_modi = hessian + tol * np.identity(d)
        else:
            hessian_modi = hessian
        p = - np.linalg.inv(hessian_modi) @ fun_g(x)
        alpha = backtracking(p, x, fun, fun_g)
        x += alpha * p
        xx = list(x)
        gradient.append(fun_g(x))
        xs.append(xx)
        count += 1
    # print("Newton steps:", count)
    plot_gradient(np.array(gradient), "Newton's Method - "+name)
    plot_distance(np.array(xs), "Newton's Method - "+name)
    # plot_convergence(np.array(xs), fun, "Newton's Method")
    return x
    # return x, fun(x), count


def test_1(x):
    print("Ellipsoid test")
    res1 = steepest_descent(np.array(x), func.ellipsoid, func.ellipsoid_d1, name="f1")
    print(res1)
    print(func.ellipsoid(res1))

    res2 = newton(np.array(x), func.ellipsoid, func.ellipsoid_d1, func.ellipsoid_d2, name="f1")
    print(res2)
    print(func.ellipsoid(res2))


def test_2(x):
    print("Rosenbrock test")
    res1 = steepest_descent(np.array(x), func.Rosenbrock, func.Grad_Ros, name="f2")
    print(res1)
    print(func.Rosenbrock(res1))

    res2 = newton(np.array(x), func.Rosenbrock, func.Grad_Ros, func.Hessian_Ros, name="f2")
    print(res2)
    print(func.Rosenbrock(res2))


def test_3(x):
    print("Log-Ellipsoid test")
    res1 = steepest_descent(np.array(x), func.log_ellipsoid, func.log_ellipsoid_d1, name="f3")
    print(res1)
    print(func.log_ellipsoid(res1))

    res2 = newton(np.array(x), func.log_ellipsoid, func.log_ellipsoid_d1, func.log_ellipsoid_d2, name="f3")
    print(res2)
    print(func.log_ellipsoid(res2))


def test_4(x):
    print("Attractive-Sector test - F4")
    res1 = steepest_descent(np.array(x), func.f_4, func.f_4grad, name="f4")
    print(res1)
    print(func.f_4(res1))

    res2 = newton(np.array(x), func.f_4, func.f_4grad, func.f_4hessian, name="f4")
    print(res2)
    print(func.f_4(res2))


def test_5(x):
    print("Attractive-Sector test - F5")
    res1 = steepest_descent(np.array(x), func.f_5, func.f_5grad, name="f5")
    print(res1)
    print(func.f_5(res1))

    res2 = newton(np.array(x), func.f_5, func.f_5grad, func.f_5hessian, name="f5")
    print(res2)
    print(func.f_5(res2))


x0 = [2.0, 5.0, 10.0, 4.0, 8.0]
x1 = [2.0, 4.0]
# test_1(x0)
# plt.show()
# test_2(x1)
# plt.show()
test_3(x0)
plt.show()
# test_4(x0)
# plt.show()
# test_5(x0)
# plt.show()


def testing(fun, fun_g, fun_h, length=2, name=""):
    time_s = 0
    time_n = 0
    iter_s = 0
    iter_n = 0
    succ_s = 0
    succ_n = 0
    val_s = 0
    val_n = 0
    for i in range(100):
        x = np.random.rand(length) * 5

        tmp1 = time.process_time()
        x_, val, c = steepest_descent(x, fun, fun_g)
        tmp2 = time.process_time()
        time_s += tmp2-tmp1
        iter_s += c
        val_s += val
        if np.linalg.norm(fun_g(x_)) <= 1e-6:
            succ_s += 1

        tmp1 = time.process_time()
        x_, val, c = newton(x, fun, fun_g, fun_h)
        tmp2 = time.process_time()
        time_n += tmp2 - tmp1
        iter_n += c
        val_n += val
        if np.linalg.norm(fun_g(x_)) <= 1e-6:
            succ_n += 1

    print(name, "\nSteepest:", time_s/100, iter_s/100, succ_s/100, val_s/100,
          "\nNewton:", time_n/100, iter_n/100, succ_n/100, val_n/100, '\n')


# testing(func.ellipsoid, func.ellipsoid_d1, func.ellipsoid_d2, name="f1")
# testing(func.Rosenbrock, func.Grad_Ros, func.Hessian_Ros, name="f2")
# testing(func.log_ellipsoid, func.log_ellipsoid_d1, func.log_ellipsoid_d2, name="f3")
# testing(func.f_4, func.f_4grad, func.f_4hessian, name="f4")
# testing(func.f_5, func.f_5grad, func.f_5hessian, name="f5")

