import numpy as np
import handin1 as func
import matplotlib.pyplot as plt


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def backtracking(alpha, rho, c, p, x, fun, fun_g):
    while fun(x+alpha*p) > fun(x) + c * alpha * fun_g(x).T @ p:
        alpha = rho * alpha
    return alpha


def plot_convergence(xs, name):
    _x = xs[-1]
    it = np.arange(xs.shape[0])
    plt.plot(it, np.linalg.norm(xs-_x, axis=1), 'b')
    plt.yscale("log")
    plt.title(name)
    plt.xlabel("iteration")
    plt.ylabel("distance of x & x*")
    plt.show()


def plot_gradient(g, name):
    it = np.arange(g.shape[0])
    plt.plot(it, np.linalg.norm(g, axis=1), 'b')
    plt.yscale("log")
    plt.title(name)
    plt.xlabel("iteration")
    plt.ylabel("gradient")
    plt.show()


def steepest_descent(x, fun, fun_g, alpha=1.0, rho=0.6, c=1e-4, max_iter=1000):
    count = 0
    gradient = []
    xs = []
    while np.linalg.norm(fun_g(x)) > 1e-6 and count < max_iter:
        p = -fun_g(x)
        alpha = backtracking(alpha, rho, c, p, x, fun, fun_g)
        x += alpha * p
        xx = list(x)
        gradient.append(fun_g(x))
        xs.append(xx)
        count += 1
    print("steepest steps:", count)
    plot_gradient(np.array(gradient), "Steepest Descent")
    plot_convergence(np.array(xs), "Steepest Descent")
    return x


def newton(x, fun, fun_g, fun_h, beta=1e-3, alpha=1.0, rho=0.6, c=1e-4, max_iter=1000):
    count = 0
    gradient = []
    xs = []
    while np.linalg.norm(fun_g(x)) > 1e-6 and count < max_iter:
        hessian = fun_h(x)
        if not is_pos_def(hessian):
            diag = hessian.diagonal()
            min_diag = np.amin(diag)
            tol = 0
            if min_diag <= 0:
                tol = -min_diag + beta
            d = hessian.shape[0]
            hessian_modi = hessian + tol * np.identity(d)
            while (not is_pos_def(hessian_modi)) or (hessian_modi != hessian_modi.T).any():
                tol = max(2*tol, beta)
                hessian_modi = hessian + tol * np.identity(d)
        else:
            hessian_modi = hessian
        p = - np.linalg.inv(hessian_modi) @ fun_g(x)
        alpha = backtracking(alpha, rho, c, p, x, fun, fun_g)
        x += alpha * p
        xx = list(x)
        gradient.append(fun_g(x))
        xs.append(xx)
        count += 1
    print("Newton steps:", count)
    plot_gradient(np.array(gradient), "Newton's Method")
    plot_convergence(np.array(xs), "Newton's Method")
    return x


def test_1(x):
    print("Ellipsoid test")
    res1 = steepest_descent(np.array(x), func.ellipsoid, func.ellipsoid_d1)
    print(res1)
    print(func.ellipsoid(res1))

    res2 = newton(np.array(x), func.ellipsoid, func.ellipsoid_d1, func.ellipsoid_d2)
    print(res2)
    print(func.ellipsoid(res2))


def test_2(x):
    print("Rosenbrock test")
    res1 = steepest_descent(np.array(x), func.Rosenbrock, func.Grad_Ros)
    print(res1)
    print(func.ellipsoid(res1))

    res2 = newton(np.array(x), func.Rosenbrock, func.Grad_Ros, func.Hessian_Ros)
    print(res2)
    print(func.ellipsoid(res2))


def test_3(x):
    print("Log-Ellipsoid test")
    res1 = steepest_descent(np.array(x), func.log_ellipsoid, func.log_ellipsoid_d1)
    print(res1)
    print(func.ellipsoid(res1))

    res2 = newton(np.array(x), func.log_ellipsoid, func.log_ellipsoid_d1, func.log_ellipsoid_d2)
    print(res2)
    print(func.ellipsoid(res2))


def test_4(x):
    print("Attractive-Sector test - F4")
    res1 = steepest_descent(np.array(x), func.f_4, func.f_4grad)
    print(res1)
    print(func.ellipsoid(res1))

    res2 = newton(np.array(x), func.f_4, func.f_4grad, func.f_4hessian)
    print(res2)
    print(func.ellipsoid(res2))


def test_5(x):
    print("Attractive-Sector test - F5")
    res1 = steepest_descent(np.array(x), func.f_5, func.f_5grad)
    print(res1)
    print(func.ellipsoid(res1))

    res2 = newton(np.array(x), func.f_5, func.f_5grad, func.f_5hessian)
    print(res2)
    print(func.ellipsoid(res2))


x0 = [2.0, 4.0]
# test_1(x0)
test_2(x0)
# test_3(x0)

x1 = [2.0, 1.0]
# test_4(x1)
# test_5(x1)

