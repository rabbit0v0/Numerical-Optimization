import numpy as np
import handin1 as func


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def backtracking(alpha, rho, c, p, x, fun, fun_g):
    while fun(x+alpha*p) > fun(x) + c * alpha * fun_g(x).T @ p:
        alpha = rho * alpha
    return alpha


def steepest_descent(x, fun, fun_g, alpha=1.0, rho=0.6, c=1e-4, max_iter=5000):
    count = 0
    while np.linalg.norm(fun_g(x)) > 1e-6 and count < max_iter:
        p = -fun_g(x)
        alpha = backtracking(alpha, rho, c, p, x, fun, fun_g)
        x += alpha * p
        count += 1
    print("steepest steps:", count)
    return x


def newton(x, fun, fun_g, fun_h, beta=1e-3, alpha=1.0, rho=0.6, c=1e-4, max_iter=5000):
    count = 0
    while np.linalg.norm(fun_g(x)) > 1e-6 and count < max_iter:
        hessian = fun_h(x)
        if not is_pos_def(hessian):
            diag = hessian.diagonal()
            min_diag = np.amin(diag)
            tol = 0
            if min_diag <= 0:
                tol = -min_diag + beta
            d = hessian.shape()[0]
            hessian_modi = hessian + tol * np.identity(d)
            while (not is_pos_def(hessian_modi)) or (hessian_modi != hessian_modi.T).any():
                tol = max(10*tol, beta)
                hessian_modi = hessian + tol * np.identity(d)
        else:
            hessian_modi = hessian
        p = - np.linalg.inv(hessian_modi) @ fun_g(x)
        alpha = backtracking(alpha, rho, c, p, x, fun, fun_g)
        x += alpha * p
        count += 1
    print("Newton steps:", count)
    return x


def test_1():
    print("Ellipsoid test")
    x = np.array([2, 4])
    x = x.astype(float)
    res1 = steepest_descent(x, func.ellipsoid, func.ellipsoid_d1)
    print(res1)
    print(func.ellipsoid(res1))

    x = np.array([2, 4])
    x = x.astype(float)
    res2 = newton(x, func.ellipsoid, func.ellipsoid_d1, func.ellipsoid_d2)
    print(res2)
    print(func.ellipsoid(res2))


def test_2():
    print("Rosenbrock test")
    x = np.array([2, 4])
    x = x.astype(float)
    res1 = steepest_descent(x, func.Rosenbrock, func.Grad_Ros)
    print(res1)
    print(func.ellipsoid(res1))

    x = np.array([2, 4])
    x = x.astype(float)
    res2 = newton(x, func.Rosenbrock, func.Grad_Ros, func.Hessian_Ros)
    print(res2)
    print(func.ellipsoid(res2))


test_1()
test_2()

