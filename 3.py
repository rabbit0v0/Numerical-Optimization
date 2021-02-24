import numpy as np


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def fun1(x):
    Q = np.array([[3, 2], [2, 3]])
    b = np.array([-2, 7])
    if (not is_pos_def(Q)) or (Q != Q.T).any():
        raise ValueError('Matrix A needs to be symmetric positive definite.')
    return 1/2 * x.T @ Q @ x - b.T @ x


def fun1_g(x):
    Q = np.array([[3, 2], [2, 3]])
    b = np.array([-2, 7])
    if (not is_pos_def(Q)) or (Q != Q.T).any():
        raise ValueError('Matrix A needs to be symmetric positive definite.')
    return Q @ x - b


def backtracking(alpha, rho, c, p, x, fun, fun_g):
    while fun(x+alpha*p) > fun(x) + c * alpha * fun_g(x).T @ p:
        alpha = rho * alpha
    return alpha


def steepest_descent(x):
    count = 0
    while np.linalg.norm(fun1_g(x)) > 1e-6:
        p = -fun1_g(x)
        alpha = backtracking(1.0, 0.6, 1e-4, p, x, fun1, fun1_g)
        x += alpha * p
        count += 1
    print("steps:", count)
    return x


# def newton(Q, b, x):
#


def test_sd():
    x = np.array([0, 0])
    x = x.astype(float)
    print(steepest_descent(x))


test_sd()


