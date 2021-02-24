import numpy as np


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def fun1(Q, b, x):
    return 1/2 * x.T @ Q @ x - b.T @ x


def fun1_g(Q, b, x):
    return Q @ x - b


def backtracking(alpha, rho, c, p, Q, b, x, fun, fun_g):
    while fun(Q, b, (x+alpha*p)) > fun(Q, b, x) + c * alpha * fun_g(Q, b, x).T @ p:
        alpha = rho * alpha
    return alpha


def steepest_descent(Q, b, x):
    if (not is_pos_def(Q)) or (Q != Q.T).any():
        raise ValueError('Matrix A needs to be symmetric positive definite.')

    count = 0
    while np.linalg.norm(fun1_g(Q, b, x)) > 1e-6:
        p = -fun1_g(Q, b, x)
        alpha = backtracking(1.0, 0.6, 1e-4, p, Q, b, x, fun1, fun1_g)
        x += alpha * p
        count += 1
    print("steps:", count)
    return x


# def newton(Q, b, x):
#


def test_sd():
    Q = np.array([[3, 2], [2, 3]])
    b = np.array([-2, 7])
    x = np.array([0, 0])
    x = x.astype(float)
    print(steepest_descent(Q, b, x))


test_sd()


