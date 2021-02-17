from scipy.optimize import minimize
import numpy as np
import time


# x is an array
def ellipsoid(x, alpha=1000):
    d = len(x)
    res = 0
    for i in range(d):
        res += alpha**(i/(d-1)) * x[i]**2
    return res


def log_ellipsoid(x, alpha=1000, epsilon=1e-16):
    elli = ellipsoid(x, alpha)
    return np.log(epsilon+elli)


def Rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


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


def opti(func):
    # function value
    r1, r2, r3 = 0, 0, 0
    # time, Simon said we can calculate average time of success/fail. Doing that.
    t1, t2, t3 = 0, 0, 0
    # frequency of success, minimize would fail
    s1, s2, s3 = 0, 0, 0
    # number of iteration
    i1, i2, i3 = 0, 0, 0
    points = 1000

    for i in range(points):
        # length = np.random.randint(2, 10, 1)[0]
        length = 5
        x = np.random.rand(length)*5

        tmp1 = time.process_time()
        res1 = minimize(func, x, method="BFGS", tol=1e-6, options={"disp": False, "maxiter": 500})
        tmp2 = time.process_time()
        t1 += tmp2-tmp1
        i1 += res1.nit
        if res1.success:
            s1 += 1
        r1 += res1.fun

        tmp1 = time.process_time()
        res2 = minimize(func, x, method="Nelder-Mead", tol=1e-6, options={"disp": False, "maxiter": 500})
        tmp2 = time.process_time()
        t2 += tmp2-tmp1
        i2 += res2.nit
        r2 += res2.fun
        if res2.success:
            s2 += 1

        tmp1 = time.process_time()
        res3 = minimize(func, x, method="CG", tol=1e-6, options={"disp": False, "maxiter": 500})
        tmp2 = time.process_time()
        t3 += tmp2 - tmp1
        i3 += res3.nit
        r3 += res3.fun
        if res3.success:
            s3 += 1

    print("BFGS", i1/points, t1/points, r1/points, s1/points)
    print("Nelder-Mead", i2 / points, t2 / points, r2 / points, s2 / points)
    print("CG", i3 / points, t3 / points, r3 / points, s3 / points)


opti(ellipsoid)

