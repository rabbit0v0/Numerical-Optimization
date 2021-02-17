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
    r1_s, r2_s, r3_s = 0, 0, 0
    r1_f, r2_f, r3_f = 0, 0, 0
    # time, Simon said we can calculate average time of success/fail. Doing that.
    t1_s, t2_s, t3_s = 0, 0, 0
    t1_f, t2_f, t3_f = 0, 0, 0
    # frequency of success, minimize would fail
    s1, s2, s3 = 0, 0, 0
    # number of iteration
    i1, i2, i3 = 0, 0, 0
    i1_s, i2_s, i3_s = 0, 0, 0
    i1_f, i2_f, i3_f = 0, 0, 0
    points = 1000

    for i in range(points):
        # length = np.random.randint(2, 10, 1)[0]
        length = 5
        x = np.random.rand(length)*5

        tmp1 = time.process_time()
        res1 = minimize(func, x, method="BFGS", tol=1e-6, options={"disp": False})
        tmp2 = time.process_time()
        i1 += res1.nit
        if res1.success:
            s1 += 1
            t1_s += tmp2-tmp1
            r1_s += res1.fun
            i1_s += res1.nit
        else:
            t1_f += tmp2-tmp1
            r1_f += res1.fun
            i1_f += res1.nit

        tmp1 = time.process_time()
        res2 = minimize(func, x, method="Nelder-Mead", tol=1e-6, options={"disp": False})
        tmp2 = time.process_time()
        i2 += res2.nit
        if res2.success:
            s2 += 1
            t2_s += tmp2 - tmp1
            r2_s += res2.fun
            i2_s += res2.nit
        else:
            t2_f += tmp2 - tmp1
            r2_f += res2.fun
            i2_f += res2.nit

        tmp1 = time.process_time()
        res3 = minimize(func, x, method="CG", tol=1e-6, options={"disp": False})
        tmp2 = time.process_time()
        i3 += res3.nit
        if res3.success:
            s3 += 1
            t3_s += tmp2 - tmp1
            r3_s += res3.fun
            i3_s += res3.nit
        else:
            t3_f += tmp2 - tmp1
            r3_f += res3.fun
            i3_f += res3.nit

    print("BFGS success", i1_s/s1, t1_s/s1, r1_s/s1, s1/points)
    print("BFGS fail", i1_f / (points-s1), t1_f / (points-s1), r1_f / (points-s1))

    print("Nelder-Mead success", i2_s/s2, t2_s/s2, r2_s/s2, s2/points)
    print("Nelder-Mead fail", i2_f / (points-s2), t2_f / (points-s2), r2_f / (points-s2))

    print("CG success", i3_s/s3, t3_s/s3, r3_s/s3, s3/points)
    print("CG fail", i3_f / (points - s3), t3_f / (points - s3), r3_f / (points - s3))


opti(ellipsoid)


