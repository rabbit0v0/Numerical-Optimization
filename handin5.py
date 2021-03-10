import numpy as np
from numpy.core.fromnumeric import argmin
import handin1 as func
import matplotlib.pyplot as plt
import time


def is_pos_def(x):
    # try:
    #     np.linalg.inv(x)
    # except:
    #     return False
    # return True

    return np.all(np.linalg.eigvals(x) > 0)


def trust_region_subproblem(gradient, hessian, lambda_0, delta, max_iter=100):
    d = hessian.shape[0]
    lambda_j, vector = np.linalg.eig(hessian)
    lambda_1 = min(lambda_j)
    min_index = argmin(lambda_j)
    lam = lambda_0
    p = np.linalg.inv(hessian) @ (-gradient)
    count = 0
    while lam >= -lambda_1 and count < max_iter:
        count += 1
        B_ = hessian + lam * np.identity(d)
        # try:
        if vector[:, min_index].T @ gradient == 0:
            p = hard_case(gradient, hessian, delta, lambda_j, vector, min_index)
            return p
        else:
            p = np.linalg.inv(B_) @ (-gradient)

        R = np.linalg.cholesky(B_)
        q = np.linalg.inv(R.T) @ p

        lam_new = lam + (np.linalg.norm(p) / np.linalg.norm(q)) ** 2 * (
            (np.linalg.norm(p) - delta) / delta
        )
        if lam_new == lam or (count >= max_iter and np.linalg.norm(p) <= delta) or (lam_new < -lambda_1):
            # return p
            break
        else:
            lam = lam_new

    # print(lambda_0, count)
    return p


def sub_solve_2(g, B, x0, delta0):
    atol = 1e-8
    delta, x0 = delta0, np.array(x0)
    x, gx, hx = x0, g(x0), B
    gx_norm = np.linalg.norm(gx)
    lmds, Q = np.linalg.eig(hx)
    minIdx  = np.argmin(lmds)
    #is_psd  = lmds[minIdx] >= -atol
    if Q[:, minIdx].T @ gx != 0:
        if lmds[minIdx] > atol:
            lmd = 0
        else:
            lmd = -lmds[minIdx] + atol
        while True:
            B_inv = Q @ np.diag(1 / (lmds + lmd)) @ Q.T
            p     = B_inv @ -gx
            p_n   = np.linalg.norm(p)
            if p_n <= delta + atol:
                break
            p_n2  = p_n*p_n
            q_n2  = p.T @ B_inv @ p
            lmd   = lmd + (p_n2/q_n2) * (p_n - delta) / delta
    else:
        # print(f"Hard case! (||gx|| = {gx_norm})")
        p_n = delta
        if gx_norm == 0:
            p = delta*Q[:, minIdx]
        else:
            p = delta*(-gx/gx_norm)
            gBg = gx.T @ hx @ gx
            if gBg > 0:
                tau = (gx_norm*gx_norm*gx_norm) / (delta * gBg)
                if tau < 1:
                    p, p_n = p*tau, p_n*tau
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


def trust_region_sr1(_x, fun, fun_g, _delta, tol=1e-6, eta=1e-4, r=1e-8, max_iter=1000):
    xs = [_x]
    deltas = [_delta]
    x = _x
    delta = _delta
    # B = _B
    B = np.identity(len(_x))
    count = 0
    while count < max_iter and np.linalg.norm(fun_g(x)) > tol:
        count += 1
        g = fun_g(x)
        # if is_pos_def(B):
        #     if np.linalg.norm(np.linalg.inv(B) @ g) <= delta:
        #         s = -np.linalg.inv(B) @ g
        #     else:
        #         s = trust_region_subproblem(g, B, 0, delta)
        # else:
        #     s = trust_region_subproblem(g, B, -min(np.linalg.eigvals(B))+1e-3, delta)
        s = sub_solve_2(fun_g, B, x, delta)
        y = fun_g(x+s) - fun_g(x)
        ared = fun(x) - fun(x+s)
        pred = -(g.T @ s + 1/2 * s.T @ B @ s)

        rho = ared / pred

        if rho > 0.75:
            if np.linalg.norm(s) > 0.8 * delta:
                delta = 2 * delta
        elif 0.1 <= rho <= 0.75:
            pass
        else:
            delta = 0.5 * delta

        if rho > eta:
            x = x + s
        else:
            # break  # x_k = x_k+1
            pass

        if np.array_equal(y, B @ s):
            pass
        elif (y - B @ s).T @ s < r:  # r?
            pass
        elif abs(s.T @ (y - B @ s)) >= r * np.linalg.norm(s) * np.linalg.norm(y - B @ s):
            B = B + np.outer((y - B @ s), (y - B @ s)) / ((y - B @ s).T @ s)


        xs.append(x)
        deltas.append(delta)

    return xs, fun(x), count, deltas



def testing(fun, fun_g, fun_h, length=2, name=""):
    time_s = 0
    iter_s = 0
    succ_s = 0
    val_s = 0
    for i in range(100):
        x = np.random.rand(length) * 5

        tmp1 = time.process_time()
        xs, val, c, dels = trust_region_sr1(x, fun, fun_g, 1)
        # xs, val, c, dels = trust_region_sr1(x, fun, fun_g, fun_h(x), 1)
        x_ = xs[-1]
        tmp2 = time.process_time()
        time_s += tmp2-tmp1
        iter_s += c
        val_s += val
        if np.linalg.norm(fun_g(x_)) <= 1e-6:
            succ_s += 1

    print(name, "\n", "time:", time_s/100, "iter", iter_s/100, "succ", succ_s/100, "val", val_s/100, '\n')


testing(func.ellipsoid, func.ellipsoid_d1, func.ellipsoid_d2, name="f1")
testing(func.Rosenbrock, func.Grad_Ros, func.Hessian_Ros, name="f2")
testing(func.log_ellipsoid, func.log_ellipsoid_d1, func.log_ellipsoid_d2, name="f3")
testing(func.f_4, func.f_4grad, func.f_4hessian, name="f4")
testing(func.f_5, func.f_5grad, func.f_5hessian, name="f5")

