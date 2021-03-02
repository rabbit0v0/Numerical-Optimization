import numpy as np


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def trust_region_subproblem(gradient, hessian, lambda_0, delta):
    d = hessian.shape[0]
    lambda_1 = min(np.linalg.eigvals(hessian))
    lam = lambda_0
    while lam >= -lambda_1:
        B_ = hessian + lam * np.identity(d)
        try:
            R = np.linalg.cholesky(B_)
            p = np.linalg.inv(B_) @ (-gradient)
            q = np.linalg.inv(R.T) @ p
            lam_new = lam + (np.linalg.norm(p)/np.linalg.norm(q))**2 * ((np.linalg.norm(p)-delta)/delta)
            if lam_new == lam:
                break
            else:
                lam = lam_new
        except:
            print("something wrong")
    return lam


def cauchy_point(gradient, hessian, delta):
    temp = gradient.T @ hessian @ gradient
    if temp <= 0:
        tau = 1
    else:
        tau = min(1, np.linalg.norm(gradient)**3 / (delta * temp))
    return -tau * delta / np.linalg.norm(gradient) * gradient


def trust_region(_x, fun, fun_g, fun_h, _delta, delta_max, eta=0.5, max_iter=500):
    x = _x
    delta = _delta
    count = 0
    while count < max_iter:
        g = fun_g(x)
        B = fun_h(x)
        if not is_pos_def(B):
            d = B.shape[0]
            lam = trust_region_subproblem(g, B, -min(np.linalg.eigvals(B)), delta)  # set the first lambda to -lambda_1
            B = B + lam * np.identity(d)
        else:
            lam = 0

        if np.linalg.norm(np.linalg.inv(B) @ g) <= delta:
            p = np.linalg.norm(np.linalg.inv(B) @ g)
        else:
            p = cauchy_point(g, B, delta)  # needs improvement in 4.3

        rho = (fun(x) - fun(x+p)) / (- g.T @ p - 0.5 * p.T @ B @ p)  # should we use (4.1)? t?

        if rho < 1/4:
            delta = 1/4 * delta
        else:
            if rho > 3/4 and np.linalg.norm(p) == delta:
                delta = min(2*delta, delta_max)

        if rho > eta:
            x = x + p
        else:
            break  # x_k = x_k+1


