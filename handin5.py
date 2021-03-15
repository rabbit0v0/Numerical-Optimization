import numpy as np
from numpy.core.fromnumeric import argmin
import handin1 as func
import matplotlib.pyplot as plt
import time


def is_pos_def(x):
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
    minIdx = np.argmin(lmds)
    #is_psd  = lmds[minIdx] >= -atol
    if Q[:, minIdx].T @ gx != 0:
        if lmds[minIdx] > atol:
            lmd = 0
        else:
            lmd = -lmds[minIdx] + atol
        while True:
            B_inv = Q @ np.diag(1 / (lmds + lmd + atol)) @ Q.T
            p = B_inv @ -gx
            p_n = np.linalg.norm(p)
            if p_n <= delta + atol:
                break
            p_n2 = p_n*p_n
            q_n2 = p.T @ B_inv @ p
            lmd = lmd + (p_n2/q_n2) * (p_n - delta) / delta
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
    sy = []
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
        sy.append(s.T @ y)

        ared = fun(x) - fun(x+s)
        pred = -(g.T @ s + 0.5 * s.T @ B @ s)

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
        elif np.abs((y - B @ s).T @ s) < r:  # r?
            pass
        elif np.abs(s.T @ (y - B @ s)) >= r * np.linalg.norm(s) * np.linalg.norm(y - B @ s):
            B = B + np.outer((y - B @ s), (y - B @ s)) / ((y - B @ s).T @ s)


        xs.append(x)
        deltas.append(delta)

    return xs, fun(x), count, deltas
    # return xs, fun(x), count, np.array(sy)


def fun(x):
    # not diagonal hessian
    return func.log_ellipsoid(x, 10, 0.1)


def fun_g(x):
    return func.log_ellipsoid_d1(x, 10, 0.1)


def plot2(x):
    xs, val, c, sy = trust_region_sr1(x, fun, fun_g, 1)
    plt.scatter(np.arange(len(sy)), sy)
    plt.xlabel("iteration")
    plt.ylabel("s.T @ y")
    plt.show()


# plot2([4, 2])
# exercise 2

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


def plot_convergence(xs, fun, name, color='b'):
    # _x = xs[-1]
    # xx = cal_convergence(xs)
    xx = cal_rate_convergence(xs, fun)
    it = np.arange(xx.shape[0]-1)
    plt.plot(it, xx[:-1], c=color)
    # plt.legend()
    plt.yscale("log")
    plt.title(name)
    plt.xlabel("iteration")
    plt.ylabel("convergence rate of x")
    plt.show()


def test_conv(x, fun, fun_g, label, c="c"):
    xs, val, count, dels = trust_region_sr1(x, fun, fun_g, 1)
    plot_convergence(xs, fun, label, c)


x0 = [3, 2]
test_conv(x0, func.ellipsoid, func.ellipsoid_d1, "f_1")
test_conv(x0, func.Rosenbrock, func.Grad_Ros, "f_2")
test_conv(x0, func.log_ellipsoid, func.log_ellipsoid_d1, "f_3")
test_conv(x0, func.f_4, func.f_4grad, "f_4")
test_conv(x0, func.f_5, func.f_5grad, "f_5")


def testing(fun, fun_g, length=2, name=""):
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

    print(name, "time:", time_s/100, "iter", iter_s/100, "succ", succ_s/100, "val", val_s/100, '\n')


# testing(func.ellipsoid, func.ellipsoid_d1, name="f1")
# testing(func.Rosenbrock, func.Grad_Ros, name="f2")
# testing(func.log_ellipsoid, func.log_ellipsoid_d1, name="f3")
# testing(func.f_4, func.f_4grad, name="f4")
# testing(func.f_5, func.f_5grad, name="f5")


def test_sensi(fun, fun_g):
    x = [2.0, 4.0]
    iters = []
    deltas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    for delta in deltas:
        xs1, val, c, dels = trust_region_sr1(x, fun, fun_g, delta)
        iters.append(c)
    plt.plot(deltas, iters, c='c')
    plt.xscale('log')
    plt.xlabel('delta_0')
    plt.ylabel('iterations')
    plt.title("sensitive test of delta_0")
    plt.show()


# test_sensi(func.Rosenbrock, func.Grad_Ros)


def init_x(step,start,end,dimension): #set the start&end coord, step size and dimension number pf input set
    xi=np.arange(start, end, step); x=xi #initialization of input xi for each dimension
    for i in range(dimension-1):
        x=np.vstack((np.around(x,decimals=9),np.around(xi,decimals=9))) #make x to d dimensions, from xi
    return x


def plot2d(x0,f,f_input,fcount,f_y,name):
    x=init_x(0.02,-10,10,2)
    X1,X2 = np.meshgrid(x[0], x[1])  # generate all the data point
    dtsize=X1.shape[0]  # data point number
    Y=np.zeros((dtsize,dtsize))  # initialize output results to 2D
    for i in range(dtsize):
        for j in range(dtsize):
            X=np.vstack((np.around(X1[i,j],decimals=9),np.around(X2[i,j],decimals=9))) #choose every combination of 2D inputs
            Y[i,j]=f(X)  # store the results
    fx1=np.zeros(fcount+1)
    fx2=np.zeros(fcount+1)
    for i in range(fcount+1):
        if(i<=0):
            fx1[i]=x0[0]
            fx2[i]=x0[1]
        else:
            fx1[i]=f_input[i-1][0]
            fx2[i]=f_input[i-1][1]
    #plot in 2D with color
    fig, ax = plt.subplots()
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    #lv=[0,1,3,10,30,100,500,2000,4000,7000,11000,14000,17000,25000]
    #Cset = plt.contourf(X1, X2, Y, levels=lv,norm=colors.PowerNorm(gamma=0.25),cmap='coolwarm_r')
    # lv=[0,1,3,6,10,15,25,50,100,150,200,250,300,350,410,480,560,650,750]
    # Cset = plt.contourf(X1, X2, Y, levels=lv,norm=colors.PowerNorm(gamma=0.5),cmap='coolwarm_r')
    Cset = plt.contourf(X1, X2, Y, levels=20,cmap='coolwarm_r')
    plt.plot(fx1, fx2,c="k")
    plt.colorbar(Cset)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(name+'_2d.pdf')


def test2D_1(x):
    f1_input, f1_y, f1_count, f1_dels = trust_region_sr1(x, func.Rosenbrock, func.Grad_Ros, 1)
    # plot2d(x, func.Rosenbrock, f1_input, f1_count, f1_y, name="optimizing f2 with SR1")
    print(f1_count)


# test2D_1([3.0, 2.0])


def fun1(x):
    # same eigenvalues
    return func.ellipsoid(x, 1)


def fun1_g(x):
    return func.ellipsoid_d1(x, 1)


def fun2(x):
    # vary eigenvalues
    return func.ellipsoid(x, 100)


def fun2_g(x):
    return func.ellipsoid_d1(x, 100)


def fun3(x):
    # diagonal hessian
    return func.ellipsoid(x, 10)


def fun3_g(x):
    return func.ellipsoid_d1(x, 10)


def fun5(x):
    # diagonal hessian
    return func.ellipsoid(x, 1000)


def fun5_g(x):
    return func.ellipsoid_d1(x, 1000)


def f0(x):
    d = len(x)
    res = 0
    for i in range(d):
        res += x[i]**2
    return res


def f0_g(x):
    d = len(x)
    grad = []
    for i in range(d):
        grad.append(2*x[i])
    return np.array(grad)


def f1(x):
    d = len(x)
    res = 0
    for i in range(d):
        res += (i+1) * x[i]**2
    return res


def f1_g(x):
    d = len(x)
    grad = []
    for i in range(d):
        grad.append(2*(i+1)*x[i])
    return np.array(grad)


def f2(x):
    d = len(x)
    res = 0
    for i in range(d):
        res += 10**i * x[i]**2
    return res


def f2_g(x):
    d = len(x)
    grad =[]
    for i in range(d):
        grad.append(2*x[i]*10**i)
    return np.array(grad)


# print("alpha=1")
# testing(fun1, fun1_g, 5)
# print("alpha=10")
# testing(fun3, fun3_g, 5)
# print("alpha=100")
# testing(fun2, fun2_g, 5)
# print("alpha=1000")
# testing(fun5, fun5_g, 5)
# print("f3, alpha=10")
# testing(func.log_ellipsoid, func.log_ellipsoid_d1, 3)

# print("d=3")
# testing(fun3, fun3_g, 3)
# print("d=5")
# testing(fun3, fun3_g, 5)
# print("d=7")
# testing(fun3, fun3_g, 7)
# print("d=10")
# testing(fun3, fun3_g, 10)


