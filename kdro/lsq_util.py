'''
utilities for the robust least squares experiment
'''

import cvxpy as cp
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import euclidean_distances

def matDecomp(K):
    # decompose matrix for computing RKHS norms
    try:
        L = np.linalg.cholesky(K)
    except:
        # print('warning, K is singular')
        d, v = np.linalg.eigh(K) #L == U*diag(d)*U'. the scipy function forces real eigs
        d[np.where(d < 0)] = 0 # get rid of small eigs
        L = v @ np.diag(np.sqrt(d))
    return L

def median_heuristic(X, Y):
    '''
    the famous kernel median heuristic
    '''
    distsqr = euclidean_distances(X, Y, squared=True)
    kernel_width = np.sqrt(0.5 * np.median(distsqr))

    '''in sklearn, kernel is done by K(x, y) = exp(-gamma ||x-y||^2)'''
    kernel_gamma = 1.0 / (2 * kernel_width ** 2)

    return kernel_width, kernel_gamma

class costFun():
    # cost function
    def __init__(self, method = 'boyd', model=None, mode='casadi'):
        # weights
        self.method = method
        self.model = model
        self.mode = mode

    def __call__(self, x, w):
        return self.eval(x, w)

    def eval(self, x, w):
        '''
        evaluate the cost function, with casadi operation
        input:
        x: primal decision var
        w: RV, randomness, in ml: it's the data
        '''
        if len(x.shape)==1:
            x=x.reshape(-1,1)
        elif len(x.shape)==2:
            pass
        else:
            raise NotImplementedError

        if self.method =='boyd':
            '''
            Boyd & Vandenberghe book. Figure 6.15.

            min_x || A(w) - b0 ||^2 where
            A(w) := A0 + w * B0 and w is a scalar.
            '''
            # boy data set
            A0, B0, b0 = self.model  # model of the optimization problem
            if self.mode== 'casadi':
                from casadi import sumsqr
                cost_val = sumsqr((A0 + w * B0) @ x - b0)
            elif self.mode== 'cvxpy':
                import cvxpy as cp
                cost_val = cp.sum_squares((A0 + w * B0) @ x - b0)
            elif self.mode=='numpy':
                cost_val = np.sum(((A0 + w * B0) @ x - b0)**2)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return cost_val

class rkhsFun():
    def __init__(self,kernelFun,gamma=None, casadi=False):
        '''
        kernel gamma param
        '''
        # self.data=data
        self.kernelFun=kernelFun
        self.kernel_gamma=gamma
        self.casadi=casadi

    def eval(self, x, a, data):
        '''
        compute the rkhs fucntion
            f(x) = sum a_i * k(data_i, x)
        x: the location to evaluate, can be a vector
        data: the expansion points, empirical data
        '''
        if len(x.shape) == 1:
            xloc = x.reshape(-1,1)
        elif len(x.shape) == 2:
            xloc = x
        else:
            raise NotImplementedError

        k = self.kernelFun(data, xloc, gamma=self.kernel_gamma)
        fval = a @ k
        return fval

    def __call__(self, x, a, data):
        return self.eval(x, a, data)

def dataGengenerate(data, dim_x, dim_w, n_pick, n_sample=10):
    '''
    construct data set for DRO
    '''
    A0, b0, B0 = data['A'][:n_pick, :dim_x], data['b'][:n_pick, :], data['B'][:n_pick,:dim_x]
    # generate empirical samples; randomly from [-0.5, 0.5]
    ws = 0.5 * np.random.uniform(-1, 1, size=[n_sample, dim_w])

    #     '''the model is min | (A+u*B) x - b |'''
    if len(ws.shape) < 2:
        ws = ws.reshape((-1, dim_w))

    return ws, np.asarray(A0), np.asarray(b0), np.asarray(B0)

def kDroPy(loss, data_emp, is_max = False,epsilon=0.1, dim_x=0, n_certify=0, sampling_method='bound', lb=-1, ub=1, is_f0_fixed=False, solver=None):
    '''
    # KDRO using CVXPY
    :param loss:
        an instance of costFun class.
    :param data_emp: empirical data samples
    :param is_max: take max over i instead of the average risk
    :param epsilon: eps for dro
    :param dim_x: dimension of decision var
    :param n_certify: num of points to certify the semi inf constr.
    :param lb: the upper bound for the domain
    :param ub: the lower bound
    :is_f0_fixed: whether to fix the const. f0 to a number. False by default
    '''
    n_sample, dim_w = data_emp.shape

    th = cp.Variable(shape=(dim_x, 1))
    constr = []

    # KDRO part
    a = cp.Variable(n_sample + n_certify)
    f0 = cp.Variable()

    if is_f0_fixed:# optionally, set f0 to a higher value to test if semi-inf constr. can be certified.
        constr.append(f0==0)

    if sampling_method=='bound': # sample within certain bound
        zetai = np.random.uniform(lb, ub, size=[n_certify,
                                                       dim_w])  # let the samples also live in the intervel I sampled uncertainty w
    elif sampling_method=='hull': # sample using convex hull of empirical data
        # this is equiv. to really do it in multi dimensions, need to sample coeff. from a simplex
        zetai = np.random.uniform(np.min(data_emp), np.max(data_emp), size=[n_certify,
                                                           dim_w])  # let the samples also live in the intervel I sampled uncertainty w
    else:
        raise NotImplementedError

    zetai = np.concatenate(
        [data_emp, zetai])  # in practice, we always include the empirical data in the sampled support
    kernel_width, kernel_gamma = median_heuristic(zetai, zetai) # use median heuristics for kernel width
    f_fun = rkhsFun(rbf_kernel, kernel_gamma) # this is the dual variable, rkhs function f in the paper
    fvals = f_fun(zetai, a.T, zetai) # evaluate the f at the value of zetas
    K = rbf_kernel(zetai, zetai, gamma=kernel_gamma)

    for i, w in enumerate(zetai):
        constr += [loss(th, w) <= f0 + fvals[i]]

    if is_max: # use scenario robustification
        emp = f0 + cp.max(f_fun(data_emp, a.T, zetai))
    else: # if not, take average. it's just sample average/ERM
        emp = f0 + cp.sum(f_fun(data_emp, a.T, zetai)) / n_sample
    rkhs_norm = cp.norm(a.T @ matDecomp(K))

    reg_term = epsilon * rkhs_norm
    obj = emp + reg_term
    opt = cp.Problem(cp.Minimize(obj), constr)

    if solver is None:
        opt.solve()
    else:
        opt.solve(solver=solver)
    return th.value, obj.value, a.value, f0.value,kernel_gamma, zetai

def kDroCvar(loss, data_emp, epsilon=0.1, dim_x=0, n_certify=0, lb=-1, ub=1, chance_level=0.05, method='sampling', solver=None):
    '''
    cvar variant of kdro
    this is a variant that uses conditional value at risk approximation of semi-inf constr.
    :param chance_level: the level of chance constr satisfied: p(contr violation) <= chance_level
    :param loss:
        an instance of costFun class.
    :param data_emp: empirical data samples
    :param is_max: take max over i instead of the average risk
    :param epsilon: eps for dro
    :param dim_x: dimension of decision var
    :param n_certify: num of points to certify the semi inf constr.
    :param lb: the upper bound for the domain
    :param ub: the lower bound
    :is_f0_fixed: whether to fix the const. f0 to a number. False by default
    '''
    n_sample, dim_w = data_emp.shape

    th = cp.Variable(shape=(dim_x, 1))

    # KDRO part
    a = cp.Variable(n_sample + n_certify)
    t = cp.Variable() # var used for CVaR computation

    if method =='sampling':
        location_certify = np.random.uniform(lb, ub, size=[n_certify,
                                                           dim_w])  # let the samples also live in the intervel I sampled uncertainty w
        location_certify = np.concatenate(
            [data_emp, location_certify])  # in practice, we do this to certify all empirical points, since strong duality
    elif method=='perturbation':
        '''perturbing empirical data'''
        print('please see the perturbation folder')
        raise NotImplementedError
    else:
        raise NotImplementedError

    kernel_width, kernel_gamma = median_heuristic(location_certify, location_certify)
    yFun = rkhsFun(rbf_kernel, kernel_gamma)
    fvals = yFun(location_certify, a.T, location_certify)
    K = rbf_kernel(location_certify, location_certify, gamma=kernel_gamma)

    cvar = 0
    for i, w in enumerate(location_certify):

        cvar += cp.maximum( 0, loss(th, w) - fvals[i] -t ) # compute the CVaR part of the loss

    cvar = t + cvar  / (chance_level * location_certify.shape[0])

    # compute empirical risk: take average. it's just sample average/ERM
    emp = cp.sum(yFun(data_emp, a.T, location_certify)) / n_sample
    rkhs_norm = cp.norm(a.T @ matDecomp(K))

    reg_term = epsilon * rkhs_norm
    obj = cvar + emp + reg_term
    opt = cp.Problem(cp.Minimize(obj))

    if solver is None:
        opt.solve()
    else:
        opt.solve(solver=solver)

    # the cvar value in my current formulation is equiv to original f0
    res = {'x': th.value, 'obj':obj.value, 'a':a.value,'cvar':cvar.value, 'location_certify':location_certify, 'kernel_gamma':kernel_gamma}
    return res

def robustLs(A, B, b, bound=1.0):
    '''
    cvx py implementation of worst-case RO using SDP
    :param A, B, b: lsq data
    :param bound: the magnitude of the uncertain var. default to unit ball
    :return:
        optimizer: x
        obj value
    '''
    m, n= A.shape
    nb = b.shape[1]

    #%% follow boyd (6.15) for var naming
    x = cp.Variable([n,1])
    t = cp.Variable()
    lmd= cp.Variable()

    P = bound * (B @ x)
    q = A @ x - b

    # use bmat to make a block mat
    I1 = np.eye(m)
    I2 = np.eye(nb)
    Z1 = np.zeros([nb, nb]) # zero blocks

    xmat = cp.bmat(
        [[I1, P, q],
         [P.T, lmd * I2, Z1],
         [q.T, Z1.T, t * I2]
         ]
    )

    cons = [xmat >>0]
    obj = cp.Minimize(t + lmd)
    opt = cp.Problem(obj, cons)

    opt.solve()
    return x.value, obj.value

def saa(data_emp, A, B, b, dim_x=0):
    '''saa solution'''
    lsq = costFun(method='boyd', model=[A, B, b], mode='cvxpy')

    th = cp.Variable(shape=(dim_x, 1))

    obj = 0
    for w in data_emp:
        obj += lsq(th, w)
    optlsq = cp.Problem(cp.Minimize(obj))
    optlsq.solve()
    return th.value

def kdroPlot(**kwargs):
    '''results plot function used the robust lsq experiment'''
    curves = []

    curve = kwargs['ax_main'].plot(kwargs['disturb_set'],
                         kwargs['mu'],
                         label=kwargs['label'],
                         color=kwargs['color'],
                         linestyle=kwargs['linestyle'])

    curves.append(curve)
    if True:
        errbar = kwargs['sig'] / np.sqrt(kwargs['n_run'])
        kwargs['ax_main'].fill_between(kwargs['disturb_set'],
                             np.subtract(kwargs['mu'], errbar),
                             np.add(kwargs['mu'], errbar),
                             alpha=0.2,
                             color=kwargs['color'])
