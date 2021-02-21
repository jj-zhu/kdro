
class KdroEpsBall_Cvxpy(AbstractKdro):
    '''
    An implementation of KDRO using cvxpy as the optimization package.
    The uncertainty set is chosen to be an epsilon norm ball in the RKHS
    defined by the kernel k.
    '''
    def __init__(self, dim_theta, loss_call, k, obs, cert_locs, eps):
        '''
        dim_theta: the dimension of theta (parameter to be optimized)

        loss_call: a callable (function) 
            (theta, cert_locs, i) |-> loss value, where i an index
            for indexing the appropriate element in cert_locs. 

        k: a instance of kernel. Kernel representing a kernel function.

        obs: N x d numpy array containing N observations.

        cert_locs: n x d numpy array containing n input locations to certify
            the robustness, in addition to obs. Can be None (empty) in which case the optimization is less robust. obs is part of 
            the set of certifying points by default.

        eps: epsilon to control the size of the norm ball constraint.
        '''
        assert dim_theta > 0 
        self.dim_theta = dim_theta
        self.loss_call = loss_call
        self.loss = loss
        self.k = k
        self.obs = obs
        if cert_locs is None:
            cert_locs = []
        self.cert_locs = cert_locs
        if cert_locs:
            if obs.shape[1] != cert_locs[1]:
                raise ValueError('dimension is match in obs and cert_locs. Found obs.shape[1] = {} and cert_locs.shape[1] = {}'.format(obs.shape[1], cert_locs.shape[1]))
        self.eps = eps

    def robust_opt(self):
        '''
        Return
        '''
        # d = input dimension
        n_sample, d = self.obs.shape

        # input points to define the dual RKHS function in Kdro.
        kernel_points = np.vstack((self.obs, self.cert_locs))

        # sample size for the set of certification points
        n_certify = kernel_points.shape[0]

        # All variables to be optimized
        theta = cp.Variable(self.dim_theta)

        # f0 = a bias term as part of the RKHS function. A scalar
        f0 = cp.Variable()

        # Beta is the vector of coefficients of the dual RKHS function.
        beta = cp.Variable(kernel_points.shape[0])

        # Gram matrix
        K = self.k.eval(kernel_points, kernel_points)
        # function values at the kernel_poitns
        fvals = K @ beta

        # List of constraints for cvxpy
        constraints = []
        loss_call = self.loss_call
        for i in range(n_certify):
            # wi = self.cert_locs[i]
            constraints += [loss_call(theta, self.cert_locs, i) <= f0 +
            fvals[i]]
        
        emp = f0 + cp.sum(fvals[:n_sample]) / n_sample
        # regularization term
        rkhs_norm = cp.sqrt(cp.quad_form(beta, K))
        reg_term = self.eps * rkhs_norm

        # objective function
        obj = emp + reg_term
        opt = cp.Problem(cp.Minimize(obj), constraints)
        opt.solve()

        # result
        R = {
            'theta': theta.value,
            'obj': obj.value,
            'beta': beta.value,
            'f0': f0.value,
        }
        return R


class KdroLossBinaryCrossEnt(object):
    '''
    Cross entropy loss for binary classification. Or just the negative
    likelihood. For Cvxpy.
    Linear basis function. So the logit is f(x, theta) = x*theta + t0
    where t0 is the bias term (scalar).

    Provide robustness only on the noise components.
    '''
    def __init__(self, X, Y):
        '''
        X: data matrix for the training set. n x d. Numpy array.
        Y: binary labels (0 or 1). nx1 Numpy array.
        '''
        self.X = X
        self.Y = Y

    def loss(self, theta, cert_locs, i):
        '''
        theta: parameter vector to be optimized in the logistic regression
            model
        '''
        raise NotImplementedError('Need to account for cert_loc')

        X = self.X
        Y = self.Y
        n = X.shape[0]

        # only class 0
        X0 = X[Y==0]

        F0 = KdroLossBinaryCrossEnt.cvx_model_class1(theta, X0)
        a1 = cp.sum(F0)/n

        F = KdroLossBinaryCrossEnt.cvx_model_class1(theta, X)
        Softmax = cp.log(1.0 + cp.exp(-F))
        a2 = cp.sum(Softmax)/n
        l = a1+a2
        return l

    def dim_theta(self):
        # +1 for the bias term. Last coordinate.
        return self.X.shape[1] + 1

    @staticmethod
    def cvx_model_class1(self, theta, X):
        '''
        Like np_mode_class1() but for Cvxpy.
        theta: one-dimensional Cvxpy array
        '''
        the = theta[:-1]
        t0 = the[-1]
        logit = X@the + t0
        sig = 1.0/(1.0 + cp.exp(-logit))
        # 1d array
        return sig

    @staticmethod
    def np_model_class1(self, theta, X):
        '''
        Evaluate the model for p(y=1|X). For each x in X, 
        p(y=1|x) = sigmoid( x*theta + t0 )

        theta: a numpy array of length X.shape[1] + 1
        X: numpy data matrix. n x d

        Return a 1d numpy array of length n
        '''
        the = theta.reshape(-1, 1)
        t0 = the[-1]
        the = the[:-1]
        logit = X.dot(the) + t0
        sig = 1.0/(1.0 + np.exp(-logit))
        return sig.reshape(-1)