'''
Module containing the main contributions. Kernel Distributionally Robust
Optimization (KDRO).
'''
import numpy as np
import cvxpy as cp
from abc import ABC
import abc


class AbstractKdro(ABC):
    '''
    An abstract base class of KDRO algorithms.
    '''
    @abc.abstractmethod
    def robust_opt(self):
        raise NotImplementedError()

def matDecomp(K):
    # import scipy
    # decompose matrix
    try:
        L = np.linalg.cholesky(K)
    except:
        # print('warning, Gram matrix K is singular')
        d, v = np.linalg.eigh(K) #L == U*diag(d)*U'. the scipy function forces real eigs
        d[np.where(d < 0)] = 0 # get rid of small eigs
        L = v @ np.diag(np.sqrt(d))
    return L

class KdroJointEpsBall_Cvxpy(AbstractKdro):
    '''
    Robustify the joint distribution. For supervised learning (x,y).
    '''
    def __init__(self, dim_theta, loss_call, K, Xobs, Yobs, Xcert, Ycert):
        '''
        dim_theta: the dimension of theta (parameter to be optimized)

        loss_call: a callable (function) 
            (theta, xcert, ycert) |-> loss value

        K: a Gram matrix computed such that
            K_ij = k( (x_i, y_i), (x_j, y_j)) where (x_i, y_i) is an observation,
            in the set (Xobs, Yobs) (set of observations) and (Xcert, Ycert),
            in that order.

        Xobs: N x d numpy array containing N observations.
        Yobs: N x dy numpy array containing N observations.

        Xcert: n x d numpy array containing n input locations to certify
            the robustness . Can be None (empty) in which case the
            optimization is less robust. obs is part of the set of certifying
            points by default.

        '''
        assert dim_theta > 0 
        assert Xcert.shape[0] == Ycert.shape[0]
        assert K.shape[1] >= Xcert.shape[0]

        self.dim_theta = dim_theta
        self.loss_call = loss_call
        self.K = K
        self.Xobs = Xobs
        self.Yobs = Yobs
        self.Xcert = Xcert
        self.Ycert = Ycert

    def robust_opt(self, eps, verbose=False):
        '''
        eps: epsilon to control the size of the norm ball constraint.

        Return a dictionary of optimization results.
        '''
        K = self.K
        n_sample = self.Xobs.shape[0]

        # sample size for the set of certification points
        n_certify = self.Xcert.shape[0]

        # All variables to be optimized
        theta = cp.Variable(self.dim_theta)

        # f0 = a bias term as part of the RKHS function. A scalar
        f0 = cp.Variable()

        # Beta is the vector of coefficients of the dual RKHS function.
        beta = cp.Variable(K.shape[1])

        # function values at the kernel_points
        fvals = K @ beta

        # List of constraints for cvxpy
        constraints = []
        loss_call = self.loss_call
        # always certify the observations
        for i in range(n_sample):
            constraints += [loss_call(theta, self.Xobs[i], self.Yobs[i]) 
            <= f0 + fvals[i] ]

        # certify the certifying points
        for i in range(n_certify):
            # wi = self.cert_locs[i]
            xcert_i = self.Xcert[i]
            ycert_i = self.Ycert[i]
            constraints += [loss_call(theta, xcert_i, ycert_i) <= f0 +
            fvals[i+n_sample]]
        
        emp = f0 + cp.sum(fvals[:n_sample]) / n_sample
        # regularization term
        # rkhs_norm = cp.sqrt(cp.quad_form(beta, K + 1e+1*np.eye(K.shape[0])))
        rkhs_norm = cp.norm(beta.T @ matDecomp(K ))
        reg_term = eps * rkhs_norm

        # objective function
        obj = emp + reg_term
        opt = cp.Problem(cp.Minimize(obj), constraints)

        # opt.solve(verbose=verbose)

        # opt.solve(solver=cp.ECOS, verbose=verbose, abstol=1e-6, reltol=1e-5, feastol=1e-5)
        # opt.solve(solver=cp.ECOS_BB, verbose=verbose)
        # opt.solve(solver=cp.SCS, verbose=verbose, eps=1e-5)
        # opt.solve(solver=cp.CVXOPT, verbose=verbose, kktsolver='ROBUST_KKTSOLVER')
        opt.solve(solver=cp.MOSEK, verbose=verbose)
        # result
        R = {
            'theta': theta.value,
            'obj': obj.value,
            'beta': beta.value,
            'f0': f0.value,
            'rkhs_norm': rkhs_norm.value,
        }
        return R


