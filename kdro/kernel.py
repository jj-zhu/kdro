'''
A module containing implementations of kernel functions
'''

import numpy as np
import abc
from abc import ABC

class Kernel(ABC):
    """Abstract class for kernels. Inputs to all methods are numpy arrays."""

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        pass

    def __call__(self, X, Y):
        return self.eval(X, Y)

class KGauss(Kernel):
    """
    The standard isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0. Was %s'%str(sigma2)
        self.sigma2 = sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X : n1 x d numpy array
        Y : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        #(n1, d1) = X.shape
        #(n2, d2) = Y.shape
        #assert d1==d2, 'Dimensions of the two inputs must be the same'
        sumx2 = np.reshape(np.sum(X**2, 1), (-1, 1))
        sumy2 = np.reshape(np.sum(Y**2, 1), (1, -1))
        D2 = sumx2 - 2.0*np.dot(X, Y.T) + sumy2
        K = np.exp(-D2/(2.0*self.sigma2))
        return K

    def __str__(self):
        return "KGauss(w2=%.3f)"%self.sigma2