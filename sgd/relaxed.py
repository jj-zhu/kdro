# %% imports
import numpy as np
import torch
from scipy import sparse
from torch import nn

def safe_sparse_dot(a, b, dense_output=False):
    '''from sk learn'''
    """Dot product that handle the sparse matrix case correctly
    """
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def pth_fit(self, X, y=None):
    random_state = self.random_state
    n_features = X.shape[1]

    self.random_weights_ = (np.sqrt(2 * self.gamma) * random_state.normal(
        size=(n_features, self.n_components)))

    self.random_offset_ = random_state.uniform(0, 2 * np.pi,
                                               size=self.n_components)
    return self


def pth_transform(self, X):
    '''pth implementation for random feature transform'''
    projection = torch.matmul(X, torch.from_numpy(self.random_weights_).float())  # pth version

    projection += torch.from_numpy(self.random_offset_)
    # np.cos(projection, projection)
    projection = torch.cos(projection)
    projection *= np.sqrt(2.) / np.sqrt(self.n_components)
    return projection


def lossKDRO(ypredict, y, rkhs_val_x, rkhs_val_z, loss_l, sqr=False, is_max=False):
    # the loss part without the RKHS norm
    if is_max:  # if max, instead of taking the mean, take max in partial moment. the expectation becomes sup
        partial_moment = torch.max(
            threshold_moment(ypredict, y, rkhs_val_z, loss_l, sqr=sqr))  # this is for augmented data
    else:
        partial_moment = torch.mean(
            threshold_moment(ypredict, y, rkhs_val_z, loss_l, sqr=sqr))  # this is for augmented data

    empirical = torch.mean(rkhs_val_x)

    # check dim
    assert list(partial_moment.shape) == []
    assert list(empirical.shape) == []

    return empirical, partial_moment  # + epsilon * torch.norm(w)


def threshold_moment(ypredict, y, rkhs_val, loss_l, sqr=False):
    thr = (torch.nn.functional.relu(loss_l(ypredict, y) - rkhs_val))
    if sqr:
        thr = thr ** 2
    return thr


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
