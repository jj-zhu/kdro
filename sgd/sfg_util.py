# util for SFG DRO
import torch
from sklearn.kernel_approximation import RBFSampler
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sgd.relaxed import Flatten


def buildModel(task=None, method=None, n_hidden = 100, device=None, loss_reduction='mean'):
    if task=='binary':
        n_output = 1
        loss_erm = nn.BCEWithLogitsLoss(reduction=loss_reduction)
    elif task=='multi':
        n_output = 10 # one hot
        loss_erm = nn.CrossEntropyLoss(reduction=loss_reduction)
    else:
        raise NotImplementedError

    if method=='mlp':
        modelDecision = nn.Sequential(Flatten(), nn.Linear(784, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output)).to(device)
    elif method=='linear':
        modelDecision = nn.Sequential(Flatten(), nn.Linear(784, n_output)).to(device) # the model that contains the decision var
    elif method=='cnn':
        modelDecision = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                                      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                                      Flatten(),
                                      nn.Linear(7 * 7 * 64, 100), nn.ReLU(),
                                      nn.Linear(100, n_output)).to(device)
    else:
        raise NotImplementedError



    return modelDecision, loss_erm

def computeErr(yp, y, task=None, shuffle_test=False):
    if task=='binary':
        err = ((yp > 0) * (y == 0) + (yp < 0) * (y == 1)).sum().item()/y.shape[0]
    elif task == 'multi':
        err = (yp.max(dim=1)[1] != y).sum().item()/y.shape[0]
    else:
        raise NotImplementedError
    return err

def mnistData(task='binary', batch_size=100, shuffle_test=False):
    from torchvision import datasets, transforms
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    if task=='binary':
        train_idx = mnist_train.train_labels <= 1
        mnist_train.data = mnist_train.train_data[train_idx]
        mnist_train.targets = mnist_train.train_labels[train_idx]

        test_idx = mnist_test.test_labels <= 1
        mnist_test.data = mnist_test.test_data[test_idx]
        mnist_test.targets = mnist_test.test_labels[test_idx]
    elif task=='multi':
        pass
    else:
        raise NotImplementedError
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test)

    return train_loader, test_loader

# for torch autograd
def apply_gd(x, step_gamma=0.1):
    xnew = x - step_gamma * x.grad.detach()
    x.grad.data.zero_()
    return xnew

# a general loss function
def loss_general_se(output, target):
    return torch.sum((output - target) ** 2, dim=1).reshape(-1, 1)


def sample_zeta(X, y, zeta_sample_method=None, loss_erm=None, modelDecision=None, n_sample_zeta=1, epsilon_attack=0.0, learn_task=None, F=None):
    Z = None
    yperturb = None

    for _ in range(n_sample_zeta):
        delta = zeta_sample_method(loss_erm, modelDecision, X, y, randomize=True, epsilon=epsilon_attack,
                                   alpha=0.01, num_iter=20, task=learn_task, rkhsF=F)  # used with the new pgd
        # delta = attack(loss_erm, modelDecision, X, y, randomize=True, epsilon=1.0, alpha=0.01, num_iter=20, task=learn_task, rkhsF=F) # used with the new pgd

        Zperturb = X + delta
        if Z is not None:  # also penalize the original loss at unperturbed data
            Z = torch.cat((Zperturb, Z))  # cat new perturb Z with the old
            yperturb = torch.cat((yperturb, y))
        else:
            Z = Zperturb
            yperturb = y

    return Z, yperturb

def sfg_train_step(X=None, y=None, just_rand=None, loss_erm=None, modelDecision=None, n_sample_zeta=None, epsilon_attack=None,
                   learn_task=None, F=None, epsilon=None, stat_plot=None, i_decay_csa=None, is_step_const=None,
                   model_class=None, modelSWA=None):
    Z, y = sample_zeta(X, y, zeta_sample_method=just_rand, loss_erm=loss_erm,
                       modelDecision=modelDecision, n_sample_zeta=n_sample_zeta,
                       epsilon_attack=epsilon_attack, learn_task=learn_task, F=F)
    yp = evalutateErmModel(modelDecision, Z,
                           task=learn_task)  # use original model in ERM to run thru data X
    # kdro surrogate loss
    f_emp = torch.mean(F(X, fit=True))  # emp loss under emp loss
    # '''KDRO obj'''
    obj = f_emp + epsilon * F.norm()  # obj of the optimization problem
    #  original loss l eval on z
    loss_emp = loss_erm(yp, y).reshape(-1, 1)
    assert loss_emp.shape == F(Z).shape
    cons_sip = loss_emp - F(Z)  # constraint function of SIP. G(th, zeta)
    partial_moment = 1
    if partial_moment:  # use functional constr: E h(l-f) = 0
        max_cons_g = torch.mean(torch.nn.functional.relu(cons_sip))
    else:
        max_cons_g, id_max = torch.max(cons_sip, 0)  # max of violation across all samples zeta
    # bookeeping
    stat_plot["max_cons_violation"].append(max_cons_g.data.detach())
    stat_plot["obj"].append(obj.data.detach())

    '''CSA - SGD step size'''
    # threshold
    threshold_csa = 0.1 / np.sqrt(i_decay_csa + 1)  # decay threhold
    # threshold_csa = 0.01 # constant step
    # step size
    if is_step_const:
        step_csa = 0.01
    else:
        step_csa = 0.1 / np.sqrt(i_decay_csa + 1)  # decay threhold
    i_decay_csa += 1
    # %% update dec var: th, f0, f
    # zero gradient before backward
    try:
        for w in modelDecision.parameters(): w.grad.data.zero_()
        for w in F.model.parameters(): w.grad.data.zero_()
    except:
        pass
    if max_cons_g <= threshold_csa:  # if constr satisfied
        obj.backward()
        # there is no th update since obj doens't have th in it

        # %% if cons. satisfied, polyak averaging to keep track of average
        if model_class == 'mlp':
            modelSWA.update_parameters(modelDecision)  # model with averaged weights
            # pass
        else:
            raise NotImplementedError

    else:  # cons violation
        if partial_moment:
            max_cons_g.backward()  # diff this:  E h(l-f)
        else:
            cons_sip[id_max].backward()

        # update model var
        if model_class == 'mlp':
            for weight in modelDecision.parameters():
                weight.data = apply_gd(weight, step_csa)
        else:
            raise NotImplementedError
    # update f0, weights of f
    for weight in F.model.parameters():
        weight.data = apply_gd(weight, step_csa)
    return i_decay_csa, stat_plot

def plot_images_mnist(X, y, yp, M, N, task='binary'):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M * 1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1 - X[i * N + j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("{}".format(get_digit(yp[i * N + j], task=task)))
            plt.setp(title, color=('b' if compare_digit(yp[i * N + j], y[i * N + j], task=task) else 'r'))

            # turn off ticks
            for label in ax[i][j].get_xticklabels():
                label.set_visible(False)
            for label in ax[i][j].get_yticklabels():
                label.set_visible(False)

            #             put border on wrong ones
            if compare_digit(yp[i * N + j], y[i * N + j], task=task):
                c_frame = 'b'
            else:
                c_frame = 'r'
                # ax[i][j].set_axis_off()

            for axis in ['top', 'bottom', 'left', 'right']:
                ax[i][j].spines[axis].set_linewidth(3.0)
                ax[i][j].spines[axis].set_color(c_frame)

    plt.tight_layout()
    return f

def compare_digit(yp, y, task='binary'):
    if task=='binary':
        res = 1-int((yp > 0) * (y == 0) + (yp < 0) * (y == 1))
    elif task == 'multi':
        res = yp.max(dim=0)[1] == y
    else:
        raise NotImplementedError
    return res

def get_digit(yp, task='binary'):
    if task=='binary':
        res = int(yp > 0)
    elif task == 'multi':
        res = yp.max(dim=0)[1]
    else:
        raise NotImplementedError
    return res


def run_test(test_loader, modelERM, loss_erm, loaded, attack_function=None, attack_range=None, task=None, n_test=10, device=None):
    D = {'err': [], 'attack': [], 'sig': []}
    for attack_strength in attack_range:
        i_test = 0

        err_this_attack = []
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            if task == 'binary':
                y = y.float()
            i_test += 1
            if i_test > n_test:
                break

            delta = attack_function(loss_erm, modelERM, X, y, attack_strength, task=task)

            # evaluate attack
            #         plt.figure()
            yp = evalutateErmModel(loaded['model'], X + delta, task=task)
            err_kdro = computeErr(yp, y, task=task)

            err_this_attack.append(deepcopy(err_kdro))

        # for each attack, compute average
        D['err'].append(np.mean(err_this_attack))
        D['sig'].append(np.std(err_this_attack))
        D['attack'].append(attack_strength)

    D['n_run'] = len(err_this_attack)  # total number of runs
    D['eps_dro'] = loaded['epsilon']
    D['is_erm'] = loaded['is_erm']
    try:
        D['is_pgd'] = loaded['is_pgd']
    except:
        pass

    D['task'] = task

    return D


class RKHSfunction():
    # easier to use with random features
    def __init__(self, kernel_gamma, seed=1, n_feat = 96):
        self.kernel_gamma = kernel_gamma
        self.n_feat = n_feat
        self.model = nn.Sequential(Flatten(), nn.Linear(n_feat, 1, bias=True)) # random feature. the param of this models are the weights, i.e., decision var
        self.seed = seed
        self.rbf_feature = RBFSampler(gamma=kernel_gamma, n_components=n_feat, random_state=seed) # only support Gaussian RKHS for now
        # self.rbf_feature.fit(X_example.view(X_example.shape[0], -1))

    def eval(self, X, fit=False):
        x_reshaped = (X.view(X.shape[0], -1))

        if fit:
            self.rbf_feature.fit(x_reshaped) # only transform during evaluation

        if not x_reshaped.requires_grad:
            # x_feat = self.rbf_feature.fit_transform(x_reshaped)
            x_feat = self.rbf_feature.transform(x_reshaped) # only transform during evaluation
            rkhsF = self.model(torch.from_numpy(x_feat).float())
        else:
            # raise NotImplementedError
            # print('need a pth impl of fit transform')
            # internally: self.fit(X, **fit_params).transform(X)
            x_detach = x_reshaped.detach()
            x_fitted = self.rbf_feature.fit(x_detach, y=None)
            # x_fitted.transform(x_reshaped)
            x_feat = pth_transform(x_fitted, x_reshaped)
            # assert torch.max(x_feat.detach() - self.rbf_feature.fit_transform(x_detach)) == 0 # there's randomness of course
            rkhsF = self.model(x_feat)[:, 0]

        return rkhsF

    def norm(self):
        return computeRKHSNorm(self.model)

    def set_seed(self,seed):
        # set the seed of RF. such as in doubly SGD
        self.seed = seed
        self.rbf_feature = RBFSampler(gamma=self.kernel_gamma, n_components=self.n_feat, random_state=seed) # only support Gaussian RKHS for now

    def __call__(self, X, fit=False, random_state=False):
        if random_state is True:
            self.set_seed(seed=np.random) # do a random seed reset for doubly stochastic
        # else:
        #     self.set_seed(seed=1)
        return self.eval(X, fit)


def evalutateErmModel(modelDecision, X, task=None):
    if task=='binary':
        yp = modelDecision(X.view(X.shape[0], -1))[:, 0]
    elif task == 'multi':
        yp = modelDecision(X)
    else:
        raise NotImplementedError
    return yp


def computeRKHSNorm(modelRF, no_bias = True):
    if no_bias: # do no penalize the constant in the RKHS norm (random feature, f:= f0 + w*phi, only regularize w*w)
        hnorm = torch.norm([w for w in modelRF.parameters()][
                              0])  # rkhs norm is just 2norm of weights of random features; index[0] because we don't need to consider reg of the constant f0
    else:
        hnorm = torch.norm([w for w in modelRF.parameters()][0]) +  torch.norm([w for w in modelRF.parameters()][0])

    return hnorm


def pgd_linf(loss_attack, model, X, y, epsilon=1.0, alpha=0.01, num_iter=20, randomize=False, task=None):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    for t in range(num_iter):
        loss = loss_attack(evalutateErmModel( model, X + delta, task=task), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    # for w in model.parameters(): w.grad.data.zero_()
        # for w in rkhsF.model.parameters(): w.grad.data.zero_()
    return delta.detach()


def just_rand(loss_attack, model, X, y, epsilon=1.0, alpha=0.01, num_iter=20, randomize=False, task=None, **kwargs):
    """ just use random noises to sample new samples"""
    delta = torch.rand_like(X, requires_grad=False)
    delta.data = 2* (delta.data - 0.5) * epsilon

    return delta.detach()