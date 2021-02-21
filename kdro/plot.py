"""Module containing convenient functions for plotting"""

import matplotlib
import matplotlib.pyplot as plt

def get_func_tuples():
    """
    Return a list of tuples where each tuple is of the form
        (method used in the experiments, label name, plot line style)
    """
    func_tuples = [
        ('met_kdro_lin_eps0.5', 'K-DRO $\epsilon=0.5$', 'm-^'),
        ('met_kdro_lin_eps0.1', 'K-DRO $\epsilon=0.1$', 'r-*'),
        ('met_kdro_lin_eps0.15', 'K-DRO $\epsilon=0.15$', 'g--'),
        ('met_kdro_lin_eps0.2', 'K-DRO $\epsilon=0.2$', 'y-s'),
        ('met_kdro_lin_eps0.3', 'K-DRO $\epsilon=0.3$', 'b-h'),
        ('met_kdro_lin_eps0.05', 'K-DRO $\epsilon=0.05$', 'g-s'),
        ('met_kdro_lin_eps0.01', 'K-DRO $\epsilon=0.01$', 'g-^'),
        ('met_svm_merge', 'Linear (merge)', 'k-v'),
        ('met_svm_ignore', 'Linear (no cert.)', 'k--'),
        ]
    return func_tuples

def get_func2label_map():
    """
    Return a map from method names to plot labels.
    """
    # map: job_func_name |-> plot label
    func_tuples = get_func_tuples()
    #M = {k:v for (k,v) in zip(func_names, labels)}
    M = {k:v for (k,v,_) in func_tuples}
    return M

def get_func2style_map():
    """
    Return a map from method names to matplotlib plot styles 
    """
    # map: job_func_name |-> plot label
    func_tuples = get_func_tuples()
    #M = {k:v for (k,v) in zip(func_names, labels)}
    M = {k:v for (k,_,v) in func_tuples}
    return M

def set_default_matplotlib_options():
    # font options
    font = {
    #     'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 36,
    }
    # matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    # matplotlib.use('cairo')
    matplotlib.rcParams['text.usetex'] = True
    plt.rc('font', **font)
    # plt.rc('lines', linewidth=3, markersize=10)
    # matplotlib.rcParams['ps.useafm'] = True
    # matplotlib.rcParams['pdf.use14corefonts'] = True

    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42


    plt.rc('font', **font)
    plt.rc('lines', linewidth=2)
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True
