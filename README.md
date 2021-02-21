# Code for Kernel Distributionally Robust Optimization

## Authors

Jia-Jie Zhu, Wittawat Jitkrittum

## Citing this repository

```latex
@misc{zhu2020kernel,
      title={Kernel Distributionally Robust Optimization}, 
      author={Jia-Jie Zhu and Wittawat Jitkrittum and Moritz Diehl and Bernhard Schölkopf},
      year={2020},
      eprint={2006.06981},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```

## Instruction

`kdro` is a folder for the Python module `kdro`.
The repository contains two experiments in the [KDRO paper](https://arxiv.org/abs/2006.06981):

- Robust least squares
- Distributionally robust classification using SFG-DRO

The executable files for those two experiments are located in the `examples/`
folder. See the `README` files therein. Please first follow the instructions below to set up the environment first.

## Dependency

* dill
* numpy
* scipy
* cvxpy (see https://www.cvxpy.org/install/)
* [Mosek solver](https://www.mosek.com/)
  * optionally, install the solvers of your choice. 
* cvxopt
* jupyter
* matplotlib
* sklearn

For testing SFG-DRO, you need

* pytorch
* torchvision

## Development

To install the package for development purpose, follow the following steps: 

1. Make a new Anaconda environment (if you use Anaconda. Recommended) for
this project. Switch to this environment.

2. `cd` to the folder that contains this READMD.md file.

3. Issue the following command in a terminal to install the `kdro` Python
package from this repository.

        pip install -e .

    This will install the package to your environment selected in Step 1.

4. In a Python shell with the environment activated, make sure that you can
`import kdro` without any error.

The `-e` flag offers an "edit mode", meaning that changes to any files in
this repo will be reflected immediately in the imported package.

