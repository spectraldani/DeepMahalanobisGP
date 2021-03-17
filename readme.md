# Deep Mahalanobis Gaussian process

This repository contains all the code needed to replicate the synthetic experiments presented in the Master's degree dissertation "Contributions on latent projections for Gaussian process modeling", avaible at [Federal University of Ceará's repository](http://www.repositorio.ufc.br/handle/riufc/55580). The code for the model is not neatly packaged into a Python package yet but can be readly imported and used. See any of the Jupyter notebooks for example usage.

## Dependencies
See [requirements.txt](./requirements.txt).

## How to replicate the experiments

Simply run the [main.ipynb](./main.ipynb) Jupyter notebook.

### `retrain_models` variable
The `retrain_models` variable controls the following behavior:

| `retrain_models` value |                               Behaviour                                        |
|:----------------------:|:------------------------------------------------------------------------------:|
| `True`                 | Run experiments and retrain the models from scratch                            |
| `False  `              | Load the hyperparameters used in disseration and run the experiments           |
| `"save"`               | Run experiments, retrain the models from scratch, and save the hyperparameters |
