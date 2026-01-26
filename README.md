# icicl

In-context learning traces coherent structures in transformers trained on dynamical systems.

### Dependencies

+ numpy
+ scipy
+ matplotlib
+ scikit-learn
+ pytorch
+ [dysts](https://github.com/GilpinLab/dysts)

### Contents

The scripts for running the experiments are:

`train_models.py` randomly selects pairs of dynamical systems, generates training and testing data, and trains a model and saves it to a file.

`sweep_probabilities.py` sweeps over a set of trained models and datasets, and compares the kgram conditional probabilities of the model with a ground truth computed via Ulam's method.

The utilities for the experiments are:

`transitions.py` contains the functions for computing empirical transition probabilities from a transformer model.

`markov.py` and `operators.py` contain the functions for computing ground truth transition probabilities from a dynamical system.

`incontext.ipynb` is a notebook for analyzing the results of the experiments.

