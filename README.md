# icicl

In-context learning traces coherent structures in transformers trained on dynamical systems.

### Dependencies

+ numpy
+ scipy
+ matplotlib
+ scikit-learn
+ torch
+ [dysts](https://github.com/GilpinLab/dysts)

### Contents

The main experimental results are given by notebooks:

`train_single_model.ipynb` is a notebook for training a single model on a single dynamical system, showing out-of-distribution generalization performance and double descent behavior.

`analyze_embedding_dimension.ipynb` is a notebook for analyzing trained models to probe how the properties of the transformers change with the embedding dimension.

In order to run for multiple randomly-sampled dynamical systems, the scrips for running the experiments are:

`transitions.py` contains the functions for computing empirical transition probabilities from a transformer model.

`markov.py` and `operators.py` contain the functions for computing ground truth transition probabilities from a dynamical system.

`train_models.py` randomly selects pairs of dynamical systems, generates training and testing data, and trains a model and saves it to a file.

`sweep_probabilities.py` sweeps over a set of trained models and datasets, and compares the kgram conditional probabilities of the model with a ground truth computed via Ulam's method.


