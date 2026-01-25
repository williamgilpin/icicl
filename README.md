# icicl

In-context learning traces coherent structures





### Contents

`train_models.py` randomly selects pairs of dynamical systems, generates training and testing data, and trains a model and saves it to a file.

`sweep_probabilities.py` sweeps over a set of trained models and data sets, and compares the kgram conditional probabilities of the model with a ground truth computed via Ulam's method.