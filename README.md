# icicl

In-context learning of transfer operators in transformers trained on dynamical systems.

<img src="resources/github.ong" alt="Transfer operator of a trained model" width="100%"/>

### Dependencies

+ numpy
+ scipy
+ matplotlib
+ scikit-learn
+ torch
+ [dysts](https://github.com/GilpinLab/dysts)

### Usage

The main experimental results are given by notebooks:

`train_single_model.ipynb` is a notebook for training a single model on a single dynamical system, showing out-of-distribution performance and double descent behavior.

`analyze_embedding_dimension.ipynb` is a notebook for analyzing trained models to probe how the properties of the transformers change with the embedding dimension.

`estimate_transfer_operator.ipynb` is a notebook for estimating the transfer operator of a trained model and comparing it to the ground truth transfer operator of the fully-observed test system.

`icicl/` contains the utility functions for the experiments.


