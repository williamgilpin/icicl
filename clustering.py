import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin

class UniformGridClusterer(BaseEstimator, ClusterMixin):
    """
    An sklearn-compatible "clusterer" that partitions R^d into a uniform
    axis-aligned grid.

    Parameters:
        n_bins (int or sequence of int, optional): Number of bins per dimension. If an 
            int, the same value is used for all dimensions. If None, and K is provided, a near-cubic grid is chosen.
        K (int, optional): Target number of cells (symbols). If provided and n_bins is None,
            we set n_bins_j = ceil(K^(1/d)) for all j (the actual K_ may exceed K).
        clip (bool, default True): Clip values at the outermost bin instead of raising for boundary hits.

    Attributes:
        n_bins (int or sequence of int, optional): Number of bins per dimension. If an 
            int, the same value is used for all dimensions. If None, and K is provided, a near-cubic grid is chosen.
        K (int, optional): Target number of cells (symbols). If provided and n_bins is None,
            we set n_bins_j = ceil(K^(1/d)) for all j (the actual K_ may exceed K).
        clip (bool, default True): Clip values at the outermost bin instead of raising for boundary hits.
        n_bins_ (array-like of int): Number of bins per dimension.
        K_ (int): Target number of cells (symbols).
        bin_edges_ (list of array-like): Edges of the bins per dimension.
        bin_centers_1d_ (list of array-like): Centers of the bins per dimension.
        labels_ (array-like of int): Labels of the clusters.
        cluster_centers_ (array-like of float): Centers of the clusters.

    Examples:
        >>> from icicl.operators import UniformGridClusterer
        >>> clusterer = UniformGridClusterer(n_bins=10)
        >>> clusterer.fit(X)
        >>> labels = clusterer.predict(X)
        >>> cluster_centers = clusterer.cluster_centers_
        >>> n_bins = clusterer.n_bins_
    """
    def __init__(self, n_bins=None, K=None, clip=True):
        self.n_bins = n_bins
        self.K = K
        self.clip = clip

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, d].")
        N, d = X.shape

        # choose n_bins per dimension
        if self.n_bins is None:
            if self.K is None:
                raise ValueError("Provide either n_bins or K.")
            m = int(np.ceil(self.K ** (1.0 / d)))
            n_bins = np.full(d, m, dtype=int)
        else:
            n_bins = np.asarray(self.n_bins, dtype=int)
            if n_bins.ndim == 0:
                n_bins = np.full(d, int(n_bins), dtype=int)
            if len(n_bins) != d:
                raise ValueError("len(n_bins) must equal X.shape[1].")
            if np.any(n_bins < 1):
                raise ValueError("All n_bins must be >= 1.")

        self.n_bins_ = n_bins
        self.K_ = int(np.prod(self.n_bins_, dtype=int))

        # edges and per-dimension bin centers
        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        # handle constant dims robustly
        span = np.where(xmax > xmin, xmax - xmin, 1.0)
        edges = [np.linspace(xmin[j], xmin[j] + span[j], n_bins[j] + 1) for j in range(d)]
        centers_1d = [0.5 * (e[:-1] + e[1:]) for e in edges]
        self.bin_edges_ = edges
        self.bin_centers_1d_ = centers_1d

        # assign bins
        bj = []
        for j in range(d):
            # indices in [0, n_bins[j]]; subtract 1 to get [0, n_bins[j]-1]
            idx = np.searchsorted(self.bin_edges_[j], X[:, j], side="right") - 1
            if self.clip:
                idx = np.clip(idx, 0, self.n_bins_[j] - 1)
            else:
                if (idx < 0).any() or (idx >= self.n_bins_[j]).any():
                    raise ValueError(f"Samples fall outside grid along dim {j}.")
            bj.append(idx.astype(np.int64))
        bj = np.stack(bj, axis=0)  # shape [d, N]

        # linearize multi-index -> label in [0, K_-1]
        labels = np.ravel_multi_index(bj, dims=tuple(self.n_bins_))
        self.labels_ = labels

        # compute cluster centers for all grid cells (can be large if K_ is large)
        # construct mesh of centers and reshape to [K_, d]
        grids = np.meshgrid(*self.bin_centers_1d_, indexing="ij")
        centers = np.stack([g.reshape(-1) for g in grids], axis=1)
        self.cluster_centers_ = centers  # shape [K_, d]

        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self.n_bins_):
            raise ValueError("X must be 2D with the same number of features as seen in fit.")
        d = X.shape[1]
        bj = []
        for j in range(d):
            idx = np.searchsorted(self.bin_edges_[j], X[:, j], side="right") - 1
            idx = np.clip(idx, 0, self.n_bins_[j] - 1) if self.clip else idx
            bj.append(idx.astype(np.int64))
        bj = np.stack(bj, axis=0)
        return np.ravel_multi_index(bj, dims=tuple(self.n_bins_))
    
