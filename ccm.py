import numpy as np
import warnings

from scipy.linalg import hankel
from scipy.stats import t as t_dist
from sklearn.neighbors import NearestNeighbors

def progress_bar(i, n, n_bar=20):
    """
    Print a progress bar to stdout

    Args:
        i (int): Current iteration
        n (int): Total number of iterations
        n_bar (int): Number of characters in the progress bar

    Returns:
        None
    """
    idots = int(i / n * n_bar)
    stars = '#' * idots
    spaces = ' ' * (n_bar - idots)
    bar_str = f"[{stars}{spaces}] "
    print(bar_str, end='\r')
    if i == n - 1:
        print("\n")
    return None

def batch_pearson(x, y=None, pvalue=False, eps=1e-8):
    """
    Memory-efficient Pearson correlation along the last axis.

    Args:
        x (ndarray[..., M]): input tensor
        y (ndarray[..., M], optional): second tensor; if None, uses sorted(x)
        pvalue (bool): if True, also return two-tailed p-value
        eps (float): small constant to avoid division by zero

    Returns:
        corr (ndarray[...]): Pearson r
        (optional) p (ndarray[...]): two-tailed p-value
    """
    if y is None:
        # unavoidable O(...×M) cost of sorting, but no additional copy for centering
        y = np.sort(x, axis=-1)

    n = x.shape[-1]

    # compute sums and sums of squares without full-array temporaries
    sum_x  = np.sum(x,  axis=-1)
    sum_y  = np.sum(y,  axis=-1)
    sum_x2 = np.einsum('...i,...i->...', x, x)
    sum_y2 = np.einsum('...i,...i->...', y, y)
    sum_xy = np.einsum('...i,...i->...', x, y)

    # covariance and variances
    cov   = sum_xy - sum_x * sum_y / n
    var_x = sum_x2  - sum_x**2  / n
    var_y = sum_y2  - sum_y**2  / n

    # Pearson r
    denom = np.sqrt(var_x * var_y) + eps
    corr = cov / denom

    if pvalue:
        # Student’s t for Pearson r
        t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + eps))
        p = 2 * t_dist.sf(np.abs(t_stat), df=n - 2)
        return corr, p

    return corr

def batch_spearman(x, y=None, pvalue=False):
    """
    Calculate the Spearman correlation between two sets of time series along the
    last axis

    Args:
        x (ndarray): A tensor of shape (batch, N, M)
        y (ndarray): A tensor of shape (batch, N, M). If None, the indices of the
            time series are used and the Spearman correlation is calculated
            relative to a monotonic function
        pvalue (bool): Whether to return the p-value of the correlation

    Returns:
        corr (ndarray): A tensor of shape (batch, N) containing the Spearman correlation
            between each pair of time series
    """
    if y is None:
        y = np.arange(x.shape[-1])[None, :]
    xrank = np.argsort(np.argsort(x, axis=-1), axis=-1).astype(np.float32)
    yrank = np.argsort(np.argsort(y, axis=-1), axis=-1).astype(np.float32)
    return batch_pearson(xrank, yrank, pvalue=pvalue)


def hankel_matrix(data, q, p=None):
    """
    Find the Hankel matrix dimensionwise for multiple multidimensional 
    time series
    
    Args:
        data (ndarray): An array of shape (N, T, 1) or (N, T, D) corresponding to a 
            collection of N time series of length T and dimensionality D
        q (int): The width of the matrix (the number of features)
        p (int): The height of the matrix (the number of samples)
        
    Returns:
        hmat (ndarray): An array of shape (N, T - p, D, q) containing the Hankel
            matrices for each time series

    """
    
    if len(data.shape) == 3:
        return np.stack([_hankel_matrix(item, q, p) for item in data])
    
    if len(data.shape) == 1:
        data = data[:, None]
    hmat = _hankel_matrix(data, q, p)    
    return hmat
    

def _hankel_matrix(data, q, p=None):
    """
    Calculate the hankel matrix of a multivariate timeseries
    
    Args:
        data (ndarray): T x D multidimensional time series
        q (int): The number of columns in the Hankel matrix
        p (int): The number of rows in the Hankel matrix

    Returns:
        ndarray: The Hankel matrix of shape (T - p, D, q)
    """
    if len(data.shape) == 1:
        data = data[:, None]

    # Hankel parameters
    if not p:
        p = len(data) - q
    all_hmats = list()
    for row in data.T:
        first, last = row[-(p + q) : -p], row[-p - 1 :]
        out = hankel(first, last)
        all_hmats.append(out)
    out = np.dstack(all_hmats)
    return np.transpose(out, (1, 0, 2))[:-1]

def embed_ts(X, m, padding="constant"):
    """
    Create a time delay embedding of a time series or a set of time series

    Args:
        X (array-like): A matrix of shape (n_timepoints, n_dims) or 
            of shape (n_timepoints)
        m (int): The number of dimensions

    Returns:
        Xp (array-like): A time-delay embedding
    """
    X = X.copy()[::-1]
    if padding:
        if len(X.shape) == 1:
            X = np.pad(X, [m, m], padding)
        if len(X.shape) == 2:
            X = np.pad(X, [[m, m], [0, 0]], padding)
        if len(X.shape) == 3:
            X = np.pad(X, [[0, 0], [m, m], [0, 0]], padding)
    Xp = hankel_matrix(X, m)
    Xp = np.moveaxis(Xp, (0, 1, 2), (1, 2, 0))
    Xp = Xp[:, ::-1, ::-1]
    Xp = Xp[:, m-1:-m]
    return Xp



class ConvergentCrossMapping:
    """
    Find the causal relationships among sets of univariate time series.
    The i,j th element of the causal matrix denotes the degree to which i is caused by j
    Equivalently, it measures how much the dynamics x_i is coupled to x_j via the
    matrix sum_j A_{ij} x_j.

    Attributes:
        d_embed (int): Number of dimensions into which to embed the time series. 
            Defaults to 3, the minimum number of dimensions required to resolve
            aperiodic dynamics.
        k (int): Number of neighbors to consider for cross-mapping
        verbose (bool): Whether to show progress bar
        library_sizes (np.ndarray): Array of library sizes to use for cross-mapping. If
            None, use all library sizes
        max_library_size (int): Maximum library size to use for cross-mapping. Defaults 
            to None, in which case the number of library sizes equals the number of 
            timepoints
        minibatch (bool): Whether to use minibatch cross-mapping. Used for large datasets. 
            Defaults to False
        minibatch_size (int): Size of minibatch to use for cross-mapping. Defaults to 1000
        store_intermediates (bool): Whether to store intermediate results
        significance_threshold (float): Threshold for significance in cross-mapping. Defaults
            to None, in which case the causal matrix is not thresholded
        dilation_factor (float): Factor by which decimate the time series, in order to
            test for scaling of causal relationships with the number of timepoints. Defaults
            to 1.5
        sweep_d_embed (bool): Whether to sweep the embedding dimension and return the 
            maximum CCM score across all embedding dimensions. Defaults to False.

    References:
        Sugihara, George, et al. "Detecting causality in complex ecosystems." Science 
            338.6106 (2012): 496-500.

    Examples:
        >>> from ccm import ConvergentCrossMapping
        >>> X = np.random.randn(100, 10)
        >>> ccm = ConvergentCrossMapping(d_embed=3, verbose=True)
        >>> cmat = ccm.fit(X)
        >>> print(cmat) # (10, 10) causal matrix


    """
    def __init__(
            self, 
            d_embed=3, 
            k=None,
            verbose=True, 
            library_sizes=None, 
            max_library_size=None,
            minibatch=False,
            minibatch_size=1000,
            store_intermediates=False, 
            significance_threshold=None,
            dilation_factor=1.5,
            sweep_d_embed=False
        ):
        self.d_embed = d_embed
        self.verbose = verbose
        self.library_sizes = library_sizes
        self.max_library_size = max_library_size
        self.minibatch = minibatch
        self.minibatch_size = minibatch_size
        self.store_intermediates = store_intermediates
        self.k = k
        self.significance_threshold = significance_threshold
        self.dilation_factor = dilation_factor
        self.sweep_d_embed = sweep_d_embed
        if self.k is None:
            self.k = self.d_embed + 1



    def compute_crossmap(self, Xe, Y, X=None, stride=-1, tpred=0, tol=1e-10):
        """
        Use cross-mapping to to predict Y from Xe

        Args:
            Xe (np.ndarray): A matrix of shape (n_genes, nt, d_embed)
            Y (np.ndarray): A matrix of shape (n_genes, nt, 1)
            stride (int): Stride to use for cross-mapping. Defaults to -1, in which case
                the entire time series is used
            tpred (int): Timepoint to predict. Defaults to 0, in which case the last timepoint
                is predicted

        Can modify to hold out test
        """
        m, ntx, d_embed = Xe.shape[0], Xe.shape[1], Xe.shape[2]
        nt = Y.shape[0]
        if len(Y.shape) < 3:
            Y = Y.T[..., None] # (n_genes, nt, 1)
        else:
            Y = Y.T

        causal_matrix = np.zeros((m, m))

        ## Outer index runs over causes, which we use for lookups into the downstream
        ## causees. 
        for i in range(m):

            tree = NearestNeighbors(n_neighbors=min(ntx, self.k+1), algorithm='auto', metric='euclidean')
            tree.fit(Xe[i])
            dists, idx  = tree.kneighbors(Xe[i])
            dists, idx = dists[:, 1:].T, idx[:, 1:].T # Remove self distance
            dmin = np.min(dists, axis=0) + 1e-8
            wgts = np.exp(-dists / dmin) # (k, nt) weights of k neighbors for each point

            y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2)
            y_pred, y_target = y_pred[:, :, 0].copy(), Y[:, :y_pred.shape[1], 0].copy()

            y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2)
            y_target = Y[:, :y_pred.shape[1], :].copy()
            y_pred, y_target = np.squeeze(y_pred), np.squeeze(y_target)

            ## Score the prediction, weighted by the p-value
            rho, pval = batch_pearson(y_pred, y_target, pvalue=True)

            ## Set any non-significant CCM scores to zero
            if self.significance_threshold is not None:
                causal_matrix[pval > self.significance_threshold] = 0

            causal_matrix[i] = rho.copy() * (1 - pval)

        ## Fully vectorized triggers memory error
        # idx = np.argsort(dX, axis=1)[:, 1:self.k+1]
        # dists = np.sort(dX, axis=1)[:, 1:self.k+1]
        # wgts = np.exp(-dists / np.min(dists, axis=1, keepdims=True))
        # Xe_sel = np.moveaxis(Xe[:, idx.T], (0,1,2,3,4), (1,2,3,0,4))
        # y_pred = np.sum(Xe_sel * np.swapaxes(np.swapaxes(wgts, 1, 2)[None, ..., None], 0, 1), axis=3)
        # y_true = Xe.copy()
        # ytc = y_true - np.mean(y_true, axis=-1, keepdims=True)
        # ypc = y_pred - np.mean(y_pred, axis=-1, keepdims=True)
        # rho = np.sum(ytc * ypc, axis=-1) / np.sqrt(np.sum(ytc ** 2, axis=-1) * np.sum(ypc ** 2, axis=-1))
        # causal_matrix = np.mean(rho.copy(), axis=-1)

        np.fill_diagonal(causal_matrix, 0)
        return causal_matrix

    def fit(self, X, y=None):
        """
        Fit the model to the data

        Args:
            X (np.ndarray): Upstream data matrix of shape (n_timepoints, n_features)
            y (np.ndarray): Downstream data matrix of shape (n_timepoints, n_features)
                Defaults to None, in which case X is used as the target

        Returns:
            np.ndarray: Stack of causal matrices

        """
        if self.sweep_d_embed:
            self.d_embed = 2
            self.sweep_d_embed = False
            cmat = self.fit(X, y)
            self.sweep_d_embed = True 
            for d_embed in np.arange(3, 12):
                if self.verbose:
                    print(f"Fitting model with d_embed: {d_embed}")
                self.d_embed = d_embed
                self.sweep_d_embed = False
                cmat = np.nanmax([cmat, self.fit(X, y)], axis=0)
                self.sweep_d_embed = True
            return cmat
        if y is None:
            y = X


        self.n = X.shape[0]

        if self.library_sizes is None:
            if self.max_library_size is None:
                # self.library_sizes = np.arange(1, int(np.floor(self.n  / (self.d_embed + 1))))[::-1]
                self.library_sizes = np.unique((self.dilation_factor ** np.arange(0, int(np.floor(np.log(self.n  / (self.d_embed + 1))/np.log(self.dilation_factor))))).astype(int))[::-1]
            else:
                self.library_sizes = np.unique(np.linspace(1, int(np.floor(self.n  / (self.d_embed + 1))), self.max_library_size).astype(int))[::-1]
        ## check that library sizes increase monotonically
        if not np.all(np.diff(self.library_sizes) <= 0):
            warnings.warn("Stride sizes must decrease monotonically. Sorting library sizes.")
            self.library_sizes = np.sort(self.library_sizes)[::-1]

        all_causmat = np.zeros((len(self.library_sizes), X.shape[1], X.shape[1]))

        ## Iterate over library sizes to test robustness of causal matrix
        Xe = embed_ts(X, m=self.d_embed)
        # corr_stream = StreamingCorrelation(X.shape[1], lambda i: i)
        for i, stride in enumerate(self.library_sizes):
            
            if self.verbose:
                progress_bar(i, len(self.library_sizes))

            subset_inds = np.arange(0, Xe.shape[1], stride)
            if self.minibatch and Xe[:, ::stride].shape[1] > self.minibatch_size:
                subset_inds = subset_inds[:self.minibatch_size]
                
            all_causmat[i] = self.compute_crossmap(Xe[:, subset_inds], y[:-(self.d_embed - 1)][subset_inds])

        if self.store_intermediates:
            self.ac = all_causmat.copy()

        rho_mono = batch_spearman(all_causmat.T, pvalue=False) 
        np.fill_diagonal(rho_mono, 0)
        cause_matrix = all_causmat[-1] * np.abs(rho_mono)

        return cause_matrix