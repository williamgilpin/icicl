import numpy as np
import warnings
import os
import uuid

# from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr
from scipy.special import betainc

from .utils import embed_ts, multivariate_embed_ts
from .utils import batch_pearson, batch_spearman, flatten_along_axis, StreamingCorrelation
from .utils import progress_bar, debug_print
from .utils import hollow_matrix, banded_matrix, max_linear_correlation_ridge

## Disable optimization warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", message="The iteration is not making good progress")
warnings.filterwarnings("ignore", message="overflow encountered")
warnings.filterwarnings('ignore', message='Forecast type not recognized')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

## Where to write temporary files
temp_dir = ""

relu = lambda x: np.maximum(0, x)
# from umap.umap_ import fuzzy_simplicial_set

import hnswlib
def neighbors_hnswlib(X, metric='euclidean', k=20):
    """
    Use hnswlib for approximate k-nearest neighbors of each point in a dataset.
    Returns the indices and distances of the neighbors.

    Args:
        X (np.ndarray): dataset of shape (n, d)
        metric (str): distance metric to use
        k (int): number of nearest neighbors to use in the distance calculation

    Returns:
        idx (np.ndarray): indices of the neighbors
        dists (np.ndarray): distances to the neighbors
    """
    n, d = X.shape
    if metric == 'euclidean':
        metric = 'l2'
    elif metric == 'cosine':
        metric = 'angular'
    else:
        raise ValueError(f"Metric {metric} not supported")

    # Initialize the HNSW index: space='l2' for Euclidean, 'cosine' for angular distance
    index = hnswlib.Index(space=metric, dim=d)
    # Prepare index to hold n elements; tune M and ef_construction as desired
    index.init_index(max_elements=n, M=12, ef_construction=200)
    # Add all vectors (cast to float32) with integer labels 0…n−1
    index.add_items(X, np.arange(n))
    # Set query-time parameter for recall/speed trade-off
    # index.set_ef(200)
    # Perform k+1 neighbor queries for each point
    idx, dists = index.knn_query(X, k+1) # Both are (n, k+1)
    return idx, np.sqrt(dists)


def simplex_neighbors(X, metric='euclidean', k=20, tol=1e-6):
    """
    Compute the distance between points in a dataset using the simplex distance metric.

    Args:
        X (np.ndarray): dataset of shape (n, d)
        metric (str): distance metric to use
        k (int): number of nearest neighbors to use in the distance calculation
        tol (float): tolerance for the distance calculation

    Returns:
        np.ndarray: distance matrix of shape (n, m)

    """

    idx, dists = neighbors_hnswlib(X, metric, k)

    dists, idx = dists[:, 1:].T, idx[:, 1:].T
    
    rhos = dists[0]
    sigmas = np.array([find_sigma(drow, tol=tol)[0] for drow in dists.T])
    sigmas += tol # Add a small tolerance to avoid division by zero

    # sigmas, dists_transformed = compute_sigmas_vectorized(dists, tol=1e-6)

    wgts = np.exp(-relu(dists - rhos[None, :]) / sigmas[None, :])
    return wgts, idx, sigmas


def find_sigma(dists, tol=1e-6):
    """
    Given a list of distances to k nearest neighbors, find the sigma for each point

    Args:
        dists (np.ndarray): A matrix of shape (k,)
        tol (float): The tolerance for the sigma

    Returns:
        float: The sigma corresponding to the neighborhood scale
        np.ndarray: The transformed distances
    """
    k = dists.shape[0]
    rho = np.min(dists)
    func = lambda sig: sum(np.exp(-relu(dists - rho) / (sig + tol))) - np.log2(k)
    jac = lambda sig: sum(np.exp(-relu(dists - rho) / (sig + tol)) * relu(dists - rho)) / (sig + tol)**2
    sigma = fsolve(func, rho, fprime=jac, xtol=tol)[0]
    dists_transformed = np.exp(-relu(dists - rho) / (sigma + tol))
    return sigma, dists_transformed


def compute_sigmas_vectorized(dists, tol=1e-6, max_iter=50, jac_eps=1e-6):
    """
    Vectorized Newton's method to solve for sigma for each column of the distance matrix.

    Args:
        dists (np.ndarray): shape (k, n_points)
        tol (float): convergence tolerance
        max_iter (int): maximum iterations

    Returns:
        sigmas (np.ndarray): shape (n_points,)
        dists_transformed (np.ndarray): shape (k, n_points)
    """
    k, n = dists.shape
    rho = np.min(dists, axis=0)                 # (n,)
    D = np.maximum(dists - rho, 0.0) + tol                # (k, n)
    sigma = rho.copy()                              # initial guess

    for _ in range(max_iter):
        denom = sigma + tol                         # guaranteed ≥ tol
        E = np.exp(-D / denom)                      # (k, n)
        f = E.sum(axis=0) - np.log2(k)              # (n,)
        if np.all(np.abs(f) < tol):
            break
        jac = (E * D).sum(axis=0) / (denom**2)      # (n,)
        jac_safe = np.maximum(jac, jac_eps)         # avoid zero
        sigma -= f / jac_safe                       # Newton update
        sigma = np.maximum(sigma, tol)              # enforce σ ≥ tol

    dists_transformed = np.exp(-D / (sigma + tol))
    return sigma, dists_transformed


def calculate_sigma(X0, d_embed=4, tol=1e-6, channelwise=True):
    """Given a matrix of time series, calculate the sigma for each time series.

    Args:
        X0: (ntx, d) matrix of time series
        d_embed: embedding dimension
        tol: tolerance for simplex neighbors
        channelwise: whether to embed each time series separately or not 

    Returns:
        all_sig: (ntx, d) matrix of sigmas if channelwise is True, otherwise 
            (ntx, 1) matrix of sigmas
    """
    X = X0.squeeze().copy()
    if channelwise:
        Xe = embed_ts(X, m=d_embed)
    else:
        Xe = X[None, ...]
    m, ntx, d_embed = Xe.shape[0], Xe.shape[1], Xe.shape[2]
    all_sig = list()
    for Xe_i in Xe:
        wgts, idx, sig = simplex_neighbors(Xe_i, k=min(ntx - 1, d_embed + 1), tol=tol)
        all_sig.append(sig)
    all_sig = np.array(all_sig)
    if channelwise:
        all_sig = np.pad(all_sig, [[0, 0], [0, d_embed - 1]], mode="edge")
    return all_sig


def data_processing_inequality(M, i, j, k):
    """
    Filter out edges resulting from indirect relationships. If the matrix M represents
    mutual information, then this function filters out edges where X -> Y -> Z induces
    a non-zero mutual information between X and Z.

    The criterion is that if I[i, k] < min(I[i, j], I[j, k]), then the edge from i to k 
    is filtered.

    Args:
        M (np.ndarray): Mutual information matrix
        i (int): Source node
        j (int): Intermediate node
        k (int): Target node

    Returns:
        tuple: Source and target nodes
    """
    m_ij, m_ik, m_jk = M[i, j], M[i, k], M[j, k]
    lowest = m_ij
    edge = (i, j)
    if m_ik < lowest:
        lowest = m_ik
        edge = (i, k)
    if m_jk < lowest:
        edge = (j, k)
    
    return edge

def filter_loops(M0, max_neighbors=100):
    """
    Filter out loops from the mutual information matrix based on the Data Processing 
    Inequality (DPI).

    Args:
        M0 (np.ndarray): Mutual information matrix
        max_neighbors (int): Maximum number of neighbors to consider for each node

    Returns:
        np.ndarray: Mutual information matrix with loops removed
    """
    M = M0.copy()
    n = M.shape[0]

    # Sparsify large matrices
    if n > max_neighbors:
        n_links_per_node = max_neighbors / n
        threshold = np.percentile(M.ravel(), 100 * (1 - n_links_per_node))
        M[np.abs(M) < threshold] = 0
    
    # Build adjacency list (neighbors[i] = indices j where M[i,j] != 0)
    neighbors = []
    for i in range(n):
        nbrs = set(np.where(M[i] != 0)[0])
        neighbors.append(nbrs)
    
    set_to_zero = []
    for i in range(n):
        for j in neighbors[i]:
            # Intersect neighbors of i and j
            common_k = neighbors[i].intersection(neighbors[j])
            # Optionally discard i and j from the intersection:
            common_k.discard(i); common_k.discard(j);
            
            for k in common_k:
                # Decide whether to zero out the edge i -> k
                res = data_processing_inequality(M, i, j, k)
                if res is not None:
                    set_to_zero.append(res)
    
    if set_to_zero:
        # Convert list of tuples into array so we can do M[set_to_zero] = 0
        set_to_zero = np.array(set_to_zero).T
        M[tuple(set_to_zero)] = 0

    return M

from scipy.optimize import fsolve, minimize_scalar
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, MultiTaskElasticNetCV

class CausalDetection:
    """
    Find the causal relationships among sets of univariate time series.
    The i,j th element of the causal matrix denotes the degree to which i is caused by j
    Equivalently, it measures how much the dynamics x_i is coupled to x_j via the
    matrix sum_j A_{ij} x_j.

    Attributes:
        d_embed (int): Number of dimensions to embed the time series into
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
        neighbors (str): Type of neighbors to use for cross-mapping. Defaults to "simplex"
            which uses fuzzy simplicial set neighbors, which take longer but are more accurate  
            than classic 'knn' for K nearest neighbors.
        forecast (str): Type of forecast to use for cross-mapping, either "sum" or "smap".
            Defaults to "smap"
        prune_indirect (bool): Whether to prune indirect relationships due to causal
            transitivity. This helps reduce false positives. Defaults to False
        ensemble (bool): Whether to use ensemble-level cross-mapping. Defaults to False
        significance_threshold (float): Threshold for significance in cross-mapping. Defaults
            to None, in which case the causal matrix is not thresholded
        dilation_factor (float): Factor by which decimate the time series, in order to
            test for scaling of causal relationships with the number of timepoints. Defaults
            to 1.5
        sweep_d_embed (bool): Whether to sweep the embedding dimension. Defaults to False
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
            neighbors="simplex", 
            forecast="smap",
            prune_indirect=False,
            ensemble=True,
            significance_threshold=None,
            dilation_factor=1.5,
            sweep_d_embed=False
        ):
        self.n_genes = None
        self.d_embed = d_embed
        self.causal_matrix = None
        self.all_causmat = None
        self.verbose = verbose
        self.library_sizes = library_sizes
        self.max_library_size = max_library_size
        self.minibatch = minibatch
        self.minibatch_size = minibatch_size
        self.store_intermediates = store_intermediates
        self.k = k
        self.neighbors = neighbors
        self.forecast = forecast
        self.prune_indirect = prune_indirect
        self.ensemble = ensemble
        self.significance_threshold = significance_threshold
        self.dilation_factor = dilation_factor
        self.sweep_d_embed = sweep_d_embed
        if self.k is None:
            self.k = self.d_embed + 1

        if self.forecast == "simplex" and self.neighbors == "knn":
            warnings.warn("Simplex neighbors and S-map forecast are not compatible, falling back to sum over neighbors")
            self.forecast = "sum"

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
            if self.neighbors == "simplex":
                wgts, idx, sig = simplex_neighbors(Xe[i], k=min(ntx - 1, self.k), tol=tol)
            else:
                if self.neighbors != "knn":
                    warnings.warn("Neighbor type not recognized, falling back to K nearest neighbors")
                tree = NearestNeighbors(n_neighbors=min(ntx, self.k+1), algorithm='auto', metric='euclidean')
                tree.fit(Xe[i])
                dists, idx  = tree.kneighbors(Xe[i])
                dists, idx = dists[:, 1:].T, idx[:, 1:].T # Remove self distance
                dmin = np.min(dists, axis=0) + 1e-8
                wgts = np.exp(-dists / dmin) # (k, nt) weights of k neighbors for each point

            if self.forecast == "smap":
                # print(wgts.shape, idx.shape)
                Ax = (Xe[:, idx.T, :1] * wgts.T[None, ..., None]).squeeze().copy()  # Input batch matrix (B x T x M)
                Ay = (Y[:, idx.T] * wgts.T[None, ..., None]).squeeze().copy()  # Input batch matrix (B x T x M)
                Cx = Xe[:, :idx.shape[1], 0].squeeze().copy()  # Target batch matrix (B x T)
                Cy = Y[:, :idx.shape[1], 0].squeeze().copy() # Not used. Target batch matrix (B x T)
                M = Ax.shape[2] # Number of features
                lambda_reg = 0.5 * M # Scale regularization parameter to the number of features
                I = np.eye(M)[None, :, :]  # Identity matrix broadcasted across batches (B x M x M)
                AtA = np.einsum('btm,btn->bmn', Ax, Ax)  ## Compute A^T @ A over batch (B x M x M) --> (B x M x M)
                AtC = np.einsum('btm,bt->bm', Ax, Cx)    ## Compute A^T @ C over batch (B x M) --> (B x M)
                B_sol = np.linalg.solve(AtA + lambda_reg * I, AtC[:, :, None])  # Solve (B x M x 1)
                B_sol = B_sol.squeeze(-1)  # Final shape (B x M)
                # B_sol = np.ones_like(B_sol)
                y_pred = np.einsum('btm,bm->bt', Ay, B_sol) ## Predict Y with the B coeffs fit from X
                y_target = Y[:, :y_pred.shape[1], 0].copy()
            else:
                if self.forecast != "sum":
                    warnings.warn("Forecast type not recognized, falling back to sum over neighbors")
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


    def compute_crossmap_ensemble(self, Xe, Y, batch_indices=None, stride=-1, tpred=0, tol=1e-10):
        """
        Use cross-mapping to to predict Y from Xe

        Args:
            Xe (np.ndarray): A matrix of shape (n_genes, nt, d_embed)
            Y (np.ndarray): A matrix of shape (n_genes, nt, 1)
            batch_indices (np.ndarray): A list of indices to use for batched cross-mapping.
                Defaults to None, in which case all indices are used
            stride (int): Stride to use for cross-mapping. Defaults to -1, in which case
                the entire time series is used
            tpred (int): Timepoint to predict. Defaults to 0, in which case the last timepoint
                is predicted

        Can modify to hold out test
        """
        # print("a", flush=True)
        m, ntx, d_embed = Xe.shape[0], Xe.shape[1], Xe.shape[2]
        nt = Y.shape[0]
        if len(Y.shape) < 3:
            Y = Y.T[..., None] # (n_genes, nt, 1)
        else:
            Y = Y.T
        # print("b", flush=True)
        # all_y_pred = np.zeros((m, m, ntx))

        ## If the prediction array would be larger than 500MB, store in a temporary file
        hash_id = uuid.uuid4().hex
        fname = os.path.join(temp_dir, f"temp_rice_{hash_id}.npy")
        if 4 * m * m * ntx < 5e8: # 500MB
            all_y_pred = np.zeros((m, m, ntx))
        else:
            if self.verbose: print(f"Storing temporary file at {fname}", flush=True)
            all_y_pred = np.memmap(
                fname, 
                dtype=np.float64, 
                mode="w+", 
                shape=(m, m, ntx)
            )
        
        # hash_id = uuid.uuid4().hex
        # fname = os.path.join(temp_dir, f"temp_rice_{hash_id}.npy") # 1GB
        # if self.verbose: print(f"Storing temporary file at {fname}", flush=True)
        # all_y_pred = np.memmap(
        #     fname, 
        #     dtype=np.float64, 
        #     mode="w+", 
        #     shape=(m, m, ntx)
        # )

        k = min(ntx - 1, self.k)
        causal_matrix = np.zeros((m, m))
        # I = np.eye(m, dtype=Xe.dtype)[None, :, :]       # shape (M, M), not (B, M, M)
        I = np.eye(k)[None, :, :]
        # print("c", flush=True)
        lambda_reg = 0.5 * m * 100000 # Scale regularization parameter to the number of features
        
        # print("d", flush=True)
        ## Outer index runs over causes, which we use for lookups into the downstream
        ## causees. 
        y_target = Y[:, :ntx, 0].copy()
        for i in range(m):
            if self.neighbors == "simplex":
                ## This is the slow step
                wgts, idx, sig = simplex_neighbors(Xe[i], k=k, tol=tol)
            else:
                if self.neighbors != "knn":
                    warnings.warn("Neighbor type not recognized, falling back to K nearest neighbors")
                tree = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
                tree.fit(Xe[i])
                dists, idx  = tree.kneighbors(Xe[i])
                dists, idx = dists[:, 1:].T, idx[:, 1:].T # Remove self distance
                dmin = np.min(dists, axis=0)
                wgts = np.exp(-dists / dmin) # (k, nt) weights of k neighbors for each point

            ## Regularized smap often seems to overfit
            if self.forecast == "smap":

                # ## toggle here 
                Ax = (Xe[:, idx.T, :1] * wgts.T[None, ..., None]).squeeze()  # Input batch matrix (B x T x M)
                Ay = (Y[:, idx.T] * wgts.T[None, ..., None]).squeeze()  # Input batch matrix (B x T x M)
                Cx = Xe[:, :idx.shape[1], 0].squeeze().copy()  # Target batch matrix (B x T)
                # # Cy = Y[:, :idx.shape[1], 0].squeeze().copy() # Not used. Target batch matrix (B x T)
                
                # AtA = np.einsum('btm,btn->bmn', Ax, Ax)  ## Compute A^T @ A over batch (B x M x M) --> (B x M x M)
                # AtC = np.einsum('btm,bt->bm', Ax, Cx)    ## Compute A^T @ C over batch (B x M) --> (B x M)
                AtA = Ax.transpose(0, 2, 1) @ Ax  # This is equivalent to the first einsum
                AtC = (Ax.transpose(0, 2, 1) @ Cx[..., None]).squeeze(-1)  # This is equivalent to the second einsum
                # print(Ax.shape, Ay.shape, Cx.shape, AtA.shape, AtC.shape, flush=True)

                # inverse = np.linalg.inv(AtA + lambda_reg * I)
                # B_sol = np.einsum('bmn,bm->bn', inverse, AtC)[:, :, None]
                # B_sol = B_sol.squeeze(-1)  # Final shape (B x M)

                B_sol = np.linalg.solve(AtA + lambda_reg * I, AtC[:, :, None]).squeeze(-1)
                y_pred = np.einsum('btm,bm->bt', Ay, B_sol) ## Predict Y with the B coeffs fit from X
                ## Include all context points
                # y_pred = flatten_along_axis((Ay * B_sol[:, None]).squeeze(), (0, 2)).T

            else:
                if self.forecast != "sum":
                    warnings.warn("Forecast type not recognized, falling back to sum over neighbors")
                y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2)
                y_pred, y_target = y_pred[:, :, 0].copy(), Y[:, :y_pred.shape[1], 0].copy()
                y_pred = np.sum(Y[:, idx.T] * wgts.T[None, ..., None], axis=2)
                y_target = Y[:, :y_pred.shape[1], :].copy()
                y_pred, y_target = np.squeeze(y_pred), np.squeeze(y_target)

            all_y_pred[i] = y_pred

        # For each response, fit a ridge regression model over timepoints
        # To assign a causal score to each upstream gene
        ## This might be the bottleneck when stride gets close to 1
        # for i in range(m):

        #     # Loop over responses
        #     Xd, Yd = all_y_pred[:, i].T,  y_target[i][:, None] # sweep downstreams
        #     # Xd, Yd = all_y_pred[i].T, all_y_true[i][:, None] # sweep upstreams
           
        #     Xd = (Xd - np.mean(Xd, axis=0, keepdims=True))
        #     Yd = (Yd - np.mean(Yd, axis=0, keepdims=True))
        #     # print("2", flush=True)

        #     # lambda_val = 1e0 / Xd.shape[1] * 1e10
        #     # ridge = Ridge(alpha=lambda_val, fit_intercept=False)
        #     # ridge.fit(Xd, Yd)
        #     # Yd_pred = ridge.predict(Xd) # error doesn't distinguish upstream
        #     # corr = pearsonr(Yd_pred.squeeze(), Yd.squeeze())
        #     # corr = corr[0] * (1 - corr[1])
        #     # r2 = corr
        #     # A = ridge.coef_.T.squeeze()
        #     # A = np.abs(A)
        #     # A *= r2

        #     ## Strong regularization limit
        #     A = (Xd.T @ Yd).T
        #     # print("3", flush=True)
        #     Yd_pred = Xd @ A.T
        #     # print("4", flush=True)
        #     corr = pearsonr(Yd_pred.squeeze(), Yd.squeeze())
        #     # print("5", flush=True)
        #     corr = corr[0] * (1 - corr[1])
        #     # print("6", flush=True)
        #     # corr = np.nan_to_num(corr, nan=1.0) # constant time series no variance
        #     r2 = corr 
        #     # print("7", flush=True)
        #     A = np.abs(A)
        #     # print("8", flush=True)
        #     A *= r2
        #     # print("9", flush=True)
        #     causal_matrix[:, i] = A.squeeze()
        

        for i in range(m):
            # First pass: compute sums for means, inner products, and Y variance
            n = 0
            sum_x = np.zeros(m)
            sum_y = 0.0
            sum_y2 = 0.0
            sum_xy = np.zeros(m)

            for t in range(ntx):
                x = all_y_pred[:, i, t]       # shape (m,)
                y = y_target[i][t]            # scalar
                n += 1
                sum_x  += x
                sum_y  += y
                sum_y2 += y*y
                sum_xy += x * y

            mu_x = sum_x / n
            mu_y = sum_y / n

            # Cross‐product A_j = ∑ₜ (xₜⱼ − μₓⱼ)(yₜ − μᵧ)
            A = sum_xy - n * mu_x * mu_y

            # Variance of Y: ∑ₜ (yₜ − μᵧ)²
            sum_y2c = sum_y2 - n * mu_y**2

            # streaming Pearson correlation between predicted and actual Y
            sum_pred2 = 0.0
            sum_pred_y = 0.0
            for t in range(ntx):
                x = all_y_pred[:, i, t]
                x_cent = x - mu_x
                y_cent = y_target[i][t] - mu_y
                y_pred = np.dot(x_cent, A)
                sum_pred2  += y_pred**2
                sum_pred_y += y_pred * y_cent

            # Pearson r between Y_pred and Y, with exact p-value given by the beta distribution
            r = sum_pred_y / np.sqrt(sum_pred2 * sum_y2c)
            a = n / 2.0 - 1.0
            pval = 2 * betainc(a, a, 0.5 * (1 - abs(r)))
            r2 = r * (1 - pval)
            causal_matrix[:, i] = np.abs(A) * r2

        ## if temp_hashid.npy is on disk, delete it
        if os.path.exists(fname):
            os.remove(fname)

        np.fill_diagonal(causal_matrix, 0)
        return causal_matrix

    def _strided_op(self, X, op=lambda x: x):
        """
        Apply a function repeated to strided subsets of a time series
        """
        self.library_sizes = np.unique((self.dilation_factor ** np.arange(0, int(np.floor(np.log(X.shape[0]  / (self.d_embed + 1))/np.log(self.dilation_factor))))).astype(int))[::-1]
        for stride in self.library_sizes:
            yield op(X[..., ::stride])

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

        # ## If minibatch is enabled, only use library sizes that are less than the minibatch size
        # if self.minibatch:
        #     self.library_sizes = self.library_sizes[self.library_sizes < self.minibatch_size]

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
                
            if self.ensemble:
                all_causmat[i] = self.compute_crossmap_ensemble(Xe[:, subset_inds], y[:-(self.d_embed - 1)][subset_inds])
            else:
                all_causmat[i] = self.compute_crossmap(Xe[:, subset_inds], y[:-(self.d_embed - 1)][subset_inds])

            # corr_stream.update(all_causmat[i])

        if self.store_intermediates:
            self.ac = all_causmat.copy()

        # batch_spearman(np.random.randn(X.shape[1], X.shape[1], len(self.library_sizes)), pvalue=False)

        # rho_mono = batch_pearson(all_causmat.T, pvalue=False) # Memory error when batch dimension is too large    
        rho_mono = batch_spearman(all_causmat.T, pvalue=False) # Memory error when batch dimension is too large
        # rho_mono = corr_stream.finalize()
        np.fill_diagonal(rho_mono, 0)
        cause_matrix = all_causmat[-1] * np.abs(rho_mono)

        ## Prune indirect connections
        if self.prune_indirect:
            cause_matrix = filter_loops(cause_matrix)

        return cause_matrix