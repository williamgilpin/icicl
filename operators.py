import numpy as np
from sklearn.cluster import KMeans


def invariant_distribution(P, tol=1e-12, max_iter=10_000, fix_dangling=True, seed=None):
    """
    Given a row-stochastic transition matrix P, returns a stationary (invariant) 
    distribution π satisfying π^T P = π^T, sum(π)=1, π>=0. This can be used to compute 
    the long-term behavior of a Markov chain.
    Uses power iteration on P^T. If P has zero-rows (no outgoing transitions), optionally
    makes them uniform (common 'dangling' fix) to ensure stochasticity.

    Args:
        P (array-like): row-stochastic transition matrix (K,K)
        tol (float): convergence tolerance on ||π_{k+1}-π_k||_1.
        max_iter (int): Maximum number of iterations.
        fix_dangling (bool): If True, replace any all-zero row with uniform distribution.
        seed (int or None): Random seed for initialization.

    Returns:
        pi (array-like): Stationary distribution (K,)
    """
    P = np.asarray(P, dtype=float)
    K = P.shape[0]
    if P.shape != (K, K):
        raise ValueError("P must be square")

    # Handle dangling states (zero-outgoing-prob rows)
    if fix_dangling:
        row_sums = P.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.squeeze() == 0)
        if np.any(zero_rows):
            P = P.copy()
            P[zero_rows, :] = 1.0 / K

    # Power iteration on P^T
    rng = np.random.default_rng(seed)
    pi = rng.random(K)
    pi /= pi.sum()
    Pt = P.T
    for _ in range(max_iter):
        new = Pt @ pi
        new_sum = new.sum()
        if new_sum == 0 or not np.isfinite(new_sum):
            raise FloatingPointError("Iteration produced non-finite or zero vector")
        new /= new_sum
        if np.linalg.norm(new - pi, 1) < tol:
            pi = new
            break
        pi = new
    else:
        # Not converged; still return best iterate
        pass
    # Ensure nonnegative and normalized
    pi = np.maximum(pi, 0)
    pi /= pi.sum()
    return pi



def transition_matrix(seq, vocab_size, tau=1, normalize=True, dtype=float):
    """
    Estimate the lag-τ Markov transition matrix P where
    P[i, j] = Pr(s_t = j | s_{t-τ} = i) from an integer sequence.

    Args:
        seq (Sequence[int]): Observed states, each in [0, vocab_size).
        vocab_size (int): Number of possible states.
        tau (int, default=1): Lag between conditioning and target (τ ≥ 1).
        normalize (bool, default=True): If True, return probabilities; if False, return raw counts.
        dtype (type, default=float): dtype for the returned matrix when normalize=True.

    Returns:
        P (np.ndarray, shape (vocab_size, vocab_size)): Row-stochastic transition matrix
            (or counts if normalize=False). Rows with no outgoing observations are all zeros.
    """
    if tau < 1:
        raise ValueError("tau must be ≥ 1")
    seq = np.asarray(seq, dtype=np.int64)
    n = len(seq)
    if n <= tau:
        return np.zeros((vocab_size, vocab_size), dtype=(dtype if normalize else np.int64))

    src = seq[:-tau]   # s_{t-τ}
    dst = seq[tau:]    # s_t

    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    np.add.at(counts, (src, dst), 1)  # accumulate counts for (src -> dst)

    if not normalize:
        return counts

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = counts / row_sums
    P[row_sums.squeeze() == 0] = 0  # keep rows with no evidence as zeros
    return P.astype(dtype)



from clustering import UniformGridClusterer

class SymbolicMarkovChain:
    """
    Partition a multivariate time series into K symbols with k-means,
    then estimate a tau-lag Markov transition matrix between symbols.

    Parameters:
        K (int): Number of clusters (symbols).
        tau (float): Time lag. For each t_i, transitions are computed to the first index j
            with T_j >= T_i + tau (if such j exists).
        T (array-like, optional): Time stamps (need not be uniform; any real-valued, sortable type).
        D (array-like, optional): Multivariate observations aligned with T.
        clustering_method (str, optional): Clustering method to use. "kmeans" or "uniform".

    Attributes:
        K (int): Number of clusters (symbols).
        tau (float): Time lag. For each t_i, transitions are computed to the first index j
            with T_j >= T_i + tau (if such j exists).
        kmeans (KMeans): Trained k-means object.
        labels_ (np.ndarray, shape [N,]): Symbol of each sample (time-sorted).
        counts_ (np.ndarray, shape [K, K]): Tau-lag transition counts.
        P_ (np.ndarray, shape [K, K]): Row-stochastic transition probabilities.
        order_ (np.ndarray, shape [N,]): Indices that sort the input times.
        T_sorted_ (np.ndarray, shape [N,]): Sorted times.

    Examples:
        >>> from icicl.operators import SymbolicMarkovChain
        >>> chain = SymbolicMarkovChain(K=10, tau=1)
        >>> chain.fit(D, K, tau, T)
        >>> labels = chain.predict(D)
        >>> counts = chain.counts_
        >>> P = chain.P_
        
    """

    def fit(self, D, K, tau, T=None, clustering_method="kmeans"):
        """
        Fit the SymbolicMarkovChain to the data.
        Args:
            D (array-like): Multivariate observations aligned with T.
            K (int): Number of clusters (symbols).
            tau (float): Time lag. For each t_i, transitions are computed to the first index j
                with T_j >= T_i + tau (if such j exists).
            T (array-like, optional): Time stamps (need not be uniform; any real-valued, sortable type).
            clustering_method (str, optional): Clustering method to use. "kmeans" or "uniform".

        Returns:
            self (SymbolicMarkovChain): The fitted SymbolicMarkovChain.
        """
        if T is None:
            T = np.arange(D.shape[0])
        # Validate and sort by time
        T = np.asarray(T).astype(float).ravel()
        D = np.asarray(D)
        if T.ndim != 1:
            raise ValueError("T must be 1D.")
        if D.ndim != 2:
            raise ValueError("D must be 2D [N, d].")
        if len(T) != len(D):
            raise ValueError("T and D must have the same length.")
        if not (isinstance(K, int) and K >= 1 and K <= len(T)):
            raise ValueError("K must be an integer in [1, N].")
        if not (np.isfinite(tau) and tau > 0):
            raise ValueError("tau must be a positive finite number.")

        self.K = int(K)
        self.tau = float(tau)
        self.clustering_method = clustering_method
        self.order_ = np.argsort(T, kind="mergesort")
        T_sorted = T[self.order_]
        D_sorted = D[self.order_]
        self.T_sorted_ = T_sorted

        ## Cluster the data to define symbols
        if self.clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=self.K, n_init=10, random_state=0)
        elif self.clustering_method == "uniform":
            clusterer = UniformGridClusterer(K=self.K)
        else:
            raise ValueError(f"Invalid clustering method: {self.clustering_method}")
        clusterer.fit(D_sorted)
        labels = clusterer.labels_
        self.clusterer = clusterer
        self.labels_ = labels

        # Tau-lag transitions using searchsorted on times
        counts = np.zeros((self.K, self.K), dtype=np.int64)
        idx_after = np.searchsorted(T_sorted, T_sorted + self.tau, side="left")

        # Accumulate counts for valid pairs (i -> j)
        valid = idx_after < len(T_sorted)
        src = labels[valid]
        dst = labels[idx_after[valid]]
        # vectorized bincount for 2D table
        counts += np.bincount(src * self.K + dst, minlength=self.K * self.K).reshape(self.K, self.K)

        # Row-normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            P = np.where(row_sums > 0, counts / row_sums, 0.0)

        self.counts_ = counts
        self.P_ = P
        return self