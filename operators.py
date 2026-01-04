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


def leading_transients(P, m=3, tol=1e-10, max_iter=20_000, seed=None):
    """
    Sequentially compute the longest-lived transient *left* modes of a Markov chain
    using simple deflation: at each stage subtract 1*u^T so that the found left
    eigenvector u^T is annihilated in subsequent searches.

    Args:
        P (array-like): Row-stochastic transition matrix (N, N).
        m (int): Number of transient modes to compute (excludes the stationary mode).
        tol (float): L2 tolerance for convergence of each mode.
        max_iter (int): Maximum iterations per mode.
        seed (int | None): RNG seed.

    Returns:
        evals (ndarray): Length-m array of eigenvalue estimates |λ| in non-increasing order.
        evecs (ndarray): (m, N) array; evecs[j] is the j-th transient left mode (L2-normalized, sum≈0).
        pi (ndarray): Stationary distribution (N,).
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if P.shape != (n, n): raise ValueError("P must be square")
    if not (1 <= m <= n - 1): raise ValueError("m must be in [1, N-1]")

    # 1) Fix dangling and get π
    pi = invariant_distribution(P, fix_dangling=True, seed=seed)

    # 2) Build deflated operator that kills π; then iteratively deflate found transients
    Pdef = deflated_operator(P, pi)  # P - 1*pi^T
    Pt = P.T

    rng = np.random.default_rng(seed)
    evals, modes = [], []

    ones = np.ones(n)

    def project_stable(v):
        # Remove any leakage back into π and enforce zero-sum (left transient modes satisfy u^T 1 = 0)
        v = v - (v @ pi) * pi
        v = v - (v.sum() / n) * ones
        return v

    for _ in range(m):
        # Initialize in the zero-sum, π-orthogonal subspace
        v = rng.normal(size=n)
        v = project_stable(v)
        nv = np.linalg.norm(v)
        if nv == 0 or not np.isfinite(nv): v = rng.normal(size=n); v = project_stable(v); nv = np.linalg.norm(v)
        v /= nv

        # Power iteration on (Pdef)^T to find current dominant transient left mode
        Pt_def = Pdef.T
        prev = v.copy()
        for _it in range(max_iter):
            w = Pt_def @ v
            nw = np.linalg.norm(w)
            if nw == 0 or not np.isfinite(nw): raise FloatingPointError("Deflated power iteration became singular")
            v = w / nw
            v = project_stable(v)
            nv = np.linalg.norm(v)
            if nv == 0 or not np.isfinite(nv): raise FloatingPointError("Projection produced degenerate vector")
            v /= nv
            if np.linalg.norm(v - prev) < tol: break
            prev = v

        # Stabilize sign (deterministic)
        imax = np.argmax(np.abs(v))
        if v[imax] < 0: v = -v

        # Eigenvalue estimate from original P (Rayleigh quotient for left vector)
        lam = float(v @ (Pt @ v))

        evals.append(lam)
        modes.append(v.copy())

        # 3) Deflate this mode so the next search is self-consistent
        Pdef = deflated_operator(Pdef, v)  # cumulative: P - 1*pi^T - sum 1*u_i^T

    return np.asarray(evals), np.vstack(modes), pi



def deflated_operator(P, pi):
    """
    A deflated operator that removes a trivial mode from a row-stochastic matrix

    Args:
        P: (N, N) array-like, row-stochastic ideally
        pi: (N,) np.ndarray, stationary distribution

    Returns:
        Ptil: (N, N) sparse.csr_matrix, deflated operator
    """
    n = P.shape[0]
    return P - np.outer(np.ones(n), pi)


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
    # if tau < 1:
    #     raise ValueError("tau must be ≥ 1")
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


def transitions_from_time_labels(T, labels, K, tau_time, normalize=True, dtype=float):
    """
    Compute τ-time transitions using explicit timestamps by mapping each index i
    to the first index j with T[j] >= T[i] + tau_time.

    Args:
        T (array-like): Monotonic time stamps (1D, real-valued).
        labels (array-like): Discrete state labels aligned with T, each in [0, K).
        K (int): Number of states.
        tau_time (float): Positive lag in the same units as T.
        normalize (bool, default=True): If True, return probabilities; else raw counts.
        dtype (type, default=float): dtype for probabilities when normalize=True.

    Returns:
        (np.ndarray, shape (K, K)): Transition matrix (row-stochastic if normalize=True).
    """
    T = np.asarray(T, dtype=float).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if T.ndim != 1:
        raise ValueError("T must be 1D.")
    if labels.ndim != 1:
        raise ValueError("labels must be 1D.")
    if len(T) != len(labels):
        raise ValueError("T and labels must have the same length.")
    if not (np.isfinite(tau_time) and tau_time > 0):
        raise ValueError("tau_time must be positive and finite.")
    if not (isinstance(K, int) and K >= 1):
        raise ValueError("K must be a positive integer.")

    counts = np.zeros((K, K), dtype=np.int64)
    idx_after = np.searchsorted(T, T + tau_time, side="left")
    valid = idx_after < len(T)
    src = labels[valid]
    dst = labels[idx_after[valid]]
    counts += np.bincount(src * K + dst, minlength=K * K).reshape(K, K)

    if not normalize:
        return counts
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(row_sums > 0, counts / row_sums, 0.0)
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

    def fit(self, X, K, tau, T=None, clustering_method="kmeans"):
        """
        Fit a SymbolicMarkovChain to a multivariate time series.

        Args:
            X (array-like): Multivariate observations aligned with T.
            K (int): Number of clusters (symbols).
            tau (float): Time lag. For each t_i, transitions are computed to the first index j
                with T_j >= T_i + tau (if such j exists).
            T (array-like, optional): Time stamps (need not be uniform; any real-valued, sortable type).
            clustering_method (str, optional): Clustering method to use. "kmeans" or "uniform".

        Returns:
            self (SymbolicMarkovChain): The fitted SymbolicMarkovChain.
        """
        if T is not None:
            # Validate and sort by time
            T = np.asarray(T).astype(float).ravel()
            if T.ndim != 1:
                raise ValueError("T must be 1D.")
            if len(T) != len(X):
                raise ValueError("T and D must have the same length.")
            if not (isinstance(K, int) and K >= 1 and K <= len(T)):
                raise ValueError("K must be an integer in [1, N].")
            order = np.argsort(T)
            T, X = T[order], X[order]
        
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, d].")
        if not (np.isfinite(tau) and tau > 0):
            raise ValueError("tau must be a positive finite number.")

        self.K = int(K)
        self.tau = float(tau)
        self.clustering_method = clustering_method


        ## Cluster the data to define symbols
        if self.clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=self.K, n_init=10, random_state=0)
        elif self.clustering_method == "uniform":
            clusterer = UniformGridClusterer(K=self.K)
        else:
            raise ValueError(f"Invalid clustering method: {self.clustering_method}")
        clusterer.fit(X)
        labels = np.asarray(clusterer.labels_)
        self.clusterer = clusterer
        self.labels_ = labels

        ## If times are provided, use them to compute transitions among symbols
        if T is not None:
            P = transitions_from_time_labels(T, labels, self.K, self.tau, normalize=True, dtype=float)
        else:
            ## Otherwise, compute transitions among observations directly
            P = transition_matrix(labels, self.K, tau=int(self.tau), normalize=True, dtype=float)

        self.P_ = P
        return self

    def predict(self, X):
        """
        Predict the labels for a new set of observations.
        """
        return self.clusterer.predict(X)

    def fit_predict(self, X, *args, **kwargs):
        """
        Fit the SymbolicMarkovChain and predict the labels for a new set of observations.
        """
        self.fit(X, *args, **kwargs)
        return self.predict(X)
