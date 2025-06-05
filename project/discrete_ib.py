import numpy as np

class DiscreteIB:
    """
    Discrete Information Bottleneck (IB) implementation for clustering discrete data.

    Given:
      - p_m: array of shape (N,), prior probabilities for each 'meaning' (chip)
      - p_w_given_m: array of shape (N, V), empirical distribution of 'words' given each meaning
      - n_clusters: desired number of clusters (K)
      - beta: trade-off parameter, balancing compression (I(M;K)) and relevance (I(K;W))

    Produces:
      - q_k_given_m: array of shape (N, K), probability of assigning meaning m to cluster k
      - q_k: array of shape (K,), marginal distribution of clusters
      - q_w_given_k: array of shape (K, V), distribution over words for each cluster
      - q_m_given_k: array of shape (K, N), distribution over meanings for each cluster (via Bayes)
    """

    def __init__(self, p_m, p_w_given_m, n_clusters, beta):
        self.p_m = np.asarray(p_m)                            # shape (N,)
        self.p_w_given_m = np.asarray(p_w_given_m)            # shape (N, V)
        self.n_clusters = n_clusters                          # K
        self.beta = beta                                      # trade-off parameter

        # Dimensions
        self.N = self.p_m.shape[0]
        self.V = self.p_w_given_m.shape[1]
        self.K = n_clusters

        # Validate dimensions
        if self.p_w_given_m.shape[0] != self.N:
            raise ValueError("p_w_given_m must have same first dimension as p_m")

        # Initialize q(k|m) uniformly
        self.q_k_given_m = np.ones((self.N, self.K)) / self.K  # shape (N, K)

        # Initialize placeholders for other distributions
        self.q_k = np.ones(self.K) / self.K                    # shape (K,)
        self.q_w_given_k = np.zeros((self.K, self.V))          # shape (K, V)
        self.q_m_given_k = np.zeros((self.K, self.N))          # shape (K, N)

        # Small epsilon to avoid divide-by-zero
        self.eps = 1e-12

    def _update_marginals(self):
        """
        Recompute q_k and q_w_given_k based on current q_k_given_m.
        """
        # q_k = sum_m p(m) q(k|m)
        self.q_k = np.einsum('m,mk->k', self.p_m, self.q_k_given_m)  # shape (K,)

        # q_w_given_k[k, w] = sum_m p(m) q(k|m) p(w|m) / q_k[k]
        for k in range(self.K):
            # numerator: vector of length V
            numerator = np.einsum('m,mw->w', self.p_m * self.q_k_given_m[:, k], self.p_w_given_m)
            # divide by q_k[k], but avoid division by zero
            if self.q_k[k] > self.eps:
                self.q_w_given_k[k, :] = numerator / (self.q_k[k] + self.eps)
            else:
                # If q_k is zero, distribute uniformly
                self.q_w_given_k[k, :] = np.ones(self.V) / self.V

        # Ensure numerical stability: normalize q_w_given_k
        self.q_w_given_k = np.maximum(self.q_w_given_k, self.eps)
        self.q_w_given_k /= self.q_w_given_k.sum(axis=1, keepdims=True)

    def _compute_kl_divergences(self):
        """
        Compute D_{KL}( p(w|m) || q(w|k) ) for all m, k.
        Returns a (N, K) array of KL divergences.
        """
        # Clip distributions to avoid log(0)
        p = np.clip(self.p_w_given_m, self.eps, 1.0)      # shape (N, V)
        q = np.clip(self.q_w_given_k, self.eps, 1.0)      # shape (K, V)

        # Compute logs
        log_p = np.log(p)                                 # shape (N, V)
        log_q = np.log(q)                                 # shape (K, V)

        # D[m, k] = sum_w p[m,w] * (log_p[m,w] - log_q[k, w])
        D = np.zeros((self.N, self.K))
        for k in range(self.K):
            D[:, k] = np.sum(p * (log_p - log_q[k, :]), axis=1)
        return D  # shape (N, K)

    def train(self, max_iters=100, tol=1e-6, verbose=False):
        """
        Run the IB algorithm to update q(k|m), q(w|k), q(k), q(m|k).
        Terminates when q(k|m) changes by less than tol (in max absolute diff) or max_iters is reached.
        """
        for iteration in range(max_iters):
            q_old = self.q_k_given_m.copy()

            # 1. Update q_k and q(w|k) based on current q(k|m)
            self._update_marginals()

            # 2. Compute KL divergences D_{m,k}
            D = self._compute_kl_divergences()  # shape (N, K)

            # 3. Update q(k|m) ∝ q(k) * exp(-beta * D_{m,k})
            #    Then normalize over k for each m.
            # Use log-space for stability: log_q_new = log(q_k) - beta * D
            log_qk = np.log(np.clip(self.q_k, self.eps, None))  # shape (K,)
            exponent = log_qk[np.newaxis, :] - self.beta * D    # shape (N, K)

            # To avoid overflow, subtract max per row
            row_max = np.max(exponent, axis=1, keepdims=True)   # shape (N, 1)
            exp_shifted = np.exp(exponent - row_max)            # shape (N, K)
            q_new = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

            self.q_k_given_m = q_new

            # Check convergence
            max_diff = np.max(np.abs(self.q_k_given_m - q_old))
            if verbose:
                print(f"Iteration {iteration+1}, max q_k_given_m change = {max_diff:.2e}")
            if max_diff < tol:
                if verbose:
                    print(f"Converged in {iteration+1} iterations.")
                break

        # After convergence or max_iters, compute q(m|k)
        self.q_m_given_k = np.zeros((self.K, self.N))
        for k in range(self.K):
            # p(m,k) = p(m) * q(k|m), so p(m|k) = p(m) q(k|m) / q(k)
            if self.q_k[k] > self.eps:
                self.q_m_given_k[k, :] = (self.p_m * self.q_k_given_m[:, k]) / (self.q_k[k] + self.eps)
            else:
                self.q_m_given_k[k, :] = np.ones(self.N) / self.N  # fallback uniform

        # Ensure normalization
        self.q_m_given_k /= np.sum(self.q_m_given_k, axis=1, keepdims=True)