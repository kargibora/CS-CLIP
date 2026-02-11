import torch
import torch.nn.functional as F

class CCA:
    def __init__(self, k=None, eps=1e-12):
        """
        k: number of top components to keep; if None, uses min(Dx, Dy)
        eps: regularization on covariances
        """
        self.k = k
        self.eps = eps
        self.Wx = None
        self.Wy = None
        self.rhos = None

    def fit(self, X, Y):
        """
        X: [N, Dx] tensor
        Y: [N, Dy] tensor
        """
        # center
        Xc = X - X.mean(0, keepdim=True)
        Yc = Y - Y.mean(0, keepdim=True)
        N = Xc.shape[0]

        # covariances
        Cxx = (Xc.T @ Xc) / (N - 1) \
              + self.eps * torch.eye(Xc.size(1), device=X.device)
        Cyy = (Yc.T @ Yc) / (N - 1) \
              + self.eps * torch.eye(Yc.size(1), device=Y.device)
        Cxy = (Xc.T @ Yc) / (N - 1)

        # compute inverse square roots
        def inv_sqrt(mat):
            # symmetric eigen-decomp
            D, V = torch.linalg.eigh(mat)
            D = (D + self.eps).clamp(min=self.eps).rsqrt()
            return (V * D.unsqueeze(0)) @ V.T

        Cxx_inv_sqrt = inv_sqrt(Cxx)
        Cyy_inv_sqrt = inv_sqrt(Cyy)

        # form T and do skinny SVD
        T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
        U, S, Vt = torch.linalg.svd(T, full_matrices=False)
        # S = [rho_1, rho_2, ...]
        D = Xc.size(1)
        E = Yc.size(1)
        k = self.k or min(D, E)
        U, S, Vt = U[:, :k], S[:k], Vt[:k, :]

        # store projections and correlations
        self.Wx = Cxx_inv_sqrt @ U       # [Dx, k]
        self.Wy = Cyy_inv_sqrt @ Vt.T    # [Dy, k]
        self.rhos = S                     # [k]

        return S.mean().item(), S

    def project(self, X):
        """
        Project new embeddings X ([N, Dx]) into the CCA space [N, k].
        Note: assumes you’ve already called fit().
        """
        assert self.Wx is not None, "You must call fit() first."
        Xc = X - X.mean(0, keepdim=True)
        return Xc @ self.Wx

    def similarity(self, X, Y, metric='cosine'):
        """
        Compute pairwise similarity between projected X, Y in CCA space.
        metric: 'cosine' or 'dot'
        Returns: [N, M] matrix
        """
        Xp = self.project(X)
        Yp = self.project(Y)
        if metric == 'cosine':
            Xn = F.normalize(Xp, dim=-1)
            Yn = F.normalize(Yp, dim=-1)
            return Xn @ Yn.T
        elif metric == 'dot':
            return Xp @ Yp.T
        else:
            raise ValueError(f"Unknown metric {metric!r}")

class CKA:
    def __init__(self, kernel: str = 'linear', gamma: float = None):
        """
        kernel: 'linear' or 'rbf'
        gamma: bandwidth for RBF; if None, uses 1 / feature_dim
        """
        assert kernel in ('linear', 'rbf'), "kernel must be 'linear' or 'rbf'"
        self.kernel = kernel
        self.gamma = gamma

    def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
        """Center an [N,N] Gram matrix K."""
        N = K.size(0)
        H = torch.eye(N, device=K.device) - torch.ones(N, N, device=K.device) / N
        return H @ K @ H

    def _linear_gram(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the linear Gram matrix X X^T."""
        return X @ X.T

    def _rbf_gram(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the RBF Gram matrix with bandwidth gamma."""
        # pairwise squared distances
        D2 = torch.cdist(X, X, p=2).pow(2)
        gamma = self.gamma or (1.0 / X.size(1))
        return torch.exp(-gamma * D2)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Compute CKA similarity between X and Y.
        X: [N, d_x], Y: [N, d_y]
        Returns:
            cka: float in [0,1]
        """
        # optionally center features per-dimension (not strictly required)
        Xc = X - X.mean(0, keepdim=True)
        Yc = Y - Y.mean(0, keepdim=True)

        # compute Gram matrices
        if self.kernel == 'linear':
            K = self._linear_gram(Xc)
            L = self._linear_gram(Yc)
        else:
            K = self._rbf_gram(Xc)
            L = self._rbf_gram(Yc)

        # center them
        Kc = self._center_gram(K)
        Lc = self._center_gram(L)

        # HSIC estimates
        # note: dividing by (N-1)^2 makes HSIC unbiased for linear kernel
        N = X.size(0)
        norm = 1.0 / (N - 1)**2
        hsic_xy = norm * (Kc * Lc).sum()
        hsic_xx = norm * (Kc * Kc).sum()
        hsic_yy = norm * (Lc * Lc).sum()

        # CKA score
        cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-12)
        return cka.item()