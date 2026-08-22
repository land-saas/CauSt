"""GraphST backbone (Long et al., Nature Communications 2023) in plain PyTorch.

A port of GraphST's spatial-domain encoder: a two-layer graph convolution
``h = Â (Â X W1) W2`` trained to reconstruct the scaled expression matrix,
regularised by a Deep-Graph-Infomax-style contrastive term that tells the
true neighbourhood readout from one computed on row-shuffled features
(bilinear discriminator, BCE loss). Settings follow the reference: 64-d latent,
600 epochs, Adam 1e-3 without weight decay, ``alpha = 10`` on reconstruction,
``beta = 1`` on the contrastive term, a 3-nearest-neighbour graph with
``Â = D^-1/2 A D^-1/2 + I``, and per-gene scaling without centring clipped at
10. As in GraphST's own clustering step, the embedding handed downstream is
the top principal components of the reconstructed features, with the PCA
frozen on the training slice so other slices are embedded zero-shot.

Vendored rather than imported: the PyPI package declares none of its real
dependencies and relies on deprecated scipy import paths. Credit: GraphST,
MIT licence, https://github.com/JinmiaoChenLab/GraphST.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .base import BaseSpatialModel

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]

warnings.filterwarnings(
    "ignore", message=".*does not have a deterministic implementation.*"
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "the GraphST backbone needs PyTorch; install it with "
            'pip install "caust[stagate]"'
        )


def _dense(X: Any) -> np.ndarray:
    return np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float32)


def knn_graphs(coords: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """(directed kNN mask, symmetric binary adjacency), both dense ``(N, N)``."""
    n = coords.shape[0]
    nn_ = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n)).fit(coords)
    _, idx = nn_.kneighbors(coords)
    directed = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    directed[rows, idx[:, 1:].ravel()] = 1.0
    adj = np.minimum(directed + directed.T, 1.0)
    return directed, adj


def normalized_adj(adj: np.ndarray) -> np.ndarray:
    """GraphST's ``D^-1/2 A D^-1/2 + I`` (identity added after normalising)."""
    deg = adj.sum(1)
    d = np.where(deg > 0, deg**-0.5, 0.0)
    out: np.ndarray = (
        d[:, None] * adj * d[None, :] + np.eye(adj.shape[0], dtype=np.float32)
    ).astype(np.float32)
    return out


if torch is not None:

    class GraphSTNet(nn.Module):
        def __init__(self, in_dim: int, latent_dim: int):
            super().__init__()
            self.weight1 = nn.Parameter(torch.empty(in_dim, latent_dim))
            self.weight2 = nn.Parameter(torch.empty(latent_dim, in_dim))
            nn.init.xavier_uniform_(self.weight1)
            nn.init.xavier_uniform_(self.weight2)
            self.disc = nn.Bilinear(latent_dim, latent_dim, 1)
            nn.init.xavier_uniform_(self.disc.weight)
            nn.init.zeros_(self.disc.bias)

        def encode(self, x: Any, adj: Any) -> Any:
            return adj @ (x @ self.weight1)

        def reconstruct(self, z: Any, adj: Any) -> Any:
            return adj @ (z @ self.weight2)

        def readout(self, emb: Any, mask: Any) -> Any:
            g = (mask @ emb) / mask.sum(1, keepdim=True)
            return torch.sigmoid(torch.nn.functional.normalize(g, p=2, dim=1))

        def forward(
            self, x: Any, x_shuffled: Any, adj: Any, mask: Any
        ) -> tuple[Any, Any, Any]:
            z = self.encode(x, adj)
            h = self.reconstruct(z, adj)
            emb, emb_a = torch.relu(z), torch.relu(self.encode(x_shuffled, adj))
            g, g_a = self.readout(emb, mask), self.readout(emb_a, mask)
            ret = torch.cat([self.disc(emb, g), self.disc(emb_a, g)], 1)
            ret_a = torch.cat([self.disc(emb_a, g_a), self.disc(emb, g_a)], 1)
            return h, ret, ret_a


class GraphSTModel(BaseSpatialModel):
    """GraphST as a frozen CauST backbone (see module docstring)."""

    def __init__(
        self,
        latent_dim: int = 64,
        n_epochs: int = 600,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        alpha: float = 10.0,
        beta: float = 1.0,
        n_neighbors: int = 3,
        n_pcs: int = 20,
        scale_max: float = 10.0,
        device: str = "auto",
        random_state: int = 41,
        verbose: bool = False,
    ):
        _require_torch()
        from .stagate import resolve_device

        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.beta = beta
        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs
        self.scale_max = scale_max
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.net: Any = None
        self.pca: PCA | None = None
        self.loss_history: list[float] = []
        self._X: np.ndarray | None = None  # raw (unscaled) training matrix
        self._X_scaled: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._adj: Any = None
        self._Z: np.ndarray | None = None

    # -- preprocessing ----------------------------------------------------
    def _scale(self, X: np.ndarray, std: np.ndarray) -> np.ndarray:
        scaled: np.ndarray = np.minimum(X / std, self.scale_max).astype(np.float32)
        return scaled

    def _graph(self, adata: AnnData) -> tuple[Any, Any]:
        directed, adj = knn_graphs(
            np.asarray(adata.obsm["spatial"], dtype=float), self.n_neighbors
        )
        mask = directed + np.eye(adj.shape[0], dtype=np.float32)
        return (
            torch.as_tensor(normalized_adj(adj), device=self.device),
            torch.as_tensor(mask, device=self.device),
        )

    # -- training ---------------------------------------------------------
    def fit(self, adata: AnnData) -> GraphSTModel:
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        raw = _dense(adata.X)
        std = raw.std(axis=0)
        std[std == 0] = 1.0
        self._std = std
        X = self._scale(raw, std)
        x = torch.as_tensor(X, device=self.device)
        adj, mask = self._graph(adata)
        n = X.shape[0]
        label = torch.cat([torch.ones(n, 1), torch.zeros(n, 1)], 1).to(self.device)
        net = GraphSTNet(X.shape[1], self.latent_dim).to(self.device)
        opt = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        bce = nn.BCEWithLogitsLoss()
        self.loss_history = []
        net.train()
        for epoch in range(1, self.n_epochs + 1):
            perm = torch.as_tensor(rng.permutation(n), device=self.device)
            h, ret, ret_a = net(x, x[perm], adj, mask)
            loss = self.alpha * torch.nn.functional.mse_loss(x, h) + self.beta * (
                bce(ret, label) + bce(ret_a, label)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.loss_history.append(loss.item())
            if self.verbose and epoch % max(1, self.n_epochs // 10) == 0:
                print(
                    f"  [graphst] epoch {epoch}/{self.n_epochs} loss {loss.item():.4f}"
                )
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self._X = raw
        self._X_scaled = X
        self._adj = adj
        with torch.no_grad():
            h = net.reconstruct(net.encode(x, adj), adj).cpu().numpy()
        self.pca = PCA(
            n_components=min(self.n_pcs, min(h.shape) - 1),
            random_state=self.random_state,
        ).fit(h)
        self._Z = None
        return self

    def _check_fitted(self) -> None:
        if self.net is None or self.pca is None:
            raise ValueError("model is not fitted; call fit() first.")

    # -- frozen inference -------------------------------------------------
    def _reconstruct(self, X_scaled: np.ndarray, adj: Any) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(X_scaled, device=self.device)
            h = self.net.reconstruct(self.net.encode(x, adj), adj)
        out: np.ndarray = h.cpu().numpy()
        return out

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Embed an (unscaled) expression matrix on the fitted slice's graph."""
        self._check_fitted()
        assert self._std is not None and self.pca is not None
        h = self._reconstruct(
            self._scale(np.asarray(X, dtype=np.float32), self._std), self._adj
        )
        return np.asarray(self.pca.transform(h), dtype=np.float64)

    def get_embedding(self, adata: AnnData | None = None) -> np.ndarray:
        self._check_fitted()
        assert self._std is not None and self.pca is not None
        if adata is None:
            if self._Z is None:
                assert self._X_scaled is not None
                self._Z = np.asarray(
                    self.pca.transform(self._reconstruct(self._X_scaled, self._adj)),
                    dtype=np.float64,
                )
            return self._Z
        adj, _ = self._graph(adata)
        h = self._reconstruct(self._scale(_dense(adata.X), self._std), adj)
        return np.asarray(self.pca.transform(h), dtype=np.float64)

    def _get_expression(self) -> np.ndarray:
        """The raw training matrix (``forward`` applies the frozen scaling)."""
        self._check_fitted()
        assert self._X is not None
        return self._X

    def get_knockout_embedding(self, gene_idx: int, mode: str = "zero") -> np.ndarray:
        """Rank-1 knockout: the model is linear in the scaled features."""
        self._check_fitted()
        if mode not in self.KNOCKOUT_MODES:
            raise ValueError(f"mode must be one of {self.KNOCKOUT_MODES}, got {mode!r}")
        assert self._X is not None and self._X_scaled is not None
        assert self._std is not None and self.pca is not None
        column = self._X_scaled[:, gene_idx : gene_idx + 1]
        if mode == "mean":
            # The generic path replaces the raw gene by its raw mean and then
            # scales, so the scaled column moves to scale(raw mean).
            raw_mean = float(self._X[:, gene_idx].mean())
            column = column - min(raw_mean / self._std[gene_idx], self.scale_max)
        with torch.no_grad():
            col = torch.as_tensor(column, device=self.device)
            row = self.net.weight1[gene_idx : gene_idx + 1, :] @ self.net.weight2
            shift = self._adj @ (self._adj @ (col @ row))
        delta = shift.cpu().numpy() @ self.pca.components_.T
        out: np.ndarray = self.get_embedding() - delta
        return out
