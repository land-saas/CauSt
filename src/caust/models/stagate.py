"""STAGATE backbone: a graph attention autoencoder, in plain PyTorch.

A faithful re-implementation of STAGATE (Dong & Zhang, Nature Communications
2022) as used by the CauST proposal: an encoder ``G -> 512 -> 30`` and a
weight-tied decoder ``30 -> 512 -> G`` built from single-head graph attention
layers, trained full-batch to reconstruct the (log-normalized) expression of
every spot from its spatial neighbourhood. The 30-d bottleneck is the spatial
embedding that CauST perturbs.

Details kept exactly as in the reference implementation because they change
the learned embedding:

- attention logits go through a **sigmoid** (not LeakyReLU) before the
  per-target softmax, and self-loops are added to the spatial graph;
- the two ``attention=False`` layers are plain linear projections;
- the decoder's weights are re-pointed at transposed *views* of the encoder
  weights on every forward pass (the original's ``.data`` assignment): no
  autograd through the tie, but the optimizer's in-place updates of the decoder
  parameters land in the shared encoder storage, so the encoder is effectively
  updated twice per step. The first decoder layer reuses the encoder's
  attention coefficients.

Only ``torch`` is required -- no torch_geometric.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from ..graph import CONN_KEY, build_spatial_graph
from .base import BaseSpatialModel

try:  # torch is an optional extra: pip install "caust[stagate]"
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without the extra
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "the STAGATE backbone needs PyTorch; install it with "
            'pip install "caust[stagate]" (or uv sync --extra stagate)'
        )


def resolve_device(device: str = "auto") -> Any:
    """Map ``'auto'`` to CUDA, then Apple MPS, then CPU."""
    _require_torch()
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _segment_softmax(logits: Any, index: Any, n_nodes: int) -> Any:
    """Softmax of ``logits`` grouped by target node ``index`` (edge-wise)."""
    # Subtract the per-group max for numerical stability (the max is a
    # constant w.r.t. the gradient of a softmax, so detaching it is exact).
    group_max = torch.full((n_nodes,), -float("inf"), device=logits.device)
    group_max = group_max.scatter_reduce(
        0, index, logits.detach(), reduce="amax", include_self=True
    )
    exp = torch.exp(logits - group_max[index])
    denom = torch.zeros(n_nodes, device=logits.device).index_add(0, index, exp)
    return exp / (denom[index] + 1e-16)


if torch is not None:

    class GraphAttention(nn.Module):
        """Single-head graph attention with optionally tied attention."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
            # Shaped (1, heads=1, out) exactly like the reference so xavier's
            # fan calculation -- and therefore the init scale of the attention
            # logits -- matches; a 2-D (1, out) shape would be sqrt(2) larger
            # and saturate the sigmoid early in training.
            self.att_src = nn.Parameter(torch.empty(1, 1, out_dim))
            self.att_dst = nn.Parameter(torch.empty(1, 1, out_dim))
            nn.init.xavier_normal_(self.weight, gain=1.414)
            nn.init.xavier_normal_(self.att_src, gain=1.414)
            nn.init.xavier_normal_(self.att_dst, gain=1.414)
            self.attention: tuple[Any, Any] | None = None

        def project(self, x: Any, weight: Any | None = None) -> Any:
            return x @ (self.weight if weight is None else weight)

        def forward(
            self,
            x: Any,
            edge_index: Any,
            *,
            weight: Any | None = None,
            tied: tuple[Any, Any] | None = None,
        ) -> Any:
            h = self.project(x, weight)
            if tied is None:
                alpha_src = (h * self.att_src[0]).sum(-1)
                alpha_dst = (h * self.att_dst[0]).sum(-1)
                self.attention = (alpha_src, alpha_dst)
            else:
                alpha_src, alpha_dst = tied
            src, dst = edge_index[0], edge_index[1]
            logits = torch.sigmoid(alpha_src[src] + alpha_dst[dst])
            coef = _segment_softmax(logits, dst, h.shape[0])
            out = torch.zeros_like(h).index_add(0, dst, h[src] * coef.unsqueeze(-1))
            return out

    class STAGATENet(nn.Module):
        """Encoder ``G -> hidden -> latent`` with a weight-tied decoder."""

        def __init__(self, in_dim: int, hidden_dim: int = 512, latent_dim: int = 30):
            super().__init__()
            self.conv1 = GraphAttention(in_dim, hidden_dim)
            self.conv2 = GraphAttention(hidden_dim, latent_dim)  # linear use
            self.dec1 = GraphAttention(latent_dim, hidden_dim)  # tied use
            self.dec2 = GraphAttention(hidden_dim, in_dim)  # linear use

        def encode(self, x: Any, edge_index: Any) -> Any:
            h1 = torch.nn.functional.elu(self.conv1(x, edge_index))
            return self.conv2.project(h1)

        def forward(self, x: Any, edge_index: Any) -> tuple[Any, Any]:
            h1 = torch.nn.functional.elu(self.conv1(x, edge_index))
            z = self.conv2.project(h1)
            # Weight tying exactly as the reference does it: the decoder
            # parameters are re-pointed at *views* of the transposed encoder
            # weights every forward pass. Autograd does not flow through the
            # assignment, but the optimizer updates the decoder parameters in
            # place -- and because they alias the encoder storage, the decoder
            # gradients reach the encoder that way (with their own Adam
            # moments). Reproducing this is what makes the loss curve match.
            self.dec1.weight.data = self.conv2.weight.data.t()
            self.dec2.weight.data = self.conv1.weight.data.t()
            h3 = torch.nn.functional.elu(
                self.dec1(z, edge_index, tied=self.conv1.attention)
            )
            recon = self.dec2.project(h3)
            return z, recon


def _edge_index_with_self_loops(adj: sp.spmatrix, device: Any) -> Any:
    coo = (sp.csr_matrix(adj) + sp.eye(adj.shape[0], format="csr")).tocoo()
    idx = np.vstack([coo.row, coo.col]).astype(np.int64)
    return torch.as_tensor(idx, device=device)


def _dense32(X: Any) -> np.ndarray:
    return np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float32)


class STAGATEModel(BaseSpatialModel):
    """STAGATE as a frozen CauST backbone.

    Parameters
    ----------
    hidden_dim, latent_dim
        Encoder widths (``512`` and ``30`` in the paper).
    n_epochs, lr, weight_decay, grad_clip
        Training schedule of the reference implementation.
    n_neighbors
        Spatial kNN graph degree; 6 matches the Visium hexagonal lattice (and
        the reference's ``rad_cutoff`` radius graph on regular arrays).
    device
        ``'auto'`` (CUDA > MPS > CPU) or an explicit torch device string.
    random_state
        Seeds torch before weight initialisation.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        latent_dim: int = 30,
        n_epochs: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 5.0,
        n_neighbors: int = 6,
        device: str = "auto",
        random_state: int = 0,
        verbose: bool = False,
    ):
        _require_torch()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.n_neighbors = n_neighbors
        self.device = resolve_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.net: Any = None
        self.loss_history: list[float] = []
        self._X: np.ndarray | None = None
        self._edge_index: Any = None
        self._Z: np.ndarray | None = None

    # -- training ---------------------------------------------------------
    def fit(self, adata: AnnData) -> STAGATEModel:
        if CONN_KEY not in adata.obsp:
            build_spatial_graph(adata, n_neighbors=self.n_neighbors)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = _dense32(adata.X)
        x = torch.as_tensor(X, device=self.device)
        edge_index = _edge_index_with_self_loops(adata.obsp[CONN_KEY], self.device)
        net = STAGATENet(X.shape[1], self.hidden_dim, self.latent_dim).to(self.device)
        opt = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.loss_history = []
        net.train()
        for epoch in range(1, self.n_epochs + 1):
            opt.zero_grad()
            _, recon = net(x, edge_index)
            loss = torch.nn.functional.mse_loss(x, recon)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
            opt.step()
            self.loss_history.append(loss.item())
            if self.verbose and (epoch % max(1, self.n_epochs // 10) == 0):
                print(f"  [stagate] epoch {epoch}/{self.n_epochs} loss {loss:.4f}")
        net.eval()
        for p in net.parameters():  # frozen from here on
            p.requires_grad_(False)
        self.net = net
        self._X = X
        self._edge_index = edge_index
        self._Z = None
        return self

    def _check_fitted(self) -> None:
        if self.net is None:
            raise ValueError("model is not fitted; call fit() first.")

    # -- frozen inference -------------------------------------------------
    def _embed(self, X: np.ndarray, edge_index: Any) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
            z = self.net.encode(x, edge_index)
        out: np.ndarray = z.cpu().numpy().astype(np.float64)
        return out

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Frozen forward pass on the *fitted* slice's graph."""
        self._check_fitted()
        return self._embed(X, self._edge_index)

    def get_embedding(self, adata: AnnData | None = None) -> np.ndarray:
        """Embed the fitted slice, or any other slice with its own graph.

        Passing another ``AnnData`` (same genes, own coordinates) is the
        zero-shot transfer the benchmark measures: the model is not retrained.
        """
        self._check_fitted()
        if adata is None:
            if self._Z is None:
                self._Z = self.forward(self._get_expression())
            return self._Z
        if CONN_KEY not in adata.obsp:
            build_spatial_graph(adata, n_neighbors=self.n_neighbors)
        edge_index = _edge_index_with_self_loops(adata.obsp[CONN_KEY], self.device)
        return self._embed(_dense32(adata.X), edge_index)

    def _get_expression(self) -> np.ndarray:
        self._check_fitted()
        assert self._X is not None
        return self._X

    def get_knockout_embedding(self, gene_idx: int) -> np.ndarray:
        """Zero one gene column on-device (avoids re-copying the matrix)."""
        self._check_fitted()
        with torch.no_grad():
            x = torch.as_tensor(self._X, device=self.device).clone()
            x[:, gene_idx] = 0.0
            z = self.net.encode(x, self._edge_index)
        out: np.ndarray = z.cpu().numpy().astype(np.float64)
        return out
