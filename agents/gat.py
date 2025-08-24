import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):
    """A minimal single-head GAT layer.

    Keeps original behavior (skip computation for isolated nodes) with cleaned indentation.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W, gain=1.414)

        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    # --- internal helpers ---
    def _edge_scores(self, Wh: torch.Tensor) -> torch.Tensor:
        N = Wh.size(0)
        Wh1 = Wh.unsqueeze(1).expand(-1, N, -1)
        Wh2 = Wh.unsqueeze(0).expand(N, -1, -1)
        a_input = torch.cat([Wh1, Wh2], dim=-1)  # (N,N,2*C)
        scores = torch.matmul(a_input, self.a).squeeze(-1)  # (N,N)
        return self.leakyrelu(scores)

    def _masked_softmax(self, e: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        e_masked = e.masked_fill(adj == 0, float('-inf'))
        return F.softmax(e_masked, dim=1)

    # --- forward ---
    def forward(self, h: torch.Tensor, adj: torch.Tensor):  # noqa: D401
        """Compute attention-weighted features.

        Args:
            h: (N, in_features)
            adj: (N, N) adjacency with self-loops (1 where edge exists)
        Returns:
            h_prime: (N, out_features)
            attn: (N, N) attention weights (rows sum to 1 over neighbors kept)
        """
        N = h.size(0)
        Wh = torch.mm(h, self.W)  # (N, out_features)

        with torch.no_grad():
            is_lone = (adj.sum(1) == 1)  # only self-loop
            keep = ~is_lone

        final_attn = torch.zeros_like(adj)
        if keep.any():
            Wh_k = Wh[keep]
            adj_k = adj[keep][:, keep]
            if Wh_k.size(0) > 0:
                e_k = self._edge_scores(Wh_k)
                attn_k = self._masked_softmax(e_k, adj_k)
                attn_drop = F.dropout(attn_k, self.dropout, training=self.training)
                h_prime = Wh.clone()
                h_prime[keep] = torch.matmul(attn_drop, Wh_k)
                idx_keep = torch.where(keep)[0]
                final_attn[idx_keep[:, None], idx_keep] = attn_k
                idx_lone = torch.where(is_lone)[0]
                final_attn[idx_lone, idx_lone] = 1.0
            else:
                h_prime = Wh
                final_attn = torch.eye(N, device=h.device, dtype=h.dtype)
        else:
            e = self._edge_scores(Wh)
            attn = self._masked_softmax(e, adj)
            attn_drop = F.dropout(attn, self.dropout, training=self.training)
            h_prime = torch.matmul(attn_drop, Wh)
            final_attn = attn

        return h_prime, final_attn


class MultiHeadGATv2Dense(nn.Module):
    """Multi-head GATv2 (dense adjacency) with concat output.

    - Interface compatible with GraphAttention: forward(h, adj) -> (h_prime, attn)
    - No residual/LayerNorm/FFN here (kept for later steps). Dropout applied to attention and features.
    - Returns averaged attention over heads for convenient logging.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 6,
        concat: bool = True,
        negative_slope: float = 0.2,
        feat_drop: float = 0.2,
        attn_drop: float = 0.2,
        bias: bool = True,
        share_weights: bool = False,
    ):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.head_dim = out_dim // heads
        self.negative_slope = negative_slope
        self.share_weights = share_weights

        # weights per head: (H, in_dim, head_dim)
        self.W_src = nn.Parameter(torch.empty(heads, in_dim, self.head_dim))
        if share_weights:
            self.W_dst = self.W_src
        else:
            self.W_dst = nn.Parameter(torch.empty(heads, in_dim, self.head_dim))

        # attention vector per head: (H, head_dim)
        self.a = nn.Parameter(torch.empty(heads, self.head_dim))

        # dropouts
        self.feat_dropout = nn.Dropout(feat_drop)
        self.attn_dropout = nn.Dropout(attn_drop)

        self.leakyrelu = nn.LeakyReLU(negative_slope)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

        # init
        nn.init.xavier_uniform_(self.W_src, gain=1.414)
        if not share_weights:
            nn.init.xavier_uniform_(self.W_dst, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """Compute multi-head GATv2 outputs.

        Args:
            h: (N, in_dim)
            adj: (N, N) dense adjacency with self-loops
        Returns:
            h_prime: (N, out_dim)
            attn_mean: (N, N) averaged attention over heads
        """
        N = h.size(0)

        # Linear projections per head: Q (src) and K (dst)
        # x: (N, D), W: (H, D, Hd) -> (N, H, Hd)
        Q = torch.einsum('nd,hdf->nhf', h, self.W_src)
        K = torch.einsum('nd,hdf->nhf', h, self.W_dst)

        # reorder to (H, N, Hd) for broadcasting in pairwise scores
        Qh = Q.permute(1, 0, 2)
        Kh = K.permute(1, 0, 2)

        # Dense scores per head: e_h(i,j) = a_h^T LeakyReLU(Q_i + K_j)
        # (H, N, 1, Hd) + (H, 1, N, Hd) -> (H, N, N, Hd)
        sum_qk = Qh.unsqueeze(2) + Kh.unsqueeze(1)
        e = self.leakyrelu(sum_qk)
        # dot with a: (H, N, N, Hd) . (H, Hd) -> (H, N, N)
        e = (e * self.a.view(self.heads, 1, 1, self.head_dim)).sum(dim=-1)

        # mask and numerically-stable softmax across neighbors (dim=-1)
        mask = (adj > 0)
        if e.dtype in (torch.float16, torch.bfloat16):
            very_neg = -1e4
        else:
            very_neg = -1e9
        e = e.masked_fill(~mask.unsqueeze(0), very_neg)
        # row-wise max subtraction for stability
        e = e - e.amax(dim=-1, keepdim=True)
        attn_prob = F.softmax(e, dim=-1)

        # dropout on attention probabilities (used for aggregation),
        # keep a clean copy for logging so rows still sum to 1.
        attn = self.attn_dropout(attn_prob)

        # Aggregate per head with batched bmm: (H, N, N) @ (H, N, Hd) -> (H, N, Hd)
        out = torch.bmm(attn, Kh)
        # back to (N, H, Hd)
        out = out.permute(1, 0, 2)

        if self.concat:
            h_prime = out.reshape(N, self.out_dim)
        else:
            # average across heads
            h_prime = out.mean(dim=1)

        # feature dropout and optional bias
        h_prime = self.feat_dropout(h_prime)
        if self.bias is not None:
            h_prime = h_prime + self.bias

        # average attention across heads for logging: (N, N)
        attn_mean = attn_prob.mean(dim=0)

        return h_prime, attn_mean
