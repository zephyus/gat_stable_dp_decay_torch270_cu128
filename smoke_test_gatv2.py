# Minimal sanity test for MultiHeadGATv2Dense integration
import os
import torch
from agents.gat import MultiHeadGATv2Dense


def run():
    torch.manual_seed(0)
    N = 5
    n_fc = 8
    C = 3 * n_fc
    x = torch.randn(N, C, requires_grad=True)
    # fully connected with self-loops
    adj = torch.ones(N, N)

    gat = MultiHeadGATv2Dense(in_dim=C, out_dim=C, heads=6, concat=True, feat_drop=0.1, attn_drop=0.1)
    y, attn = gat(x, adj)

    assert y.shape == (N, C), f"y shape {y.shape} != {(N, C)}"
    assert attn.shape == (N, N), f"attn shape {attn.shape} != {(N, N)}"

    # masked row-sum ~ 1
    row_sum = (attn * (adj > 0)).sum(dim=1)
    if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4, rtol=1e-4):
        raise AssertionError(f"row_sum not ~1: {row_sum}")

    loss = y.pow(2).mean()
    loss.backward()
    assert torch.isfinite(x.grad).all(), "x.grad has non-finite values"

    print("PASS: MultiHeadGATv2Dense forward/backward + attention rowsum")


if __name__ == "__main__":
    run()
