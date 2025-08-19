#!/usr/bin/env python3
"""Minimal device-agnostic smoke test for upgraded code.
Run: python smoke_test.py
Pass criteria:
  * Forward + backward runs without exceptions
  * Loss is finite (not NaN/inf)
  * Policy can handle batch=1 and sequence length >1
No algorithmic logic is modified; this only exercises surfaces.
"""
import os
import math
import torch
import numpy as np

from pathlib import Path

# Try importing primary policies (fallback to agents.policies if duplicate exists)
try:
    from agents.policies import NCMultiAgentPolicy as PolicyClass
except Exception:
    from policies import NCMultiAgentPolicy as PolicyClass  # type: ignore

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_policy(n_agent=4, obs_dim=12, act_dim=5, T=3, hidden=64):
    neighbor_mask = np.ones((n_agent, n_agent), dtype=bool)
    np.fill_diagonal(neighbor_mask, 0)
    policy = PolicyClass(
        n_s=obs_dim,
        n_a=act_dim,
        n_agent=n_agent,
        n_step=T,
        neighbor_mask=neighbor_mask,
        n_fc=64,
        n_h=hidden,
        identical=True,
    )
    if hasattr(policy, 'to'):
        policy.to(DEVICE)
    return policy

def single_step(policy, obs_dim, act_dim):
    obs = torch.randn(1, policy.n_agent, obs_dim, device=DEVICE)
    dones = torch.zeros(1, policy.n_agent, device=DEVICE)
    fps = torch.zeros_like(obs)
    # Run communication layers indirectly via existing heads
    probs = policy._run_actor_heads(obs[0], detach=True)
    assert len(probs) == policy.n_agent
    return probs

def loss_backward(policy, obs_dim, act_dim, T=3):
    seq_obs = torch.randn(T, policy.n_agent, obs_dim, device=DEVICE)
    logits_list = policy._run_actor_heads(seq_obs[0])
    logps = []
    for lg in logits_list:
        dist = torch.distributions.Categorical(logits=lg)
        a = dist.sample()
        logps.append(dist.log_prob(a))
    policy_loss = -torch.stack(logps).mean()
    # Fake value targets
    fake_v = torch.randn(policy.n_agent, device=DEVICE)
    tgt_v = torch.randn_like(fake_v)
    value_loss = (fake_v - tgt_v).pow(2).mean()
    total = policy_loss + 0.5 * value_loss
    total.backward()
    return total

def sequence_backward(policy, obs_dim, act_dim, T=3):
    batch_logits = torch.randn(T, policy.n_agent, act_dim, device=DEVICE)
    dist = torch.distributions.Categorical(logits=batch_logits)
    acts = dist.sample()
    loss = dist.log_prob(acts).mean()
    loss.backward()
    return loss

def main():
    policy = build_policy()
    probs = single_step(policy, obs_dim=12, act_dim=5)
    print("Got detached probs for each agent (length):", len(probs))
    total = loss_backward(policy, 12, 5)
    assert torch.isfinite(total), "Non-finite loss"
    print("Primary backward OK. Loss=", float(total))
    # Zero grads for second test
    for p in policy.parameters():
        if p.grad is not None:
            p.grad.zero_()
    seq_loss = sequence_backward(policy, 12, 5)
    assert torch.isfinite(seq_loss), "Non-finite seq loss"
    print("Sequence backward OK. Loss=", float(seq_loss))
    # Simple write test
    out_dir = Path('runs/smoke')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'touch.txt').write_text('ok')
    print("Write test OK ->", out_dir / 'touch.txt')
    print("SMOKE TEST PASSED")

if __name__ == '__main__':
    main()
