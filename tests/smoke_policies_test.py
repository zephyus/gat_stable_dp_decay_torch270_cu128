import os
import sys
# add parent dir of this file to sys.path so 'agents' resolves when running from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import numpy as np
import torch
from agents.policies import LstmPolicy, NCMultiAgentPolicy


def test_lstm_policy():
    n_s = 8
    n_a = 5
    n_n = 0  # no neighbors
    n_step = 6
    T = n_step
    policy = LstmPolicy(n_s=n_s, n_a=n_a, n_n=n_n, n_step=T)
    # Roll forward single-step policy sampling and backward pass
    obs_seq = np.random.randn(T, n_s).astype(np.float32)
    dones = np.zeros(T, dtype=np.float32)
    acts = np.random.randint(0, n_a, size=T).astype(np.int32)
    nactions = np.zeros((T,0), dtype=np.int32)  # no neighbors
    # value targets
    Rs = np.random.randn(T).astype(np.float32)
    Advs = np.random.randn(T).astype(np.float32)

    # backward
    policy.backward(obs_seq, nactions, acts, dones, Rs, Advs, e_coef=0.01, v_coef=0.5)
    assert policy.loss is not None

    # forward prob
    ob0 = np.random.randn(n_s).astype(np.float32)
    done0 = np.array(0., dtype=np.float32)
    probs = policy.forward(ob0, done0)
    assert probs.shape[0] == n_a
    assert np.allclose(probs.sum(), 1.0, atol=1e-4)


def test_nc_multi_agent_policy():
    n_agent = 3
    n_s = 7
    n_a = 4
    n_step = 5
    n_fc = 16
    n_h = 16
    neighbor_mask = np.ones((n_agent, n_agent), dtype=np.int32) - np.eye(n_agent, dtype=np.int32)  # fully connected (no self)

    policy = NCMultiAgentPolicy(
        n_s=n_s,
        n_a=n_a,
        n_agent=n_agent,
        n_step=n_step,
        neighbor_mask=neighbor_mask,
        n_fc=n_fc,
        n_h=n_h,
        identical=True,
        model_config=None,
    )

    T = n_step
    obs = np.random.randn(n_agent, T, n_s).astype(np.float32)
    fps = np.random.randn(n_agent, T, n_a).astype(np.float32)  # use random logits-like features
    acts = np.random.randint(0, n_a, size=(n_agent, T)).astype(np.int32)
    dones = np.zeros(T, dtype=np.float32)
    Rs = np.random.randn(n_agent, T).astype(np.float32)
    Advs = np.random.randn(n_agent, T).astype(np.float32)

    policy.backward(obs, fps, acts, dones, Rs, Advs, e_coef=0.01, v_coef=0.5)
    assert policy.loss is not None

    # forward probabilities
    ob0 = np.random.randn(n_agent, n_s).astype(np.float32)
    fp0 = np.random.randn(n_agent, n_a).astype(np.float32)
    done0 = np.zeros(n_agent, dtype=np.float32)  # broadcasted later
    agent_probs = policy.forward(ob0, done0, fp0, out_type='p')
    assert len(agent_probs) == n_agent
    for p in agent_probs:
        assert p.shape[0] == n_a
        assert np.allclose(p.sum(), 1.0, atol=1e-4)


if __name__ == '__main__':
    torch.set_grad_enabled(True)
    test_lstm_policy()
    test_nc_multi_agent_policy()
    print('Smoke tests passed.')
