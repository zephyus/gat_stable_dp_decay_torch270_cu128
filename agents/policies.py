"""Cleaned version of policies with consistent indentation (no logic changes)."""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn, DEVICE
from agents.gat import GraphAttention


class Policy(nn.Module):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical):
        super().__init__()
        self.name = policy_name if agent_name is None else f"{policy_name}_{agent_name}"
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.identical = identical

    def forward(self, *_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError

    def _init_actor_head(self, n_h, n_a=None):
        if n_a is None:
            n_a = self.n_a
        self.actor_head = nn.Linear(n_h, n_a)
        init_layer(self.actor_head, 'fc')

    def _init_critic_head(self, n_h, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            if self.identical:
                n_na_sparse = self.n_a * n_n
            else:
                n_na_sparse = sum(self.na_dim_ls)
            n_h += n_na_sparse
        self.critic_head = nn.Linear(n_h, 1)
        init_layer(self.critic_head, 'fc')

    def _run_critic_head(self, h, na, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            na = torch.from_numpy(na).long()
            if self.identical:
                na_sparse = one_hot(na, self.n_a).view(-1, self.n_a * n_n)
            else:
                parts = []
                for na_val, na_dim in zip(torch.chunk(na, n_n, dim=1), self.na_dim_ls):
                    parts.append(torch.squeeze(one_hot(na_val, na_dim), dim=1))
                na_sparse = torch.cat(parts, dim=1)
            h = torch.cat([h, na_sparse.to(h.device)], dim=1)
        return self.critic_head(h).squeeze(-1)

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
        # Support both single-agent (actor_head) and multi-agent (actor_heads list) policies
        if hasattr(self, 'actor_head') and hasattr(self.actor_head, 'weight'):
            device = self.actor_head.weight.device
        elif hasattr(self, 'actor_heads') and len(self.actor_heads) > 0:
            device = self.actor_heads[0].weight.device
        else:
            device = DEVICE
        vs = vs.to(device)
        As = As.to(device)
        Advs = Advs.to(device)
        Rs = Rs.to(device)
        log_probs = actor_dist.log_prob(As)
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

    def _update_tensorboard(self, summary_writer, global_step):
        summary_writer.add_scalar(f'loss/{self.name}_entropy_loss', self.entropy_loss, global_step)
        summary_writer.add_scalar(f'loss/{self.name}_policy_loss', self.policy_loss, global_step)
        summary_writer.add_scalar(f'loss/{self.name}_value_loss', self.value_loss, global_step)
        summary_writer.add_scalar(f'loss/{self.name}_total_loss', self.loss, global_step)
        summary_writer.add_scalar('loss/entropy_loss', self.entropy_loss, global_step)
        summary_writer.add_scalar('loss/policy_loss', self.policy_loss, global_step)
        summary_writer.add_scalar('loss/value_loss', self.value_loss, global_step)
        summary_writer.add_scalar('loss/total_loss', self.loss, global_step)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None, na_dim_ls=None, identical=True):
        super().__init__(n_a, n_s, n_step, 'lstm', name, identical)
        if not self.identical:
            self.na_dim_ls = na_dim_ls
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self._init_net()
        self.to(DEVICE)
        self._reset()

    def backward(self, obs, nactions, acts, dones, Rs, Advs, e_coef, v_coef, summary_writer=None, global_step=None):
        device = DEVICE
        obs_t = torch.from_numpy(obs).float().to(device)
        dones_t = torch.from_numpy(dones).float().to(device)
        xs = self._encode_ob(obs_t)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones_t, self.states_bw)
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=self.actor_head(hs))
        vs = self._run_critic_head(hs, nactions)
        self.policy_loss, self.value_loss, self.entropy_loss = self._run_loss(
            actor_dist, e_coef, v_coef, vs,
            torch.from_numpy(acts).long(),
            torch.from_numpy(Rs).float(),
            torch.from_numpy(Advs).float()
        )
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, naction=None, out_type='p'):
        device = DEVICE
        ob_t = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(device)
        done_t = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(device)
        x = self._encode_ob(ob_t)
        h, new_states = run_rnn(self.lstm_layer, x, done_t, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return F.softmax(self.actor_head(h), dim=1).squeeze().cpu().detach().numpy()
        return self._run_critic_head(h, np.array([naction])).cpu().detach().numpy()

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_lstm * 2, device=DEVICE)
        self.states_bw = torch.zeros(self.n_lstm * 2, device=DEVICE)


class FPPolicy(LstmPolicy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None, na_dim_ls=None, identical=True):
        super().__init__(n_s, n_a, n_n, n_step, n_fc, n_lstm, name, na_dim_ls, identical)

    def _init_net(self):
        if self.identical:
            self.n_x = self.n_s - self.n_n * self.n_a
        else:
            self.n_x = int(self.n_s - sum(self.na_dim_ls))
        self.fc_x_layer = nn.Linear(self.n_x, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        n_h = self.n_fc
        if self.n_n:
            self.fc_p_layer = nn.Linear(self.n_s - self.n_x, self.n_fc)
            init_layer(self.fc_p_layer, 'fc')
            n_h += self.n_fc
        self.lstm_layer = nn.LSTMCell(n_h, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _encode_ob(self, ob):
        x = F.relu(self.fc_x_layer(ob[:, :self.n_x]))
        if self.n_n:
            p = F.relu(self.fc_p_layer(ob[:, self.n_x:]))
            x = torch.cat([x, p], dim=1)
        return x


class NCMultiAgentPolicy(Policy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64, n_s_ls=None, n_a_ls=None, model_config=None, identical=True):
        super().__init__(n_a, n_s, n_step, 'nc', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self.model_config = model_config
        self._init_net()
        # allocate zero_pad before moving to device for consistent placement
        self.zero_pad = nn.Parameter(torch.zeros(1, 2 * self.n_fc, device=DEVICE), requires_grad=False)
        self.latest_attention_scores = None
        use_gat_env_value = os.getenv('USE_GAT', '1')
        self.use_gat = use_gat_env_value == '1'
        if not self.use_gat:
            self.gat_layer = nn.Identity()
        self.to(DEVICE)
        self._reset()

    def backward(self, obs, fps, acts, dones, Rs, Advs, e_coef, v_coef, summary_writer=None, global_step=None):
        device = DEVICE
        obs_t = torch.from_numpy(obs).float().transpose(0, 1).to(device)
        dones_t = torch.from_numpy(dones).float().to(device)
        fps_t = torch.from_numpy(fps).float().transpose(0, 1).to(device)
        acts_t = torch.from_numpy(acts).long().to(device)
        hs, new_states = self._run_comm_layers(obs_t, dones_t, fps_t, self.states_bw)
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts_t)
        self.policy_loss = torch.zeros((), device=device)
        self.value_loss = torch.zeros((), device=device)
        self.entropy_loss = torch.zeros((), device=device)
        Rs_t = torch.from_numpy(Rs).float().to(device)
        Advs_t = torch.from_numpy(Advs).float().to(device)
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            pl_i, vl_i, el_i = self._run_loss(actor_dist_i, e_coef, v_coef, vs[i], acts_t[i], Rs_t[i], Advs_t[i])
            self.policy_loss = self.policy_loss + pl_i
            self.value_loss = self.value_loss + vl_i
            self.entropy_loss = self.entropy_loss + el_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)
            if self.use_gat and self.latest_attention_scores is not None:
                mask = self.adj.to(self.latest_attention_scores.device) > 0
                scores = self.latest_attention_scores[mask]
                if scores.numel() > 0:
                    summary_writer.add_histogram('GAT/attention_scores', scores.detach().cpu().numpy(), global_step)
                self.latest_attention_scores = None

    def forward(self, ob, done, fp, action=None, out_type='p'):
        device = DEVICE
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(device)
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(device)
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float().to(device)
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h, detach=True)
        action = torch.from_numpy(np.expand_dims(action, axis=1)).long().to(device)
        return self._run_critic_heads(h, action, detach=True)

    def _get_comm_s(self, i, n_n, x, h, p):
        device = h.device
        x = x.to(device)
        p = p.to(device)
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        p_i = torch.index_select(p, 0, js)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            p_i = p_i.view(1, self.n_a * n_n)
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            p_i_ls, nx_i_ls = [], []
            for j in range(n_n):
                p_i_ls.append(p_i[j].narrow(0, 0, self.na_ls_ls[i][j]))
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            p_i = torch.cat(p_i_ls).unsqueeze(0)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        fc_x_input = torch.cat([x_i, nx_i], dim=1)
        current_n_ns = fc_x_input.size(1)
        fc_x = self._get_fc_x(i, n_n, current_n_ns)
        s_x = F.relu(fc_x(fc_x_input))
        return torch.cat([s_x, F.relu(self.fc_p_layers[i](p_i)), F.relu(self.fc_m_layers[i](m_i))], dim=1)

    def _get_neighbor_dim(self, i_agent):
        n_n = int(np.sum(self.neighbor_mask[i_agent]))
        if self.identical:
            return n_n, self.n_s * (n_n + 1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        ns_ls, na_ls = [], []
        for j in np.where(self.neighbor_mask[i_agent])[0]:
            ns_ls.append(self.n_s_ls[j])
            na_ls.append(self.n_a_ls[j])
        return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls

    def _init_actor_head(self, n_a):
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        if n_n:
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
        lstm_layer = nn.LSTMCell(3 * self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_na):
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleDict()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        gat_dropout_init = 0.1
        if self.model_config:
            gat_dropout_init = self.model_config.getfloat('gat_dropout_init', 0.2)
        self.gat_layer = GraphAttention(3 * self.n_fc, 3 * self.n_fc, dropout=gat_dropout_init, alpha=0.2)
        self.adj = torch.tensor(self.neighbor_mask, dtype=torch.float32) + torch.eye(self.neighbor_mask.shape[0])
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _reset(self):
        # keep on unified DEVICE to avoid implicit CPU<->GPU copies later
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2, device=DEVICE)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2, device=DEVICE)

    def _run_actor_heads(self, hs, detach=False):
        ps = []
        for i in range(self.n_agent):
            if detach:
                ps.append(F.softmax(self.actor_heads[i](hs[i]), dim=1).squeeze().cpu().detach().numpy())
            else:
                # return raw logits (Categorical(logits=...) will apply softmax internally)
                ps.append(self.actor_heads[i](hs[i]))
        return ps

    def _run_comm_layers(self, obs, dones, fps, states):
        obs_seq = batch_to_seq(obs)
        done_seq = batch_to_seq(dones)
        fp_seq = batch_to_seq(fps)
        h, c = torch.chunk(states, 2, dim=1)
        device = h.device
        h = h.to(device)
        c = c.to(device)
        outputs = []
        for x, p, done in zip(obs_seq, fp_seq, done_seq):
            done = done.to(device)
            # normalize done shape to (1,1) for broadcasting; supports (1,), (1,n_agent), scalar
            if done.dim() == 1:
                done_norm = done.view(1, 1)
            elif done.dim() == 2 and done.size(1) != 1:
                # take first entry as global done (original logic treated done scalar)
                done_norm = done[:, :1]
            else:
                done_norm = done
            x = x.to(device).squeeze(0)
            p = p.to(device).squeeze(0)
            s_list = []
            for i in range(self.n_agent):
                n_n = int(self.neighbor_mask[i].sum().item())
                if n_n:
                    s = self._get_comm_s(i, n_n, x, h, p)
                else:
                    if self.identical:
                        x_i = x[i].unsqueeze(0)
                        current_n_ns = x_i.size(1)
                    else:
                        x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                        current_n_ns = self.n_s_ls[i]
                    fc_x = self._get_fc_x(i, 0, current_n_ns)
                    s_x = F.relu(fc_x(x_i))
                    s = torch.cat([s_x, self.zero_pad], dim=1)
                s_list.append(s.squeeze(0))
            s_all = torch.stack(s_list, dim=0)
            if self.use_gat:
                s_all, attention_scores = self.gat_layer(s_all, self.adj.to(x.device))
                self.latest_attention_scores = attention_scores.detach()
            else:
                s_all = self.gat_layer(s_all)
                self.latest_attention_scores = None
            next_h, next_c = [], []
            for i in range(self.n_agent):
                s_i = s_all[i].unsqueeze(0)
                h_i = h[i].unsqueeze(0) * (1 - done_norm)
                c_i = c[i].unsqueeze(0) * (1 - done_norm)
                nh_i, nc_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(nh_i)
                next_c.append(nc_i)
            h, c = torch.cat(next_h), torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        vs = []
        device = hs[0].device if len(hs) else DEVICE
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i]
            if n_n:
                js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long().to(device)
                na_i = torch.index_select(actions, 0, js)
                na_i_ls = [one_hot(na_i[j], self.na_ls_ls[i][j], device=device) for j in range(n_n)]
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze()
            # keep only last dim (B,1)->(B); avoid dropping batch when B=1 unintentionally
            v_i = v_i if v_i.dim() == 1 else v_i.squeeze(-1)
            vs.append(v_i.cpu().detach().numpy() if detach else v_i)
        return vs

    def _get_fc_x(self, agent_id: int, n_n: int, n_ns: int) -> nn.Linear:
        key = f'agent_{agent_id}_nn_{n_n}_in{n_ns}'
        if key not in self.fc_x_layers:
            layer = nn.Linear(n_ns, self.n_fc)
            init_layer(layer, 'fc')
            self.fc_x_layers[key] = layer.to(self.zero_pad.device)
        else:
            assert self.fc_x_layers[key].in_features == n_ns, f"fc_x[{key}] expects {self.fc_x_layers[key].in_features}, got {n_ns}"
        return self.fc_x_layers[key]


class NCLMMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64, n_s_ls=None, n_a_ls=None, groups=0, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'nclm', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.groups = groups
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self.to(DEVICE)
        self._reset()

    def backward(self, obs, fps, acts, dones, Rs, Advs, e_coef, v_coef, summary_writer=None, global_step=None):
        device = DEVICE
        obs_t = torch.from_numpy(obs).float().transpose(0, 1).to(device)
        dones_t = torch.from_numpy(dones).float().to(device)
        fps_t = torch.from_numpy(fps).float().transpose(0, 1).to(device)
        acts_t = torch.from_numpy(acts).long().to(device)
        hs, new_states = self._run_comm_layers(obs_t, dones_t, fps_t, self.states_bw)
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        bps = self._run_actor_heads(hs, acts_t)
        for i in range(self.n_agent):
            if i in self.groups:
                ps[i] = bps[i]
        vs = self._run_critic_heads(hs, acts_t)
        self.policy_loss = torch.zeros((), device=device)
        self.value_loss = torch.zeros((), device=device)
        self.entropy_loss = torch.zeros((), device=device)
        Rs_t = torch.from_numpy(Rs).float().to(device)
        Advs_t = torch.from_numpy(Advs).float().to(device)
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            pl_i, vl_i, el_i = self._run_loss(actor_dist_i, e_coef, v_coef, vs[i], acts_t[i], Rs_t[i], Advs_t[i])
            self.policy_loss = self.policy_loss + pl_i
            self.value_loss = self.value_loss + vl_i
            self.entropy_loss = self.entropy_loss + el_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, fp, action=None, out_type='p'):
        device = DEVICE
        ob_t = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(device)
        done_t = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(device)
        fp_t = torch.from_numpy(np.expand_dims(fp, axis=0)).float().to(device)
        h, new_states = self._run_comm_layers(ob_t, done_t, fp_t, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            if action is not None:
                action_t = torch.from_numpy(np.expand_dims(action, axis=1)).long().to(device)
                return self._run_actor_heads(h, action_t, detach=True)
            return self._run_actor_heads(h, detach=True)
        action_t = torch.from_numpy(np.expand_dims(action, axis=1)).long().to(device)
        return self._run_critic_heads(h, action_t, detach=True)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        n_lstm_in = 3 * self.n_fc
        if n_n:
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
        lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_backhand_actor_head(self, n_a, n_na):
        actor_head = nn.Linear(self.n_h + n_na, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleDict()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            if i not in self.groups:
                self._init_actor_head(n_a)
            else:
                self._init_backhand_actor_head(n_a, n_na)
            self._init_critic_head(n_na)

    def _run_actor_heads(self, hs, preactions=None, detach=False):
        ps = [0] * self.n_agent
        first_stage = preactions is None or (isinstance(preactions, (list, tuple)) and all(a is None for a in preactions))
        if first_stage:
            for i in range(self.n_agent):
                if i not in self.groups:
                    out = self.actor_heads[i](hs[i])
                    ps[i] = F.softmax(out, dim=1).cpu().squeeze().detach().numpy() if detach else out  # raw logits
        else:
            device = hs[0].device if len(hs) else DEVICE
            for i in range(self.n_agent):
                if i in self.groups:
                    n_n = self.n_n_ls[i]
                    if n_n:
                        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
                        na_i = torch.index_select(preactions, 0, js)
                        na_i_ls = [one_hot(na_i[j], self.na_ls_ls[i][j], device=device) for j in range(n_n)]
                        h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
                    else:
                        h_i = hs[i]
                    out = self.actor_heads[i](h_i)
                    ps[i] = F.softmax(out, dim=1).cpu().squeeze().detach().numpy() if detach else out  # raw logits
        return ps


class ConsensusPolicy(NCMultiAgentPolicy):  # Placeholder cleaned class (left minimal)
    pass


class CommNetMultiAgentPolicy(NCMultiAgentPolicy):  # Placeholder cleaned class (left minimal)
    pass


class DIALMultiAgentPolicy(NCMultiAgentPolicy):  # Placeholder cleaned class (left minimal)
    pass
