# PPO Guide

This note maps PPO formulas to the current rsl_rl implementation.

Paper: Proximal Policy Optimization Algorithms
<https://arxiv.org/abs/1707.06347>

GAE reference: High-Dimensional Continuous Control Using Generalized Advantage Estimation
<https://arxiv.org/abs/1506.02438>

## Mental Model

PPO is an on-policy actor-critic method:

```text
rollout policy pi_old     -> collect s_t, a_t, r_t, done_t
critic V_old              -> bootstrap returns and advantages
current policy pi_theta   -> re-score old actions over several mini-batch epochs

PPO learns:
    keep pi_theta close enough to pi_old through clipped rho_t
    improve actions that had positive A_t
    reduce probability of actions that had negative A_t
    fit V_theta_t to return target R_t
```

The key object is the probability ratio:

$$
\rho_t(\theta) =
\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
= \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t))
$$

## Code Map

```text
[OnPolicyRunner.learn]
|
|-- [PPO.act]
|   |-- sample a_t from pi_old(.|s_t)
|   `-- store logp_old_t, V_old_t, mu_old_t, sigma_old_t
|
|-- env.step(a_t)
|   `-- returns r_t, done_t, extras, and s_{t+1}
|
|-- [PPO.process_env_step]
|   `-- store transition and apply optional RND/time-limit reward adjustments
|
|-- [PPO.compute_returns]
|   `-- [RolloutStorage.compute_returns]: delta_t, A_t, R_t
|
`-- [PPO.update]
    |
    |-- [RolloutStorage.mini_batch_generator]
    |   `-- yields fixed pi_old rollout samples
    |
    |-- recompute logp_theta_t, V_theta_t, H_t
    |-- rho_t = exp(logp_theta_t - logp_old_t)
    |-- L_clip = -mean(min(rho_t A_t, clip(rho_t) A_t))
    |-- L_V = value regression to R_t
    |-- L_total = L_clip + c_v L_V - c_H H_t + optional losses
    `-- optimizer.step()
```

Links use workspace-relative `#L...` anchors. If VSCode opens the file but ignores the line anchor, use Quick Open with
`path:line`, for example `rsl_rl/algorithms/ppo.py:207`.

- [OnPolicyRunner.learn](../runners/on_policy_runner.py#L64)
- [PPO](./ppo.py#L19)
- [PPO.act](./ppo.py#L155)
- [PPO.process_env_step](./ppo.py#L169)
- [PPO.compute_returns](./ppo.py#L200)
- [PPO.update](./ppo.py#L207)
- [RolloutStorage.compute_returns](../storage/rollout_storage.py#L127)
- [RolloutStorage.mini_batch_generator](../storage/rollout_storage.py#L171)
- [RolloutStorage.recurrent_mini_batch_generator](../storage/rollout_storage.py#L227)

## Symbols

| Paper symbol | ASCII symbol | Code name | Meaning |
| --- | --- | --- | --- |
| $s_t$ | `s_t` | `obs` / `obs_batch` | Observation at rollout/update time. |
| $a_t$ | `a_t` | `actions` / `actions_batch` | Action sampled from $\pi_{\text{old}}$. |
| $r_t$ | `r_t` | `rewards` | Scalar reward seen by PPO, possibly shaped by add-ons. |
| $\pi_{\text{old}}$ | `pi_old` | rollout-time `policy` snapshot | Policy distribution that generated the data. |
| $\pi_\theta$ | `pi_theta` | current `self.policy` | Policy distribution during update. |
| $\log \pi_{\text{old}}(a_t|s_t)$ | `logp_old_t` | `old_actions_log_prob_batch` | Stored rollout log-probability. |
| $\log \pi_\theta(a_t|s_t)$ | `logp_theta_t` | `actions_log_prob_batch` | Recomputed current log-probability. |
| $\rho_t(\theta)$ | `rho_t` | `ratio` | PPO probability ratio. |
| $V_{\text{old}}(s_t)$ | `V_old_t` | `target_values_batch` | Stored rollout value estimate. |
| $V_\theta(s_t)$ | `V_theta_t` | `value_batch` | Current critic estimate. |
| $\delta_t$ | `delta_t` | `delta` | TD residual for GAE. |
| $A_t$ | `A_t` | `advantages_batch` | GAE-lambda advantage. |
| $R_t$ | `R_t` | `returns_batch` | Bootstrapped return target. |
| $\epsilon$ | `eps_clip` | `clip_param` | PPO clipping range. |
| $H_t$ | `H_t` | `entropy_batch` | Action distribution entropy. |
| $L_{\text{clip}}$ | `L_clip` | `surrogate_loss` | Negative clipped policy objective. |
| $L_V$ | `L_V` | `value_loss` | Critic regression loss. |
| $L_{\text{total}}$ | `L_total` | `loss` | Final PPO actor-critic loss. |
| $\mathrm{KL}(\pi_{\text{old}}, \pi_\theta)$ | `KL_old_theta` | `kl_mean` | Adaptive LR schedule signal. |

## Rollout Snapshot

`PPO.act()` samples from the current policy and stores the rollout-time quantities:

```python
self.transition.actions = self.policy.act(obs).detach()
self.transition.values = self.policy.evaluate(obs).detach()
self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
self.transition.action_mean = self.policy.action_mean.detach()
self.transition.action_sigma = self.policy.action_std.detach()
```

See [PPO.act](./ppo.py#L155).

These stored tensors are the $\pi_{\text{old}}$ side of the PPO ratio. They stay fixed during all mini-batch epochs for
this rollout.

## GAE And Returns

rsl_rl computes GAE in [RolloutStorage.compute_returns](../storage/rollout_storage.py#L127):

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$

$$
R_t = A_t + V_{\text{old}}(s_t)
$$

The code also masks terminal transitions with `next_is_not_terminal`, and `PPO.process_env_step()` handles time-limit
truncations by adding a bootstrap term before storage.

## Clipped Policy Objective

During update, rsl_rl recomputes the current log-probability for old rollout actions:

```python
actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
```

See [PPO.update](./ppo.py#L207).

The paper maximizes:

$$
L^{CLIP}(\theta) =
\mathbb{E}_t
\left[
\min\left(
\rho_t(\theta) A_t,
\mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
$$

The implementation minimizes the negative objective:

$$
L_{\text{clip}} =
\mathbb{E}_t
\left[
\max\left(
-\rho_t A_t,
-\mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
$$

## Value Loss

With `use_clipped_value_loss=True`, the critic update also clips the change from $V_{\text{old}}$:

$$
V_{\text{clip}} =
V_{\text{old}} + \mathrm{clip}(V_\theta - V_{\text{old}}, -\epsilon, \epsilon)
$$

$$
L_V =
\max\left((V_\theta - R_t)^2,\ (V_{\text{clip}} - R_t)^2\right)
$$

This mirrors PPO policy clipping: avoid large critic updates that improve the value loss too aggressively on one
mini-batch.

## Total Loss

The actor-critic loss in rsl_rl is:

$$
L_{\text{total}} =
L_{\text{clip}}
+ c_v L_V
- c_H H_t
$$

Code:

```python
loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
```

Optional terms can be added:

- RND trains a separate predictor loss and adds intrinsic rewards before GAE.
- Symmetry can augment mini-batches and/or add a mirror loss to the actor mean.
- Adaptive KL changes the optimizer learning rate, but it is not added directly to `loss`.

## Implementation Notes

1. `ratio` is called $\rho_t$ in this document to avoid confusing it with reward $r_t$.
2. Feed-forward policies use shuffled flat mini-batches; recurrent policies use padded trajectories and masks.
3. `normalize_advantage_per_mini_batch` chooses whether advantage normalization happens globally in storage or locally
   inside each mini-batch.
4. Multi-GPU training averages gradients in [PPO.reduce_parameters](./ppo.py#L479) before the optimizer step.

