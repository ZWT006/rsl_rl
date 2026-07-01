# AMP Add-on Guide

This note maps the AMP paper formulas to the current rsl_rl add-on implementation.

Paper: AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control
<https://arxiv.org/pdf/2104.02180>

## Mental Model

AMP adds a learned style reward to normal RL:

```text
policy rollout transition  -> env builds phi_pi
reference motion dataset   -> env samples phi_M
discriminator D(phi)       -> expert-like score

PPO sees:
    shaped_reward = task_reward + reward_coef * amp_style_reward

Discriminator learns:
    D_phi_M  = D(phi_M)  -> +1
    D_phi_pi = D(phi_pi) -> -1

Policy learns:
    produce transitions phi_pi that make D_phi_pi close to +1
```

PPO does not know about AMP. The runner computes the AMP reward before PPO stores the transition.

## Code Map

```text
[OnPolicyRunner.learn]
|
|-- env.step()
|   `-- returns task reward r_G_t and obs["amp"] = phi_pi
|
|-- [AMPAddon.process_env_step]
|   |
|   |-- [AMPAddon.compute_rewards]
|   |   `-- Eq. 7: d_t = D_phi_pi, r_S_t = max(0, 1 - 0.25 * (d_t - 1)^2)
|   |
|   |-- [AMPAddon.combine_rewards]
|   |   `-- Eq. 4: r_t = r_G_t + w_S * r_S_t
|   |
|   `-- [AMPStorage.add_transitions]
|       `-- stores policy samples phi_pi for the discriminator update
|
|-- [PPO.update]
|   `-- PPO learns from the shaped reward; no gradient flows through D
|
`-- [AMPAddon.update]
    |
    |-- [AMPStorage.mini_batch_generator]
    |   `-- yields policy samples b_pi from stored phi_pi
    |
    |-- [AMPAddon._sample_expert_observations]
    |   `-- samples expert/reference samples b_M from env.get_amp_expert_obs()
    |
    |-- [AMPAddon._compute_discriminator_loss]
    |   `-- Eq. 6: L_D from D_phi_pi and D_phi_M
    |
    |-- [AMPAddon._compute_gradient_penalty]
    |   `-- Eq. 8: L_GP on phi_M
    |
    `-- optimizer.step()
```

Links:

These links use workspace-relative `#L...` anchors, which work on GitHub and are the most VSCode-friendly Markdown
format. If your VSCode build opens the file but ignores the line anchor, use Quick Open with `path:line`, for example
`rsl_rl/addons/amp.py:152`.

- [OnPolicyRunner.learn AMP reward hook](../runners/on_policy_runner.py#L117)
- [AMPAddon](./amp.py#L19)
- [AMPAddon.process_env_step](./amp.py#L136)
- [AMPAddon.compute_rewards](./amp.py#L152)
- [AMPAddon.combine_rewards](./amp.py#L176)
- [PPO.update](../algorithms/ppo.py#L200)
- [AMPAddon.update](./amp.py#L185)
- [AMPStorage.add_transitions](../storage/amp_storage.py#L47)
- [AMPStorage.mini_batch_generator](../storage/amp_storage.py#L59)
- [AMPAddon._sample_expert_observations](./amp.py#L324)
- [AMPAddon._compute_discriminator_loss](./amp.py#L335)
- [AMPAddon._compute_gradient_penalty](./amp.py#L362)
- [AMPDiscriminator.forward](../modules/amp.py#L52)

## Symbols

| Paper symbol | ASCII symbol | Code name | Meaning |
| --- | --- | --- | --- |
| $s_t, s_{t+1}$ | `s_t, s_t_plus_1` | env state transition | Full simulator transition. |
| $\Phi(s)$ | `Phi(s)` | env feature map | Built by the environment or motion library. |
| $\phi_\pi$ | `phi_pi` | `obs[obs_key]`, usually `obs["amp"]` | Policy transition feature. |
| $\phi_M$ | `phi_M` | `get_amp_expert_obs(batch_size)` | Reference transition feature. |
| $D(\phi)$ | `D(phi)` | `self.discriminator(phi)` | Raw discriminator score, not a probability. |
| $r^G_t$ | `r_G_t` | `rewards` before AMP shaping | Task reward returned by the environment. |
| $r^S_t$ | `r_S_t` | `amp_rewards` | Style reward computed from $D(\phi_\pi)$. |
| $w_S$ | `w_S` | `reward_coef` | Weight of the style reward. |
| $r_t$ | `r_t` | `shaped_rewards` | Task-plus-style reward passed to PPO. |
| $d_t = D(\phi_\pi)$ | `d_t` | `logits` in `compute_rewards()` | Policy discriminator score used for Eq. 7 reward. |
| $b_\pi$ | `b_pi` | `policy_obs` | Mini-batch of policy AMP features from the rollout. |
| $b_M$ | `b_M` | `expert_obs` | Mini-batch of reference AMP features from the motion dataset. |
| $D(\phi_\pi)$ | `D_phi_pi` | `policy_logits` | Discriminator score for policy samples in Eq. 6. |
| $D(\phi_M)$ | `D_phi_M` | `expert_logits` | Discriminator score for expert samples in Eq. 6. |
| $L_\pi$ | `L_pi` | `policy_loss` | Policy-sample term $(D(\phi_\pi)+1)^2$. |
| $L_M$ | `L_M` | `expert_loss` | Expert-sample term $(D(\phi_M)-1)^2$. |
| $L_D$ | `L_D` | `discriminator_loss` | Least-squares discriminator loss from Eq. 6. |
| $L_{GP}$ | `L_GP` | `gradient_penalty` | Expert-feature gradient penalty from Eq. 8. |
| $\frac{w_{gp}}{2}$ | `w_gp_over_2` | `gradient_penalty_coef` | Code coefficient multiplying `L_GP`. |
| $L_{\text{wd}}$ | `L_wd` | `weight_decay` | Optional full-discriminator L2 regularizer. |
| $L_{\text{logit-wd}}$ | `L_logit_wd` | `logit_weight_decay` | Optional final-layer L2 regularizer. |
| $L_{\text{total}}$ | `L_total` | `loss` | Final discriminator optimization loss. |

## Rollout Path

The runner first gets the normal task reward from the environment:

```python
obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
```

See [on_policy_runner.py](../runners/on_policy_runner.py#L117).

At this point:

$$
\texttt{rewards} = r^G_t,\qquad
\texttt{obs["amp"]} = \phi_\pi = (\Phi(s_t), \Phi(s_{t+1}))
$$

Then the runner calls AMP before PPO stores the transition:

```python
rewards, amp_rewards = self.amp.process_env_step(obs, rewards, dones)
self.alg.process_env_step(obs, rewards, dones, extras)
```

See [on_policy_runner.py](../runners/on_policy_runner.py#L124).

This is why PPO only sees the shaped reward. The discriminator is not part of the PPO graph.

## Eq. 7: Style Reward

The default AMP style reward is:

$$
d_t = D(\phi_\pi)
$$

$$
r^S_t = \max\left(0,\ 1 - \frac{1}{4}(d_t - 1)^2\right)
$$

Code:

```python
logits = self.discriminator(amp_obs)
rewards = torch.clamp(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0)
```

See [AMPAddon.compute_rewards](./amp.py#L152).

Intuition:

$$
D(\phi_\pi) = -1 \Rightarrow r^S_t = 0.0
$$

$$
D(\phi_\pi) = 0 \Rightarrow r^S_t = 0.75
$$

$$
D(\phi_\pi) = 1 \Rightarrow r^S_t = 1.0
$$

The reward is computed under `torch.no_grad()`: PPO receives a scalar reward and does not backpropagate through `D`.

## Logging Notes

For `reward_type: quad`, the raw per-step style reward is bounded:

$$
0 \le r^S_t \le 1
$$

The runner logs three AMP reward views:

| Log key | Meaning |
| --- | --- |
| `AMP/mean_step_reward` | Raw per-step style reward averaged over the current rollout. |
| `AMP/mean_episode_reward` | Raw completed-episode AMP reward sum. |
| `Train/mean_amp_reward` | `reward_coef`-scaled AMP episode contribution used in the PPO reward. |

`AMP/mean_episode_reward` is the mean completed-episode sum stored in `amp_rewbuffer`:

$$
\texttt{AMP/mean\_episode\_reward}
\approx
\mathbb{E}_{\text{episodes}}
\left[\sum_t r^S_t\right]
$$

`Train/mean_amp_reward` is scaled for comparison with `Train/mean_task_reward`:

$$
\texttt{Train/mean\_amp\_reward}
\approx
\texttt{reward\_coef}\ 
\mathbb{E}_{\text{episodes}}
\left[\sum_t r^S_t\right]
$$

Therefore the raw episode AMP reward can be larger than `1` when episodes contain multiple steps, while
`AMP/mean_step_reward` remains the direct per-step diagnostic for `reward_type: quad`.

## Eq. 4: Reward Mixing

The AMP paper combines task and style reward:

$$
r_t = w_G r^G_t + w_S r^S_t
$$

The add-on default is the common simplified form:

$$
r_t = r^G_t + \texttt{reward\_coef}\ r^S_t
$$

Code:

```python
return task_rewards + self.reward_coef * amp_rewards
```

See [AMPAddon.combine_rewards](./amp.py#L176).

After this point, PPO computes returns and advantages from `r_t`, not from `r_G_t` alone.

## Policy Samples

The same policy-side feature used for the style reward is stored for the discriminator update:

```python
self.storage.add_transitions(amp_obs.detach(), dones)
```

See [AMPAddon.process_env_step](./amp.py#L136) and
[AMPStorage.add_transitions](../storage/amp_storage.py#L47).

This stored data is the code version of the paper's policy transition samples $b_\pi$.

Paper difference: AMP describes sampling policy transitions from a replay buffer `B`. This implementation keeps the
add-on simpler and samples policy AMP features from the current rollout only.

## Expert Samples

During the discriminator update, expert/reference AMP features are sampled from the environment:

```python
expert_obs = self.expert_sampler(batch_size)
```

See [AMPAddon._sample_expert_observations](./amp.py#L324).

The environment must provide:

```python
def get_amp_expert_obs(self, batch_size: int) -> torch.Tensor:
    ...
```

The returned tensor must have the same feature dimension as `obs["amp"]`:

```text
policy obs shape: (num_envs, amp_obs_dim)
expert obs shape: (batch_size, amp_obs_dim)
```

## Eq. 6: Least-Squares Discriminator Loss

AMP trains `D` as a least-squares discriminator:

$$
L_D =
\mathbb{E}_{\phi_M \sim d^M}
\left[(D(\phi_M) - 1)^2\right]
+
\mathbb{E}_{\phi_\pi \sim d^\pi}
\left[(D(\phi_\pi) + 1)^2\right]
$$

The targets are:

```text
expert/reference transition -> +1
policy/generated transition -> -1
```

Code:

```python
policy_loss = mse_loss(policy_logits, -ones)
expert_loss = mse_loss(expert_logits, +ones)
discriminator_loss = 0.5 * (policy_loss + expert_loss)
```

See [AMPAddon._compute_discriminator_loss](./amp.py#L335).

The `0.5` factor is only a scale convention. It does not change the optimum of the discriminator objective.

## Eq. 8: Gradient Penalty

AMP adds a gradient penalty on expert/reference features:

$$
L_{GP} =
\mathbb{E}_{\phi_M \sim d^M}
\left[\left\|\nabla_{\phi_M} D(\phi_M)\right\|_2^2\right]
$$

Code:

```python
expert_obs = expert_obs.detach().clone().requires_grad_(True)
expert_logits = self.discriminator(expert_obs)
grad = torch.autograd.grad(outputs=expert_logits, inputs=expert_obs, ...)[0]
gradient_penalty = grad.norm(2, dim=-1).pow(2).mean()
```

See [AMPAddon._compute_gradient_penalty](./amp.py#L362).

This implementation also supports `gradient_penalty_tolerance`:

$$
L_{GP} =
\mathbb{E}_{\phi_M \sim d^M}
\left[
\max\left(\left\|\nabla_{\phi_M}D(\phi_M)\right\|_2 - \texttt{tolerance}, 0\right)^2
\right]
$$

With `tolerance = 0.0`, this reduces to the paper-style squared gradient norm.

Coefficient note:

$$
\text{paper coefficient: } \frac{w_{gp}}{2}L_{GP}
$$

$$
\text{code coefficient: } \texttt{gradient\_penalty\_coef}\ L_{GP}
$$

So a paper value $w_{gp}=10$ corresponds to `gradient_penalty_coef = 5.0`.

## Total Discriminator Loss In This Add-on

The code builds the final discriminator loss as:

$$
\begin{aligned}
L_{\text{total}} =
&\ \texttt{loss\_coef}\ L_D
+ \texttt{gradient\_penalty\_coef}\ L_{GP} \\
&+ \texttt{weight\_decay\_coef}\ L_{\text{wd}}
+ \texttt{logit\_weight\_decay\_coef}\ L_{\text{logit-wd}}
\end{aligned}
$$

See [AMPAddon.update](./amp.py#L185).

For the paper-aligned baseline, use:

```yaml
loss_type: mse
reward_type: quad
weight_decay_coef: 0.0
logit_weight_decay_coef: 0.0
```

## Optional Engineering Regularizers

These two losses are not part of the AMP paper Eq. 8 baseline:

- [AMPAddon._compute_weight_decay](./amp.py#L382)
- [AMPAddon._compute_logit_weight_decay](./amp.py#L392)

They are reference-implementation style stabilizers:

$$
L_{\text{wd}} = \sum_{\theta \in D}\|\theta\|_2^2
$$

$$
L_{\text{logit-wd}} = \|W_{\text{last}}\|_2^2
$$

Both coefficients default to `0.0`, so they do not affect classic AMP unless explicitly enabled.

## Important Differences From The Paper

1. The paper writes $D(s, s')$; the implementation uses $D(\phi)$.
   The environment owns the feature map $\Phi$ and passes $\phi = (\Phi(s), \Phi(s'))$ through `obs["amp"]`.

2. The paper samples policy transitions from replay buffer `B`.
   This implementation samples policy AMP features from the latest rollout stored in `AMPStorage`.

3. The paper baseline is `reward_type: quad` and `loss_type: mse`.
   The implementation also exposes `log`, `bce`, and `wasserstein` variants for experiments and ablations.

4. PPO and AMP are intentionally separated.
   PPO updates the actor-critic from the shaped reward; AMP updates only the discriminator.
