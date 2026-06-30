# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn

from rsl_rl.modules import AMPDiscriminator
from rsl_rl.storage import AMPStorage
from rsl_rl.utils import resolve_optimizer


class AMPAddon:
    """Adversarial Motion Prior add-on for on-policy training.

    Paper-to-code outline:
        OnPolicyRunner.learn()
        |-- env.step()
        |   |-- returns task reward r_G_t and obs[obs_key] = phi_pi
        |   |-- phi_pi is the policy transition feature, usually (Phi(s_t), Phi(s_{t+1}))
        |-- AMPAddon.process_env_step()
        |   |-- AMPAddon.compute_rewards()             # Eq. 7: d_t = D_phi_pi, r_S_t = r_S(d_t)
        |   |-- AMPAddon.combine_rewards()             # Eq. 4: r_t = r_G_t + w_S * r_S_t
        |   `-- AMPStorage.add_transitions()           # store phi_pi for the D update
        |-- PPO.update()
        |   `-- learns from the shaped reward; PPO does not backpropagate through D
        `-- AMPAddon.update()
            |-- AMPStorage.mini_batch_generator()      # policy samples b_pi
            |-- AMPAddon._sample_expert_observations() # reference samples b_M
            |-- AMPAddon._compute_discriminator_loss() # Eq. 6: L_D from D_phi_pi and D_phi_M
            |-- AMPAddon._compute_gradient_penalty()   # Eq. 8: L_GP on phi_M
            `-- AMPStorage.clear()

    The add-on consumes policy-side AMP observations from the environment and expert-side AMP observations from an
    environment sampler. PPO remains unaware of AMP: it only receives the shaped reward returned by this add-on.
    """

    def __init__(
        self,
        env,
        obs,
        cfg: dict,
        num_steps_per_env: int,
        num_mini_batches: int,
        num_learning_epochs: int,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        self.device = device
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs

        # Multi-GPU parameters
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Keep the config local to the add-on so the runner and PPO algorithm do not need AMP-specific keys.
        cfg = copy.deepcopy(cfg)
        discriminator_cfg = cfg.pop("discriminator", {})

        # The environment owns the feature map Phi. rsl_rl only needs the key for policy samples and a sampler for
        # reference-motion samples, keeping AMP independent of a specific humanoid or motion-library implementation.
        self.obs_key = cfg.pop("obs_key", "amp")
        self.expert_sampler_name = self._pop_first(cfg, ["expert_sampler_name", "expert_sampler"], "get_amp_expert_obs")
        self.expert_sampler = self._resolve_expert_sampler(env, self.expert_sampler_name)

        # AMP combines task reward r_G_t and learned style reward r_S_t. The default "add" mode maps to
        # r_t = r_G_t + w_S * r_S_t, with reward_coef playing the role of w_S.
        self.reward_coef = self._pop_first(cfg, ["reward_coef", "amp_coef"], 1.0)
        self.reward_combine = cfg.pop("reward_combine", "add")
        self.reward_type = cfg.pop("reward_type", "quad")
        self.reward_eps = cfg.pop("reward_eps", 1.0e-6)

        # Discriminator hyper-parameters follow the AMP paper defaults where possible: least-squares loss with
        # optional gradient penalty on expert features. BCE and Wasserstein variants are kept as debug/ablation knobs.
        self.loss_type = self._pop_discriminator_cfg(discriminator_cfg, cfg, "loss_type", "mse")
        self.loss_coef = self._pop_discriminator_cfg(discriminator_cfg, cfg, "loss_coef", 1.0)
        self.gradient_penalty_coef = self._pop_discriminator_cfg(
            discriminator_cfg, cfg, "gradient_penalty_coef", 0.0
        )
        self.gradient_penalty_tolerance = self._pop_discriminator_cfg(
            discriminator_cfg, cfg, "gradient_penalty_tolerance", 0.0
        )
        # L_wd and L_logit_wd are optional engineering regularizers from Wasabi/ProtoMotions-style AMP variants. They
        # are not part of the original AMP Eq. 8 objective and default to zero, so the paper-aligned loss is L_D plus
        # gradient_penalty_coef * L_GP.
        self.weight_decay_coef = self._pop_discriminator_cfg(discriminator_cfg, cfg, "weight_decay_coef", 0.0)
        self.logit_weight_decay_coef = self._pop_discriminator_cfg(
            discriminator_cfg, cfg, "logit_weight_decay_coef", 0.0
        )
        self.max_grad_norm = self._pop_discriminator_cfg(discriminator_cfg, cfg, "max_grad_norm", None)

        learning_rate = self._pop_discriminator_cfg(discriminator_cfg, cfg, "learning_rate", 1.0e-4)
        optimizer_name = self._pop_discriminator_cfg(discriminator_cfg, cfg, "optimizer", "adam")
        optimizer_kwargs = self._pop_discriminator_cfg(discriminator_cfg, cfg, "optimizer_kwargs", {})
        optimizer_kwargs = (optimizer_kwargs or {}).copy()
        learning_rate = optimizer_kwargs.pop("lr", learning_rate)

        self._validate_cfg(cfg)
        self._validate_reward_settings()

        # The AMP observation must already be a flat transition feature. This avoids hidden assumptions about the
        # simulator state layout, frame stacking, or whether Phi includes s_t, s_{t+1}, or both.
        if self.obs_key not in obs:
            raise ValueError(
                f"AMP observation '{self.obs_key}' not found in environment observations. "
                f"Available observations: {list(obs.keys())}"
            )
        if len(obs[self.obs_key].shape) != 2:
            raise ValueError(
                f"AMP observation '{self.obs_key}' must be a 2D tensor of shape (num_envs, num_obs). "
                f"Received shape: {tuple(obs[self.obs_key].shape)}"
            )

        num_amp_obs = obs[self.obs_key].shape[-1]
        self.discriminator = AMPDiscriminator(num_amp_obs, **discriminator_cfg).to(self.device)
        self.optimizer = resolve_optimizer(optimizer_name)(
            self.discriminator.parameters(), lr=learning_rate, **optimizer_kwargs
        )
        self.storage = AMPStorage(env.num_envs, num_steps_per_env, [num_amp_obs], device=self.device)
        self.learning_rate = learning_rate

        self.last_amp_rewards = None

    def process_env_step(self, obs, rewards: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute r_S_t, form r_t, and store phi_pi for the Eq. 6/8 discriminator update."""
        amp_obs = obs[self.obs_key]

        # PPO treats the AMP reward as scalar feedback. We therefore compute it without gradients and keep the
        # discriminator update fully separated from the actor-critic update.
        amp_rewards = self.compute_rewards(amp_obs)
        shaped_rewards = self.combine_rewards(rewards, amp_rewards)

        # Store policy-side features phi_pi for the next discriminator update. The detach is intentional: AMP updates
        # D from rollout samples, but it does not backpropagate through the environment or the policy rollout tensors.
        self.storage.add_transitions(amp_obs.detach(), dones)
        self.last_amp_rewards = amp_rewards.detach()
        return shaped_rewards, amp_rewards

    @torch.no_grad()
    def compute_rewards(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """Map policy discriminator score d_t = D_phi_pi = D(phi_pi) to style reward r_S_t, as in AMP Eq. 7."""
        was_training = self.discriminator.training
        self.discriminator.eval()
        # logits is d_t in Eq. 7. During rollout, amp_obs is phi_pi from obs[obs_key].
        logits = self.discriminator(amp_obs)
        if was_training:
            self.discriminator.train()

        if self.reward_type == "quad":
            # Least-squares AMP reward from the paper:
            # r_S_t = max(0, 1 - 0.25 * (d_t - 1)^2), bounded in [0, 1] near the expert target.
            rewards = torch.clamp(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0)
        elif self.reward_type == "log":
            # GAIL-style reward: r_S_t = -log(1 - sigmoid(d_t)). Useful for comparison with BCE training.
            prob = torch.sigmoid(logits).clamp(self.reward_eps, 1.0 - self.reward_eps)
            rewards = -torch.log(1.0 - prob)
        elif self.reward_type == "wasserstein":
            # Wasserstein-style critic reward. This is unbounded and should usually be paired with careful scaling.
            rewards = logits
        else:
            raise ValueError(f"Unsupported AMP reward type: {self.reward_type}")
        return rewards.squeeze(-1)

    def combine_rewards(self, task_rewards: torch.Tensor, amp_rewards: torch.Tensor) -> torch.Tensor:
        """Combine task reward r_G_t and style reward r_S_t before PPO stores r_t, as in AMP Eq. 4."""
        amp_rewards = self._match_reward_shape(amp_rewards, task_rewards)
        if self.reward_combine == "add":
            return task_rewards + self.reward_coef * amp_rewards
        if self.reward_combine == "blend":
            return (1.0 - self.reward_coef) * task_rewards + self.reward_coef * amp_rewards
        raise ValueError(f"Unsupported AMP reward combine mode: {self.reward_combine}")

    def update(self) -> dict[str, float]:
        """Update D with policy samples b_pi and reference-motion samples b_M."""
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError(
                "AMP storage is not full. "
                f"Expected {self.storage.num_transitions_per_env} steps, got {self.storage.step}."
            )

        # These logs mirror the two parts of the adversarial objective and the discriminator score balance. A healthy
        # run usually has expert logits above policy logits without either side saturating permanently.
        mean_losses = {
            "AMP/discriminator_loss": 0.0,
            "AMP/expert_loss": 0.0,
            "AMP/policy_loss": 0.0,
            "AMP/gradient_penalty": 0.0,
            "AMP/weight_decay": 0.0,
            "AMP/logit_weight_decay": 0.0,
            "AMP/expert_logit": 0.0,
            "AMP/policy_logit": 0.0,
        }

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        num_updates = self.num_mini_batches * self.num_learning_epochs
        for minibatch in generator:
            # b_pi comes from the on-policy rollout buffer; b_M comes from the reference-motion dataset. Both must use
            # the same feature map Phi and dimensionality.
            policy_obs = minibatch.observations
            expert_obs = self._sample_expert_observations(policy_obs.shape[0])

            # If enabled, normalize discriminator inputs with both distributions so the normalizer does not drift
            # toward only policy samples or only expert samples.
            self.discriminator.update_normalization(torch.cat([policy_obs, expert_obs], dim=0))

            # policy_logits is D_phi_pi = D(phi_pi); expert_logits is D_phi_M = D(phi_M).
            policy_logits = self.discriminator(policy_obs)
            expert_logits = self.discriminator(expert_obs)

            # Main adversarial loss plus optional stabilizers. With loss_type="mse", discriminator_loss is L_D and
            # gradient_penalty is L_GP. In paper notation, gradient_penalty_coef plays the role of w_gp_over_2.
            discriminator_loss, policy_loss, expert_loss = self._compute_discriminator_loss(
                policy_logits, expert_logits
            )
            gradient_penalty = self._compute_gradient_penalty(expert_obs)
            # L_wd and L_logit_wd are disabled by default. They are kept as explicit, logged losses instead of optimizer
            # weight_decay so experiments can turn them on without changing the AMP paper baseline.
            weight_decay = self._compute_weight_decay()
            logit_weight_decay = self._compute_logit_weight_decay()

            # L_total = loss_coef * L_D + gradient_penalty_coef * L_GP
            #         + weight_decay_coef * L_wd + logit_weight_decay_coef * L_logit_wd.
            loss = (
                self.loss_coef * discriminator_loss
                + self.gradient_penalty_coef * gradient_penalty
                + self.weight_decay_coef * weight_decay
                + self.logit_weight_decay_coef * logit_weight_decay
            )

            self._gradient_step(loss)

            mean_losses["AMP/discriminator_loss"] += discriminator_loss.item() / num_updates
            mean_losses["AMP/expert_loss"] += expert_loss.item() / num_updates
            mean_losses["AMP/policy_loss"] += policy_loss.item() / num_updates
            mean_losses["AMP/gradient_penalty"] += gradient_penalty.item() / num_updates
            mean_losses["AMP/weight_decay"] += weight_decay.item() / num_updates
            mean_losses["AMP/logit_weight_decay"] += logit_weight_decay.item() / num_updates
            mean_losses["AMP/expert_logit"] += expert_logits.mean().item() / num_updates
            mean_losses["AMP/policy_logit"] += policy_logits.mean().item() / num_updates

        self.storage.clear()
        return mean_losses

    def state_dict(self) -> dict:
        return {
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict, load_optimizer: bool = True):
        self.discriminator.load_state_dict(state_dict["discriminator_state_dict"])
        if load_optimizer and "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def train(self):
        self.discriminator.train()

    def eval(self):
        self.discriminator.eval()

    def broadcast_parameters(self):
        model_params = [self.discriminator.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.discriminator.load_state_dict(model_params[0])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.discriminator.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.discriminator.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    """
    Helper functions.
    """

    @staticmethod
    def _pop_first(cfg: dict, keys: list[str], default):
        for key in keys:
            if key in cfg:
                return cfg.pop(key)
        return default

    @staticmethod
    def _pop_discriminator_cfg(discriminator_cfg: dict, cfg: dict, key: str, default):
        if key in discriminator_cfg:
            return discriminator_cfg.pop(key)
        return cfg.pop(key, default)

    def _resolve_expert_sampler(self, env, sampler_name: str) -> Callable[[int], torch.Tensor]:
        if hasattr(env, sampler_name):
            sampler = getattr(env, sampler_name)
        elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, sampler_name):
            sampler = getattr(env.unwrapped, sampler_name)
        else:
            raise AttributeError(
                f"AMP expert sampler '{sampler_name}' was not found on env or env.unwrapped. "
                "Please add a method such as env.get_amp_expert_obs(batch_size)."
            )
        if not callable(sampler):
            raise TypeError(f"AMP expert sampler '{sampler_name}' is not callable.")
        return sampler

    def _sample_expert_observations(self, batch_size: int) -> torch.Tensor:
        """Sample phi_M from the environment's reference-motion dataset sampler."""
        expert_obs = self.expert_sampler(batch_size)
        expert_obs = expert_obs.to(self.device)
        if len(expert_obs.shape) != 2 or expert_obs.shape[-1] != self.discriminator.num_inputs:
            raise ValueError(
                "AMP expert sampler returned observations with incompatible shape. "
                f"Expected (batch_size, {self.discriminator.num_inputs}), got {tuple(expert_obs.shape)}."
            )
        return expert_obs

    def _compute_discriminator_loss(
        self, policy_logits: torch.Tensor, expert_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the discriminator classification loss L_D.

        With loss_type="mse", this is AMP Eq. 6 after replacing D(s, s') with D(phi):
            L_M = E_M[(D_phi_M - 1)^2]
            L_pi = E_pi[(D_phi_pi + 1)^2]
            L_D = 0.5 * (L_pi + L_M)
        """
        if self.loss_type == "mse":
            # Least-squares AMP target convention: D_phi_M -> +1 for motion data, D_phi_pi -> -1 for policy data.
            policy_loss = nn.functional.mse_loss(policy_logits, -torch.ones_like(policy_logits))
            expert_loss = nn.functional.mse_loss(expert_logits, torch.ones_like(expert_logits))
        elif self.loss_type == "bce":
            # Standard GAIL convention after sigmoid: expert -> 1, policy -> 0.
            policy_loss = nn.functional.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
            expert_loss = nn.functional.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
        elif self.loss_type == "wasserstein":
            # Critic convention: maximize D(expert) - D(policy), so the minimization loss is D(policy) - D(expert).
            policy_loss = policy_logits.mean()
            expert_loss = -expert_logits.mean()
        else:
            raise ValueError(f"Unsupported AMP discriminator loss type: {self.loss_type}")
        discriminator_loss = 0.5 * (policy_loss + expert_loss)
        return discriminator_loss, policy_loss, expert_loss

    def _compute_gradient_penalty(self, expert_obs: torch.Tensor) -> torch.Tensor:
        """Compute L_GP, the expert-feature gradient penalty term from AMP Eq. 8."""
        if self.gradient_penalty_coef <= 0.0:
            return torch.zeros((), device=self.device)

        # AMP applies the gradient penalty on expert observation features, not on the full simulator state:
        # L_GP = E_M[(max(||grad_phi_M D_phi_M||_2 - tolerance, 0))^2].
        # The default tolerance of 0.0 recovers the paper's nonzero-gradient penalty on the data manifold.
        expert_obs = expert_obs.detach().clone().requires_grad_(True)
        expert_logits = self.discriminator(expert_obs)
        grad = torch.autograd.grad(
            outputs=expert_logits,
            inputs=expert_obs,
            grad_outputs=torch.ones_like(expert_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return torch.clamp(grad.norm(2, dim=-1) - self.gradient_penalty_tolerance, min=0.0).pow(2).mean()

    def _compute_weight_decay(self) -> torch.Tensor:
        """Compute L_wd, an optional full-discriminator L2 regularizer."""
        if self.weight_decay_coef <= 0.0:
            return torch.zeros((), device=self.device)

        weight_decay = torch.zeros((), device=self.device)
        for param in self.discriminator.parameters():
            weight_decay = weight_decay + torch.sum(torch.square(param))
        return weight_decay

    def _compute_logit_weight_decay(self) -> torch.Tensor:
        """Compute L_logit_wd, an optional final-score-layer L2 regularizer."""
        if self.logit_weight_decay_coef <= 0.0:
            return torch.zeros((), device=self.device)

        return torch.sum(torch.square(self.discriminator.logit_layer_weights()))

    def _gradient_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        if self.max_grad_norm is not None:
            # This is a debug safety valve for unusually sharp discriminator updates.
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def _match_reward_shape(self, amp_rewards: torch.Tensor, task_rewards: torch.Tensor) -> torch.Tensor:
        if task_rewards.dim() == 1:
            return amp_rewards.view(-1)
        if task_rewards.dim() == 2 and task_rewards.shape[-1] == 1:
            return amp_rewards.view(-1, 1)
        raise ValueError(
            "AMP reward shaping currently supports task rewards with shape (num_envs,) or (num_envs, 1). "
            f"Received shape: {tuple(task_rewards.shape)}"
        )

    def _validate_cfg(self, cfg: dict):
        if cfg:
            print("AMPAddon got unexpected config entries, which will be ignored: " + str(list(cfg.keys())))

    def _validate_reward_settings(self):
        if self.reward_combine not in ["add", "blend"]:
            raise ValueError("AMP reward_combine must be either 'add' or 'blend'.")
        if self.reward_type not in ["quad", "log", "wasserstein"]:
            raise ValueError("AMP reward_type must be one of: 'quad', 'log', 'wasserstein'.")
        if self.loss_type not in ["mse", "bce", "wasserstein"]:
            raise ValueError("AMP discriminator loss_type must be one of: 'mse', 'bce', 'wasserstein'.")
