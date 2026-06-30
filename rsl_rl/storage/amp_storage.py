# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import namedtuple

import torch


class AMPStorage:
    """Rollout storage for policy-side AMP observations.

    AMP needs policy samples b_pi from the latest on-policy rollout when updating the discriminator. This buffer stores
    only the AMP feature phi_pi, not actions, values, returns, or full actor observations, because PPO already owns
    those tensors in its rollout storage.

    In the AMP paper, policy transitions are sampled from a replay buffer B. This implementation keeps the add-on
    simple and samples b_pi from the current rollout only.
    """

    MiniBatch = namedtuple("MiniBatch", ["observations", "dones"])

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        observation_shape: list[int] | tuple[int, ...],
        device: str = "cpu",
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.observation_shape = observation_shape

        self.observations = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *observation_shape,
            device=self.device,
        )
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.step = 0

    def add_transitions(self, observations: torch.Tensor, dones: torch.Tensor):
        # observations are policy-side AMP transition features emitted by the environment after env.step().
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("AMP rollout buffer overflow! You should call clear() before adding new transitions.")

        self.observations[self.step].copy_(observations)
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        # The discriminator update pairs these shuffled policy samples with freshly sampled expert samples of the same
        # batch size. Dropped tail samples match the PPO storage convention when the batch is not divisible.
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size == 0:
            raise ValueError(
                f"AMP mini-batch size is zero. Batch size {batch_size} is smaller than {num_mini_batches} mini-batches."
            )
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        dones = self.dones.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield AMPStorage.MiniBatch(observations[batch_idx], dones[batch_idx])
