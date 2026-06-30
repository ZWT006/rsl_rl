# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.networks import EmpiricalNormalization, MLP


class AMPDiscriminator(nn.Module):
    """Discriminator network used by adversarial motion priors.

    The network outputs one unconstrained score D(phi). For the default least-squares AMP loss, expert reference
    features are trained toward +1 and policy rollout features are trained toward -1. A sigmoid is intentionally not
    part of this module because the LS-AMP reward and loss both operate directly on the raw score.
    """

    is_recurrent = False

    def __init__(
        self,
        num_inputs: int,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        activation: str = "relu",
        normalize_input: bool = False,
        normalizer_kwargs: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "AMPDiscriminator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_inputs = num_inputs
        self.normalize_input = normalize_input
        if normalize_input:
            # Input normalization is optional because many motion-library pipelines already normalize AMP features.
            # When enabled, the add-on updates it with both policy and expert samples before each discriminator step.
            self.normalizer = EmpiricalNormalization(num_inputs, **(normalizer_kwargs or {}))
        else:
            self.normalizer = nn.Identity()

        self.trunk = MLP(num_inputs, 1, hidden_dims, activation)
        print(f"AMP Discriminator MLP: {self.trunk}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs is the AMP feature phi, normally a compact transition descriptor built by the environment.
        obs = self.normalizer(obs)
        return self.trunk(obs)

    @torch.no_grad()
    def update_normalization(self, obs: torch.Tensor):
        """Update empirical input statistics without adding this operation to the discriminator graph."""
        if self.normalize_input:
            self.normalizer.update(obs)

    def logit_layer_weights(self) -> torch.Tensor:
        """Return the final linear layer weights for optional logit regularization."""
        for module in reversed(self.trunk):
            if isinstance(module, nn.Linear):
                return module.weight
        raise RuntimeError("AMP discriminator has no linear layer.")
