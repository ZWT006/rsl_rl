# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn

from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class Distillation:
    """Teacher-student behavior distillation.

    Paper-to-code outline:
        DistillationRunner.learn()
        |-- Distillation.act()
        |   |-- student samples a_s_t from pi_s(.|o_s_t) for environment interaction
        |   `-- teacher computes a_T_t = pi_T(o_T_t) as the supervised target
        |-- env.step(a_s_t)
        |-- Distillation.process_env_step()
        |   `-- store o_t, a_s_t, a_T_t, done_t in RolloutStorage
        `-- Distillation.update()
            |-- RolloutStorage.generator() yields time-major steps
            |-- student recomputes mu_s_t = pi_s(o_s_t) with gradients
            |-- L_BC = loss_fn(mu_s_t, a_T_t)
            |-- accumulate gradient_length steps for truncated BPTT
            `-- optimizer.step()

    The environment is driven by the student's sampled actions, while the loss trains the student mean action to match
    the frozen teacher action computed from the teacher observation group.
    """

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        optimizer="adam",
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # Device and distributed parameters. In multi-GPU mode, each rank collects rollout samples and gradients are
        # averaged before the student update.
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # The StudentTeacher module owns both networks. The teacher is frozen/eval; the optimizer updates the student
        # parameters and optional action-noise parameters.
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later

        # initialize the optimizer
        self.optimizer = resolve_optimizer(optimizer)(self.policy.parameters(), lr=learning_rate)

        # initialize the transition
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # Distillation hyper-parameters. gradient_length controls how many time steps are accumulated before one
        # backward pass, which is especially important for recurrent students as truncated BPTT length.
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # initialize the loss function
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        self.num_updates = 0

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        # Create a distillation rollout buffer. It stores the student action used in the environment and the
        # privileged teacher action used as the supervised target.
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs):
        # a_s_t drives the environment. It is sampled from the student policy so the rollout distribution matches the
        # deployed student, not the teacher.
        self.transition.actions = self.policy.act(obs).detach()
        # a_T_t is the frozen teacher target. The teacher may use privileged observation groups that the student does
        # not receive.
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        # Store o_t before env.step(); rewards and dones arrive in process_env_step().
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        # Update only the student observation normalizer. The teacher normalizer is loaded with the teacher and kept in
        # eval mode by StudentTeacher.train().
        self.policy.update_normalization(obs)

        # Rewards are stored only for logging/storage consistency. The supervised loss uses a_T_t, not returns.
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        """Optimize the student with behavior cloning loss L_BC over the collected rollout."""
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            # Recurrent students continue from the hidden state saved after the previous update. Feed-forward students
            # return None and this reset is a no-op.
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():

                # Recompute mu_s_t = pi_s(o_s_t) with gradients. This differs from act(), which sampled a_s_t under
                # torch.inference_mode() during rollout.
                actions = self.policy.act_inference(obs)

                # Behavior cloning loss: L_BC = ||mu_s_t - a_T_t||^2 for mse, or Huber(mu_s_t, a_T_t).
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # Accumulate losses across gradient_length time steps before a backward pass. This gives recurrent
                # students a truncated BPTT window and gives feed-forward students a larger effective batch.
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # Gradient step for the accumulated L_BC window.
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # Reset recurrent hidden states at episode boundaries and detach them to stop gradients crossing
                # completed episodes.
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
