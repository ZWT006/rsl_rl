# Distillation Guide

This note maps teacher-student behavior distillation to the current rsl_rl implementation.

## Mental Model

Distillation trains a deployable student policy to imitate a frozen teacher policy:

```text
student observation o_s_t  -> student pi_s produces action for the environment
teacher observation o_T_t  -> teacher pi_T produces supervised target

environment is driven by:
    a_s_t sampled from the student

loss trains:
    mu_s_t = pi_s(o_s_t)  close to  a_T_t = pi_T(o_T_t)
```

Unlike PPO, this update does not use returns, advantages, or policy ratios. Rewards are stored for logging/storage
consistency, but the optimization target is behavior cloning from the teacher.

## Code Map

```text
[DistillationRunner.learn]
|
|-- [Distillation.act]
|   |
|   |-- [StudentTeacher.act]
|   |   `-- sample a_s_t from student pi_s(.|o_s_t) for env.step()
|   |
|   `-- [StudentTeacher.evaluate]
|       `-- compute a_T_t = pi_T(o_T_t) as the supervised target
|
|-- env.step(a_s_t)
|
|-- [Distillation.process_env_step]
|   `-- store o_t, a_s_t, a_T_t, reward_t, done_t
|
`-- [Distillation.update]
    |
    |-- [RolloutStorage.generator]
    |   `-- yields time-major o_t, a_s_t, a_T_t, done_t
    |
    |-- [StudentTeacher.act_inference]
    |   `-- recompute mu_s_t = pi_s(o_s_t) with gradients
    |
    |-- L_BC = loss_fn(mu_s_t, a_T_t)
    |-- accumulate gradient_length steps
    `-- optimizer.step()
```

Links use workspace-relative `#L...` anchors. If VSCode opens the file but ignores the line anchor, use Quick Open with
`path:line`, for example `rsl_rl/algorithms/distillation.py:131`.

- [DistillationRunner.learn](../runners/distillation_runner.py#L57)
- [DistillationRunner._construct_algorithm](../runners/distillation_runner.py#L156)
- [Distillation](./distillation.py#L14)
- [Distillation.init_storage](./distillation.py#L95)
- [Distillation.act](./distillation.py#L107)
- [Distillation.process_env_step](./distillation.py#L118)
- [Distillation.update](./distillation.py#L131)
- [RolloutStorage.generator](../storage/rollout_storage.py#L157)
- [StudentTeacher.act](../modules/student_teacher.py#L122)
- [StudentTeacher.act_inference](../modules/student_teacher.py#L128)
- [StudentTeacher.evaluate](../modules/student_teacher.py#L133)
- [StudentTeacher.load_state_dict](../modules/student_teacher.py#L168)
- [StudentTeacherRecurrent.act](../modules/student_teacher_recurrent.py#L148)
- [StudentTeacherRecurrent.act_inference](../modules/student_teacher_recurrent.py#L155)
- [StudentTeacherRecurrent.evaluate](../modules/student_teacher_recurrent.py#L161)

## Symbols

| Concept | ASCII symbol | Code name | Meaning |
| --- | --- | --- | --- |
| $o^s_t$ | `o_s_t` | student obs groups in `obs` | Observation visible to the student. |
| $o^T_t$ | `o_T_t` | teacher obs groups in `obs` | Observation visible to the teacher. |
| $\pi_s$ | `pi_s` | `self.policy.student` | Student network being trained. |
| $\pi_T$ | `pi_T` | `self.policy.teacher` | Frozen teacher network. |
| $a^s_t$ | `a_s_t` | `actions` / `transition.actions` | Student sampled action used in `env.step()`. |
| $a^T_t$ | `a_T_t` | `privileged_actions` | Teacher target action stored in rollout storage. |
| $\mu^s_t$ | `mu_s_t` | `actions` in `update()` | Student mean action recomputed with gradients. |
| $d_t$ | `done_t` | `dones` | Episode boundary mask. |
| $L_{BC}$ | `L_BC` | `behavior_loss` | Behavior cloning loss between student and teacher actions. |
| $L_{\text{window}}$ | `L_window` | `loss` | Accumulated behavior loss over `gradient_length` steps. |
| $K$ | `K` | `gradient_length` | Truncated BPTT / accumulation length. |
| $h^s_t$ | `h_s_t` | student hidden state | Recurrent student memory state. |
| $h^T_t$ | `h_T_t` | teacher hidden state | Optional recurrent teacher memory state. |

## Teacher And Student Observations

`StudentTeacher` splits the observation dictionary by configured groups:

```python
student_obs = self.get_student_obs(obs)
teacher_obs = self.get_teacher_obs(obs)
```

See [StudentTeacher.get_student_obs](../modules/student_teacher.py#L139) and
[StudentTeacher.get_teacher_obs](../modules/student_teacher.py#L145).

The student usually receives deployable observations. The teacher may receive privileged observations, such as state
estimates or extra simulator information.

## Rollout Path

`Distillation.act()` stores both sides of the supervised pair:

```python
self.transition.actions = self.policy.act(obs).detach()
self.transition.privileged_actions = self.policy.evaluate(obs).detach()
```

See [Distillation.act](./distillation.py#L107).

The environment receives the student-sampled action $a^s_t$, not the teacher action:

$$
a^s_t \sim \pi_s(\cdot|o^s_t)
$$

The teacher target is deterministic in this implementation:

$$
a^T_t = \pi_T(o^T_t)
$$

The pair $(o_t, a^s_t, a^T_t, d_t)$ is stored by
[RolloutStorage.add_transitions](../storage/rollout_storage.py#L71). For distillation, `privileged_actions` means
"teacher target action", not a privileged student action.

## Behavior Cloning Loss

During update, the student action is recomputed with gradients:

```python
actions = self.policy.act_inference(obs)
behavior_loss = self.loss_fn(actions, privileged_actions)
```

See [Distillation.update](./distillation.py#L131).

For `loss_type: mse`, the objective is:

$$
L_{BC} =
\left\|\mu^s_t - a^T_t\right\|_2^2
$$

For `loss_type: huber`, `L_BC` uses Huber loss instead.

The teacher target is detached during rollout:

```python
self.transition.privileged_actions = self.policy.evaluate(obs).detach()
```

So the optimizer updates the student side of `StudentTeacher`, not the teacher.

## Gradient Windows

The code accumulates `gradient_length` time steps before a backward pass:

$$
L_{\text{window}} =
\sum_{i=0}^{K-1} L_{BC,t+i}
$$

where:

```text
K = gradient_length
```

This is useful for recurrent students because it defines the truncated BPTT window. For feed-forward students it acts
like loss accumulation over multiple time steps.

Implementation note: the current code steps the optimizer when `cnt % gradient_length == 0`. Choose `gradient_length`
so it divides `num_learning_epochs * num_steps_per_env` if you want every accumulated loss window to end in an optimizer
step.

## Recurrent Distillation

`StudentTeacherRecurrent` keeps recurrent memories for the student and optionally the teacher:

```text
h_s_t -> memory_s
h_T_t -> memory_t, only when teacher_recurrent=True
```

At the start of each epoch, [Distillation.update](./distillation.py#L131) restores `last_hidden_states`. During the
rollout replay, hidden states are reset on done environments:

```python
self.policy.reset(dones.view(-1))
self.policy.detach_hidden_states(dones.view(-1))
```

This prevents gradients from crossing episode boundaries.

## Teacher Loading

Distillation requires a loaded teacher. The runner checks this before training:

```python
if not self.alg.policy.loaded_teacher:
    raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")
```

See [DistillationRunner.learn](../runners/distillation_runner.py#L57).

`StudentTeacher.load_state_dict()` supports two paths:

1. Loading an RL actor checkpoint into the teacher.
2. Loading a previous distillation checkpoint with both student and teacher parameters.

See [StudentTeacher.load_state_dict](../modules/student_teacher.py#L168).

## Differences From PPO

1. No returns, advantages, value function, entropy bonus, or PPO ratio are used.
2. Rewards do not define the training objective; they are logged and stored for rollout consistency.
3. The teacher is kept in eval mode while the student is trained.
4. The environment is driven by the student, so the collected observation distribution matches the deployed policy.
5. `privileged_actions` in storage means teacher target actions for this training type.

