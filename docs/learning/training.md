# Training a policy

```bash
uav-sim train hover
```

```text
Training 'hover'
────────────────
  method      augmented random search
  policy      linear
  budget      120 iterations

  ██████████████████████████████████ iter  120/120  return     412.7

  trained in  518.3s (14232 episodes)
  best return 412.7

Held-out evaluation
───────────────────
  mean_return          389.42
  mean_length          468.10
  pct_time_limit       88.00
  pct_ground_contact   12.00

  saved policies/hover.npz
```

From Python:

```python
from uav_sim.gym import train, evaluate

result = train("hover", iterations=120, seed=0, save_path="policies/hover.npz")
print(evaluate("hover", result.policy, episodes=25))
```

## Why derivative-free

Training a drone to fly should not require a deep-learning stack. These
tasks are low-dimensional, the policy is small, and evaluation is cheap
enough that estimating a search direction from rollouts is competitive with
computing a gradient — while keeping the whole trainer readable and the
install to three dependencies.

Two optimisers ship, both pure NumPy.

## Augmented Random Search (default)

> H. Mania, A. Guy, B. Recht, "Simple random search provides a competitive
> approach to reinforcement learning", NeurIPS 2018.

Each iteration:

1. Sample `directions` random perturbations of the parameter vector.
2. Score the policy at **plus and minus** each one (antithetic sampling).
3. Keep the `top_directions` whose pair produced the largest response.
4. Step along their return-weighted average.

```python
gradient = sum((r_plus[i] - r_minus[i]) * delta[i] for i in chosen)
parameters += (step_size / (top_directions * reward_std)) * gradient
```

Dividing by the standard deviation of the surviving returns is what makes
one `step_size` work across tasks whose returns differ by orders of
magnitude. Without it, the learning rate has to be retuned for every reward
scale.

Ranking directions by `max(r_plus, r_minus)` means a direction only
survives if perturbing along it *matters* — in either sign.

## Cross-Entropy Method

> R. Y. Rubinstein, D. P. Kroese, *The Cross-Entropy Method*, Springer, 2004.

Sample a population, keep the best fraction, refit a diagonal Gaussian,
repeat. The variance floor is the standard "CEM with noise" fix
(Szita & Lőrincz, 2006) that stops the population collapsing onto the first
decent solution it finds.

```bash
uav-sim train hover --optimizer cem --population 40
```

CEM has to model a distribution over *every* parameter, so its sample cost
grows with dimension. ARS estimates a direction and gets a usable step from
a couple of dozen paired rollouts regardless of parameter count. On these
tasks ARS is the better default by a wide margin; CEM is included because
it is instructive and because it behaves differently on rugged landscapes.

## The policy

```python
from uav_sim.gym.policy import MLPPolicy

policy = MLPPolicy(observation_size=18, action_size=4, hidden_sizes=())
policy.parameter_count      # 76
```

Linear by default — `hidden_sizes=()`. On body-frame observations a linear
policy is structurally a PD controller, which is the shape of the solution
these tasks want. See
[matching the policy class to the optimiser](/learning/design-notes#_4-match-the-policy-class-to-the-optimiser)
for the measurement showing 76 parameters suffice.

The **output layer is zero-initialised**, so the initial policy emits
exactly zero actions — which, because the action spaces are centred on
equilibrium, means it starts out flying. This is not a refinement; it is
the difference between a search with signal and one without.

For a deeper policy:

```bash
uav-sim train trajectory --hidden 64 64
```

Expect to need more iterations. Hidden layers are worth it when the task is
genuinely non-linear — trajectory tracking with a lead term, for instance.

## Hyperparameters

```python
from uav_sim.gym.train import TrainConfig

config = TrainConfig(
    optimizer="ars",
    iterations=120,
    directions=16,            # perturbations per iteration; each costs 2 episodes
    top_directions=8,         # how many contribute to the update
    step_size=0.02,
    noise=0.03,               # finite-difference perturbation size
    episodes_per_candidate=8,
    fixed_task_seeds=True,
    hidden_sizes=(),
    seed=0,
)
```

The ones that actually matter, in order:

**`fixed_task_seeds`** — score every candidate on the same episodes.
Leaving this off multiplies the iterations needed by roughly an order of
magnitude, because the finite differences end up measuring which spawns a
candidate drew rather than how good it is.

**`episodes_per_candidate`** — with heavy spawn randomisation, too few
episodes makes every score noise. Eight is a reasonable floor for the
quadrotor tasks.

**`noise`** — too small and the returns of `+δ` and `−δ` are
indistinguishable; too large and the finite difference stops approximating
a gradient. 0.03–0.05 works across these tasks.

**`step_size`** — the usual trade-off. The return-standard-deviation
normalisation makes it far less sensitive than a raw learning rate.

## Measured learning

Hover, ARS, linear policy, `directions=8`, `episodes_per_candidate=8`,
`seed=0`:

| Iteration | Training return | Held-out return |
|---|---|---|
| 0 | 21.7 | 23.6 |
| 10 | 30.8 | 25.4 |
| 20 | 212.8 | 92.2 |
| 30 | 316.3 | 240.2 |

For scale: an untrained zero policy scores about 2 and survives ~140 steps
before drifting out of the volume; a hand-written cascaded controller
scores 638 and holds station for the full 500.

::: tip Reproduce it
```bash
uav-sim train hover --iterations 100 --directions 8 --episodes 8 --seed 0
```
Runs in a few minutes on one core. Every number in the table above came
from that command.
:::

## Watching a policy fly

```bash
uav-sim play hover --policy policies/hover.npz --episodes 5 --gif hover.gif
```

```python
from uav_sim.gym import make
from uav_sim.gym.policy import MLPPolicy
from uav_sim.gym.render import render_episode
from uav_sim.gym.train import rollout

env = make("hover", seed=0)
policy = MLPPolicy.load("policies/hover.npz")

trajectories = []
for episode in range(5):
    rollout(env, policy, seed=episode)
    trajectories.append(env.trajectory)

render_episode(env, trajectories, "hover.gif", title="learned hover")
```

Learning curves:

```python
from uav_sim.gym.render import render_learning_curve
render_learning_curve(result.history, "curve.png")
```

## Using an external RL library

```bash
pip install "uav-sim[gym]" stable-baselines3
```

```python
import gymnasium as gym
import uav_sim.gym          # registers the environments on import
from stable_baselines3 import PPO

env = gym.make("uav_sim/Hover-v0")
model = PPO("MlpPolicy", env, verbose=1).learn(500_000)
```

The environments already follow the Gymnasium contract, including the
`terminated` / `truncated` split, so nothing needs wrapping.

## See also

- [Environments](/learning/environments)
- [Designing a task](/learning/design-notes)
