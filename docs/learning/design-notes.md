# Designing a task

Four choices in these environments account for essentially all of the
difference between a task that trains in minutes and one that never leaves
the ground. None of them is about the algorithm.

They are written up here because they generalise: if you are building your
own UAV learning task, these are the places it will silently fail.

## 1. Centre the action space on equilibrium

The quadrotor's thrust channel is mapped so that **zero means hover
thrust**, not zero thrust:

```python
thrust = hover_thrust * (1.0 + thrust_range * action[0])
```

The fixed-wing tasks do the same thing against the
[trim solution](/vehicles/trim): actions are deltas from trim, so a zero
action flies straight and level.

Combined with a zero-initialised output layer, this means an untrained
policy *is already flying*. It explores around flight rather than having to
first discover that thrust exists.

### What happens without it

With a randomly initialised output layer, every candidate saturates its
tanh, commands full torque, and tumbles within a second:

```text
Random output layer, CEM, 30 generations:
  eval return  -3.0    mean episode length 21 steps    100% tumbled
```

Every candidate scored the same, because every candidate crashed the same
way. There was no signal for the optimiser to follow — the problem was not
that learning was slow, it was that there was nothing to learn *from*.

## 2. Observe in the frame you act in

The action is a **body** wrench. So the observation gives position error
and velocity in the **body** frame:

```python
rotation = self.vehicle.rotation_matrix(*state[3:6])
position_error_body = rotation.T @ (self.goal - state[:3])
velocity_body = rotation.T @ state[6:9]
```

Handing the policy a world-frame error instead means the network must
internally learn the world-to-body rotation before it can act on anything —
and it has to learn that from reward alone, in the same gradient steps it is
using to learn control.

The rotation is already known exactly. Applying it costs one matrix
multiply and removes the hardest part of the mapping.

Attitude enters as the **full rotation matrix**, not Euler angles. Euler
angles wrap at ±π, so two numerically distant inputs describe the same
attitude; the nine matrix entries are smooth and unique everywhere.

## 3. Score every candidate on the same episodes

These environments randomise the spawn heavily — position, attitude,
velocity and yaw. With fresh episode seeds each iteration, the gap between
two candidates' scores is dominated by which starting conditions they
happened to draw.

Finite differences over that measure luck, not gradient.

```python
task_seeds = [rng.integers(...) for _ in range(episodes_per_candidate)]
# reused for the whole run
```

### What this was worth

Same optimiser, same policy, same budget — only the seed handling changed:

```text
Fresh seeds each iteration:   holdout return  ~31 after 60 iterations
Fixed seed set:               holdout return  ~240 after 30 iterations
```

Generalisation is then checked separately, on seeds the search never saw.
That is where the risk of this choice lives, and it is the right place for
it: overfitting to eight spawns is visible in the held-out evaluation,
whereas a noisy gradient is invisible and just looks like a hard problem.

## 4. Match the policy class to the optimiser

The default policy is **linear** — around 80 parameters.

Derivative-free optimisers estimate over the whole parameter vector, so
their sample cost grows with dimension. A two-layer MLP here is about 5,600
parameters against a population of tens; the search is badly
under-determined before it starts.

A linear policy on body-frame errors is structurally a PD controller, which
is what these tasks want. To check that the *representation* was sufficient
rather than assuming it, we cloned a hand-written expert controller into it
by least squares:

```text
Hand-written expert:          return 638,  500/500 steps
Linear policy cloned from it: return 623,  500/500 steps
```

80 parameters are enough. The question was never capacity — it was search.

::: tip Diagnose in this order
When a learning setup does not work, this is the order that isolates the
cause fastest:

1. **Can a hand-written controller solve it?** If not, the environment or
   reward is broken, and no optimiser will save you.
2. **Can the policy class represent that controller?** Clone it in by
   supervised regression. If cloning fails, the architecture or the
   observation is wrong.
3. **Only then** is it an optimisation problem.

Each step is cheap, and skipping to step 3 is how weeks disappear into
hyperparameter sweeps against a broken reward.
:::

## Reward shaping

The position term is a Gaussian in distance rather than a negative
distance:

```python
position_term = exp(-(distance / distance_scale) ** 2)
```

Two reasons. It is **bounded and always positive**, so ending the episode
early is never cheaper than continuing to fly — with a negative-distance
reward, crashing immediately can genuinely be the optimal policy. And its
gradient is strongest near the goal, which is where precision matters.

`distance_scale` sets how far out the reward still has usable slope. Too
narrow and a policy that starts by drifting away sees a flat zero and has
nothing to climb; the hover task uses 2.5 m against spawns up to ~3 m out.

## Termination

Every environment reports *why* an episode ended:

```python
info["termination_reason"]   # ground_contact, tumbled, out_of_bounds,
                             # stalled, landed, goal_reached, diverged
```

`evaluate()` aggregates these into percentages. A policy with a mediocre
return that is 90% `time_limit` is in a completely different place from one
with the same return that is 90% `tumbled`, and the scalar return does not
distinguish them.

## Terminated versus truncated

The Gymnasium split is not bookkeeping. `terminated` means the episode
genuinely ended — the aircraft crashed, or the task was solved.
`truncated` means the clock ran out and the state was otherwise fine, so a
value function should still bootstrap from it.

Collapsing them into a single `done` teaches the agent that the world ends
after ten seconds, which it will happily plan around.

## See also

- [Environments](/learning/environments) — the six tasks in detail
- [Training a policy](/learning/training) — the optimisers
