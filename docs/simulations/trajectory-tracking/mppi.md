<!-- Erwin Lejeune — 2026-02-24 -->
# Model Predictive Path Integral (MPPI)

## Problem Statement

MPPI addresses nonlinear trajectory tracking without local linearization by sampling control perturbations and weighting trajectories by cost.
It is effective in regimes with nonconvex costs and uncertain dynamics.

## Model and Formulation

For control sequence `U`, MPPI computes update:

$$
\Delta u_t = \frac{\sum_{k=1}^{K} \exp\left(-\frac{1}{\lambda}S_k\right)\epsilon_{k,t}}{\sum_{k=1}^{K} \exp\left(-\frac{1}{\lambda}S_k\right)}
$$

where `S_k` is rollout cost and `\epsilon_{k,t}` is sampled perturbation at time `t`.

## Algorithm Procedure

1. Sample `K` noisy control sequences around nominal controls.
2. Roll out dynamics and compute trajectory costs.
3. Compute importance-weighted control correction.
4. Shift horizon and repeat at each control cycle.

## What Gets Scored, and What Gets Flown

Two details decide whether this tracks or merely follows at a distance.

**Each horizon step is scored against the reference at that step.** Scoring
the whole rollout against the reference's *present* value asks the plan to
stop where the trajectory currently is, which costs roughly half a horizon
of lag however many samples you draw.

**The nominal rollout is the plan.** MPPI's output is the weighted control
sequence; rolling it forward gives the trajectory the controller has
actually committed to. The unweighted mean of the sampled rollouts is a
different object — an average over mostly-rejected candidates that drifts
towards wherever the sampling distribution is centred. In this demo the
nominal rollout is the green line, and its first state (position and
velocity) plus the chosen acceleration are what the tracking controller
receives.

## Tuning Guidance

- Increase sample count `K` for better solution quality.
- Lower temperature `\lambda` sharpens elite trajectory selection.
- Match exploration covariance to expected disturbance magnitudes.

## Failure Modes and Diagnostics

- Insufficient samples lead to high-variance control updates.
- Overly aggressive exploration destabilizes near-hover behavior.
- Large horizon with slow hardware can violate realtime deadlines.

## Implementation and Execution

```bash
python -m flybots.simulations.trajectory_tracking.mppi
```

## Evidence

![MPPI](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/trajectory_tracking/mppi/mppi.gif)

## References

- [Williams et al., Information Theoretic MPC for Model-Based Reinforcement Learning (2017)](https://ieeexplore.ieee.org/document/7989202)
- [Theodorou et al., Policy Improvement with Path Integrals (2010)](https://doi.org/10.1007/s10514-010-9197-8)

## Related Algorithms

- [Nonlinear MPC](/simulations/trajectory-tracking/nmpc)
- [Feedback Linearisation](/simulations/trajectory-tracking/feedback-linearisation)
