# Connectivity Maintenance

24 agents, each given a goal scattered across a 400 m box — far enough apart
that flying the task straight tears the radio network into islands. A
connectivity controller ascends the gradient of algebraic connectivity, so
the fleet trades goal progress for a mesh that survives.

## Key Equations

$\lambda_2$ is the second smallest eigenvalue of the weighted graph
Laplacian $L = D - W$, strictly positive exactly while the network is
connected. Its gradient has a closed form through the Fiedler vector $v$:

$$\frac{\partial \lambda_2}{\partial p_i} = \sum_j \frac{\partial w_{ij}}{\partial p_i}\,(v_i - v_j)^2$$

Effort goes where the *Fiedler vector* disagrees most, not where the
distance is greatest — those are different edges. The Fiedler vector is
near-constant inside a tight cluster and jumps across the weak cut between
clusters, so the gradient concentrates on the links actually holding the
network together.

## Result

| run | final $\lambda_2$ | $k$-connectivity |
|---|---|---|
| task only | 5.67e-06 | 0 (fragmented) |
| connectivity-aware | 0.2762 | 1 |

The aware run does **not** reach its goals — mean goal error plateaus near
110 m. That is the price, and it is the honest way to show it.

## Reference

L. Sabattini et al., "Decentralized connectivity maintenance for cooperative
control of mobile robotic systems," IJRR, 2013.
[DOI](https://doi.org/10.1177/0278364913499085)

## Usage

```bash
python -m uav_sim.simulations.comms.connectivity_maintenance
```

## Result

![connectivity_maintenance](connectivity_maintenance.gif)
