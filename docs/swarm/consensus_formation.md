# Consensus-Based Formation Control

## Theory

Agents converge to desired offsets using a graph Laplacian-based protocol. Given adjacency matrix $A$ and desired offsets $d_i$, agent $i$ applies:

$$u_i = -\kappa \sum_{j \in \mathcal{N}_i} \left[(p_i - d_i) - (p_j - d_j)\right]$$

Under mild connectivity assumptions, all agents converge to the desired formation.

## Reference

R. Olfati-Saber, R. M. Murray, "Consensus Problems in Networks of Agents with Switching Topology and Communication Delays," IEEE TAC, 2004. [DOI: 10.1109/TAC.2004.834113](https://doi.org/10.1109/TAC.2004.834113)

## Simulation

```bash
uv run python simulations/swarm/consensus_formation/run.py
```

![Consensus Formation](../../simulations/swarm/consensus_formation/consensus_formation.gif)

## API

```python
from uav_sim.swarm.consensus_formation import ConsensusFormation
ctrl = ConsensusFormation(adjacency=A, offsets=offsets, gain=1.5)
forces = ctrl.compute_forces(positions)
```
