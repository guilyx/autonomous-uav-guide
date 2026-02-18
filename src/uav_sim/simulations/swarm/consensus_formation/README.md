# Consensus-Based Formation Control

Multiple agents converge to a desired formation using distributed consensus. Each agent adjusts its position based on the relative positions of its neighbours and the desired inter-agent offsets.

## Key Equations

$$\dot x_i = \sum_{j \in \mathcal{N}_i} (x_j - x_i - d_{ij})$$

## Reference

W. Ren, R. W. Beard, "Consensus Seeking in Multiagent Systems Under Dynamically Changing Interaction Topologies," IEEE TAC, 2005. [DOI](https://doi.org/10.1109/TAC.2005.846556)

## Usage

```bash
python -m uav_sim.simulations.swarm.consensus_formation
```

## Result

![consensus_formation](consensus_formation.gif)
