<!-- Erwin Lejeune — 2026-02-23 -->
# Consensus Formation

## Algorithm

Agents use consensus protocol over a communication graph to converge to a desired formation (hexagon). Each agent adjusts toward the average of its neighbours' positions plus a formation offset.

**Reference:** R. Olfati-Saber, R. M. Murray, "Consensus Problems in Networks of Agents with Switching Topology," IEEE TAC, 2004.

## Run

```bash
python -m uav_sim.simulations.swarm.consensus_formation
```
