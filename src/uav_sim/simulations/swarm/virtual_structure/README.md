# Virtual Structure Formation

Agents track positions defined by a rigid virtual structure that can translate and rotate. The structure moves as a whole and each agent is attracted to its assigned slot, producing coordinated formation manoeuvres.

## Key Equations

$$p_i^{\text{des}} = R(\psi_s)\,\delta_i + p_s$$

## Reference

M. B. Egerstedt, X. Hu, "Formation Constrained Multi-Agent Control," IEEE TRA, 2001. [DOI](https://doi.org/10.1109/70.976022)

## Usage

```bash
python -m uav_sim.simulations.swarm.virtual_structure
```

## Result

![virtual_structure](virtual_structure.gif)
