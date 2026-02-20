# Potential-Based Swarm

Agents interact via a Lennard-Jones-like potential that attracts at long range and repels at short range, creating a natural equilibrium distance. The swarm self-organises into a lattice-like formation.

## Key Equations

$$U(r) = \epsilon\left[\left(\frac{r_0}{r}\right)^{12} - 2\left(\frac{r_0}{r}\right)^6\right]$$

## Reference

V. Gazi, K. M. Passino, "Stability Analysis of Swarms," IEEE TAC, 2003. [DOI](https://doi.org/10.1109/TAC.2003.809765)

## Usage

```bash
python -m uav_sim.simulations.swarm.potential_swarm
```

## Result

![potential_swarm](potential_swarm.gif)
