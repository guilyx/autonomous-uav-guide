# Relay Coverage

18 agents spread from a fixed base station to watch as much ground as
possible, while every one of them holds a multi-hop radio path home. Ground
watched by an aircraft that cannot report it is not covered.

## Key Equations

Coverage counts only cells within sensing range of an agent that is still
*reachable* from the base:

$$C = \frac{1}{|Q|}\sum_{q \in Q} \mathbb{1}\left[\min_{i\,:\,\text{hops}(i) < \infty} \lVert q - p_i \rVert \le r_s\right]$$

Three forces: mutual repulsion produces coverage, an outward anchor keeps
the fleet expanding, and a connectivity tether scaled by **hop count** —
an agent three hops out is far more likely to sever the chain than one
beside the base, so uniform tethering wastes effort on agents never at risk.

## Result

| run | connected coverage | naive coverage | reachable |
|---|---|---|---|
| relay | **17.6 %** | 17.6 % | 18/18 |
| untethered | 2.3 % | 31.4 % | 1/18 |

The untethered run looks nearly twice as good if you count every agent's
footprint, and delivers almost nothing: the fleet flew apart, and only the
base can still report. In the plot its connected coverage falls off a cliff
at t ≈ 32 s — the instant the last relay link breaks.

## Reference

J. Scherer & B. Rinner, "Long-term area coverage and radio relay positioning
using swarms of UAVs," [arXiv:1810.12383](https://arxiv.org/abs/1810.12383),
2018.

## Usage

```bash
python -m uav_sim.simulations.comms.relay_coverage
```

## Result

![relay_coverage](relay_coverage.gif)
