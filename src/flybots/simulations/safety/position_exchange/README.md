# Position Exchange (CBF safety filter)

Eight agents on a ring, each commanded straight at the point diametrically
opposite it, so every straight-line path crosses the centre at the same
moment. The nominal controller is a plain PD that cannot see the other
agents. A control barrier function filter sits between it and the vehicles.

## Key Equations

The safe set is $h \ge 0$ with $h = \lVert \Delta p \rVert^2 - d^2$. Because
the input is acceleration, $h$ has relative degree two and the plain
condition constrains nothing, so the high-order form is used:

$$\ddot h + (k_1 + k_2)\dot h + k_1 k_2 h \ge 0$$

Every barrier contributes one such row, and the filter solves

$$u^\star = \arg\min_u \tfrac12 \lVert u - u_{\text{nom}} \rVert^2
\quad \text{s.t.} \quad A u \le b$$

## Result

| run | closest pair | goal error |
|---|---|---|
| unfiltered | 0.051 m | 0.000 m |
| filtered | 1.790 m | 8.633 m |
| filtered + swirl | 1.784 m | 0.000 m |

Safe distance is 1.800 m.

The middle row is the point. **A CBF guarantees safety, not liveness.**
Standing still is safe, so the QP will happily hold a symmetric fleet
still forever. A small tangential bias, applied only while the fleet is
jammed, breaks the symmetry and the swap completes.

The filtered runs settle 0.6–0.9 % inside the barrier rather than exactly
on it. That is the forward-Euler step: the condition holds in continuous
time and discretisation lets it overshoot slightly. Size the safe distance
with margin for it rather than trusting the number exactly.

## Reference

U. Borrmann, L. Wang, A. D. Ames & M. Egerstedt, "Control Barrier
Certificates for Safe Swarm Behavior," IFAC ADHS, 2015.
[DOI](https://doi.org/10.1016/j.ifacol.2015.11.154)

## Usage

```bash
python -m flybots.simulations.safety.position_exchange
```

## Result

![position_exchange](position_exchange.gif)
