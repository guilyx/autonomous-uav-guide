<!-- Erwin Lejeune — 2026-08-13 -->
# Path Smoothing

## Problem Statement

Grid and sampling planners return paths that are *correct* but not *flyable*: A\* output steps between cell centres, RRT output zigzags between random samples.
Smoothing sits between the planner and the tracker, turning a collision-free polyline into something a vehicle with inertia can follow without stopping at every vertex.

## Model and Formulation

Two stages, in order.

**Ramer–Douglas–Peucker** removes vertices that carry no shape. For the segment from `p_start` to `p_end`, find the point with the greatest perpendicular distance:

$$
d_{max} = \max_i \; \frac{\lVert (p_i - p_{start}) \times (p_{end} - p_{start}) \rVert}{\lVert p_{end} - p_{start} \rVert}
$$

Keep it and recurse if `d_max > ε`; otherwise discard everything between the endpoints.

**Cubic-spline resampling** then places points at uniform arc length along a `C²` interpolant of what survives, so curvature is continuous and the tracker's look-ahead sphere always finds a clean intersection.

## Why Both, and in That Order

Splining the raw path directly fits the planner's discretisation noise —
every grid step becomes a wiggle the vehicle then flies. RDP first
removes the noise; the spline then only has to round the corners that
carry real shape. Running them the other way round, or skipping RDP,
produces a smooth path that is longer and busier than the one A\* found.

RDP is also what keeps the corners **legal**. It only ever removes
points, never moves them, so a simplified path stays inside the corridor
its epsilon allows. The spline does move points, which is why `ε` and the
inflation radius have to be chosen together.

## Algorithm Procedure

1. Plan a raw path (A\*, RRT\*, PRM).
2. Simplify with RDP at a tolerance smaller than the obstacle inflation.
3. Resample the survivors along a cubic spline at fixed spacing.
4. Verify clearance on the resampled path before flying it — the spline may cut a corner RDP preserved.
5. Hand the result to [pure pursuit](/simulations/path-tracking/pure-pursuit).

## Tuning Guidance

- `epsilon` is the shape budget: too small keeps the zigzags, too large cuts corners into obstacles. Keep it well under the inflation radius used during planning.
- Resample spacing should be a fraction of the tracker's look-ahead, so the sphere intersection is never ambiguous.
- Smoothing shortens paths. The atlas A\* demo goes from 37.7 m raw to 35.8 m smoothed.

## Failure Modes and Diagnostics

- A spline through sparse waypoints can bulge outside the corridor between them; always re-check clearance after resampling, not just after RDP.
- Duplicate or near-duplicate waypoints make the spline parameterisation ill-conditioned.
- Over-aggressive simplification is easiest to spot as a path that clips an obstacle corner the raw path respected.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_tracking.path_smoothing_demo
```

## Evidence

![Path Smoothing](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/path_smoothing_demo/path_smoothing_demo.gif)

## References

- [Douglas and Peucker, Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line (1973)](https://doi.org/10.3138/FM57-6770-U75U-7727)
- [Ramer, An Iterative Procedure for the Polygonal Approximation of Plane Curves (1972)](https://doi.org/10.1016/S0146-664X%2872%2980017-0)

## Related Algorithms

- [Pure Pursuit 3D](/simulations/path-tracking/pure-pursuit)
- [A* 3D](/simulations/path-planning/astar-3d)
- [Min-Snap](/simulations/trajectory-planning/min-snap)
