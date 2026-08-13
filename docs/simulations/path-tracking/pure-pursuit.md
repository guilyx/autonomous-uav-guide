<!-- Erwin Lejeune — 2026-08-13 -->
# Pure Pursuit 3D

## Problem Statement

Given a waypoint list and a position controller, pure pursuit answers the only question in between: *which point should the vehicle aim at right now?*
It chases a "carrot" a fixed distance ahead on the path, which turns a sequence of corners into a smooth, continuously-defined setpoint.

## Model and Formulation

The carrot is the intersection of a sphere of radius `L` about the vehicle with the path:

$$
\lVert p_{target} - p \rVert = L, \qquad p_{target} \in \text{path}
$$

Taking the **furthest** intersection on the earliest remaining segment keeps the target moving forward. With `adaptive`, the look-ahead grows with speed:

$$
L = L_0\left(1 + 0.15\,\lVert v \rVert\right)
$$

## Making Progress Monotone

The subtle part is not finding the carrot — it is deciding which segment
to search from. Advancing the index only when the vehicle comes within
`waypoint_threshold` of the current waypoint is not enough. On a
trajectory that loops back near itself, the vehicle can sit outside that
threshold while the look-ahead sphere keeps intersecting an *earlier*
segment. The carrot stays behind it, and it circles there forever. The
min-snap demo used to hit its 90 s timeout 24 m short of the goal for
exactly this reason.

The fix is to snap the index to the nearest waypoint ahead — but bounded,
and bounded by **arc length along the path**, not by waypoint count:

```python
window = self._search_window(path, self._idx)   # progress_window × lookahead
nearest = argmin(‖path[idx:window+1] - position‖)
self._idx += nearest
```

Why arc length: on a lawnmower coverage path, adjacent lanes pass within
a metre of each other while being many metres apart *along* the path. An
index window lets the tracker hop lanes and skip most of the coverage —
93 % down to 44 % on the occupancy-mapping demo. An arc-length window
cannot, because reaching the next lane costs more path than the window
allows.

## Two Thresholds, Not One

`waypoint_threshold` decides when to advance between waypoints;
`goal_threshold` decides when the mission is finished. They want
different values. Advancing early is what keeps the path smooth;
declaring the goal reached early leaves the vehicle short by exactly that
slack — which is how missions came to be scored as never having arrived
while flying perfectly well.

## Algorithm Procedure

1. Snap the segment index forward to the nearest waypoint within the arc-length window.
2. Advance past any waypoint already inside `waypoint_threshold`.
3. Scale the look-ahead with current speed.
4. Intersect the look-ahead sphere with the remaining segments; take the first hit.
5. Smooth the target temporally to avoid a jump at segment transitions.

## Tuning Guidance

- Larger look-ahead cuts corners and smooths; smaller tracks tightly and can oscillate. It is the single most consequential parameter.
- `smoothing` is a first-order filter on the carrot, not on the vehicle — it hides segment-transition steps without adding vehicle lag.
- Set `goal_threshold` from the mission's success criterion, not from `waypoint_threshold`.

## Failure Modes and Diagnostics

- A vehicle circling one region forever is a progress problem, not a control problem.
- Coverage paths losing lanes means the progress window is measured in indices.
- Look-ahead longer than the corner radius cuts corners into obstacles.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_tracking.pure_pursuit
```

## Evidence

![Pure Pursuit](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pure_pursuit/pure_pursuit.gif)

## References

- [Coulter, Implementation of the Pure Pursuit Path Tracking Algorithm, CMU-RI-TR-92-01 (1992)](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
- [Snider, Automatic Steering Methods for Autonomous Automobile Path Tracking (2009)](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

## Related Algorithms

- [Path Smoothing](/simulations/path-tracking/path-smoothing)
- [Flight Ops Demo](/simulations/path-tracking/flight-ops-demo)
- [A* 3D](/simulations/path-planning/astar-3d)
