# Geofence and Floor (high-order CBFs)

The nominal command does the two things a geofence exists to prevent: flat
out at the boundary, then a dive at the ground. Two barriers refuse it.

## Result

| run | min wall margin | min floor margin |
|---|---|---|
| filtered | **+0.00 m** | **+0.00 m** |
| unfiltered | -13463 m | -4481 m |

Both constraints are on position, which the input reaches only through the
second derivative, so the filter must begin decelerating well before the
limit rather than clamping at it. A clamp is not a safety guarantee; it is
a report that safety was already lost.

## Usage

```bash
python -m flybots.simulations.safety.geofence_floor
```

## Result

![geofence_floor](geofence_floor.gif)
