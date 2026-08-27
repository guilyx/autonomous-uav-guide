# Obstacle Slalom (CBF)

The nominal controller flies dead straight at the goal and has never heard
of the obstacles. Everything keeping the vehicle out of them is the barrier.

## Result

| run | min clearance to surface | goal error |
|---|---|---|
| filtered | **+1.189 m** | 0.00 m |
| unfiltered | **-1.997 m** (two metres inside) | 0.00 m |

Required clearance 1.200 m. The filtered run sits 1% inside it, which is the
forward-Euler step overshooting a condition that holds in continuous time.

Both runs reach the goal, so safety cost nothing here — which is not a
general property, only what this geometry allows.

## Usage

```bash
python -m flybots.simulations.safety.obstacle_slalom
```

## Result

![obstacle_slalom](obstacle_slalom.gif)
