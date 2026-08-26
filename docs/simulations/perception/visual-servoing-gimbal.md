<!-- Erwin Lejeune — 2026-08-26 -->
# Visual Servoing — Gimbal Camera

## Problem Statement

Image-based visual servoing (IBVS) with a **pan-tilt gimbal**. The gimbal owns
the fast loop — it keeps the bounding box centred — and the drone owns the slow
loop: it flies so the *gimbal* stays near its neutral pointing angle and the
target keeps its desired apparent size.

That split is the whole point of the demo. Once a gimbal is in the loop,
centring the bounding box no longer tells the drone where to go, because the
gimbal absorbs the error before the airframe ever sees it. A controller ported
straight from the [fixed-camera case](/simulations/perception/visual-servoing-fixed)
will sit still and think it is tracking perfectly.

## Model and Formulation

The position command is rebuilt from the **gimbal angles** plus the size error:

| Measurement | Actuated by |
|---|---|
| gimbal pan away from neutral | body-lateral velocity |
| gimbal tilt away from neutral | climb rate |
| bbox size error $\Delta s$ | body-forward velocity |

$$
\mathbf{v}_{\text{body}} =
 \big[\,k_f (s^\star - s),\; -k_\ell \, \psi_{\text{pan}},\;
       k_\ell (\theta_{\text{tilt}} - \theta^\star)\,\big]
$$

with the commanded velocity rotated into the world frame by the drone heading.
Non-zero pan means the target has drifted off the nose; a tilt steeper than
nominal means the drone is too high.

## Algorithm Procedure

1. Gimbal servos pan and tilt to centre the bounding box (fast loop).
2. Read the gimbal angles as the drone's position error signal.
3. Convert pan to lateral velocity and tilt deviation to climb rate.
4. Close range on apparent size.
5. Slew heading toward the target bearing at a bounded rate.

## Tuning and Failure Modes

- **Do not close yaw through the gimbal's own pointing loop.** Commanding
  `yaw = yaw + pan` puts two pointing loops in series chasing each other. The
  heading winds up and drags the position controller — which resolves tilt
  through yaw — with it. Slew the airframe toward the target bearing at a
  bounded rate instead, and let the gimbal own the fast loop.
- **Separate the timescales deliberately.** The gimbal loop must be clearly
  faster than the airframe loop. When the two run at comparable bandwidth they
  fight, and the gimbal angle stops being a clean error signal.
- **Neutral tilt is a tuning parameter, not a constant.** $\theta^\star$ sets
  the standoff geometry: too shallow and the drone creeps onto the target, too
  steep and it climbs away from it.
- Gain $k_\ell$ too high produces oscillatory motion that the gimbal dutifully
  cancels, hiding the instability from the image right up until it diverges.
- Occlusion leaves the gimbal pointing at empty space, and the drone will fly to
  hold that pointing angle unless tracking loss is handled explicitly.

## Implementation and Execution

```bash
python -m uav_sim.simulations.perception.visual_servoing_gimbal
```

## Evidence

![Visual Servoing — Gimbal Camera](https://media.githubusercontent.com/media/guilyx/flybots/main/src/uav_sim/simulations/perception/visual_servoing_gimbal/visual_servoing_gimbal.gif)

## References

- [Chaumette and Hutchinson, Visual Servo Control Part I (2006)](https://doi.org/10.1109/MRA.2006.250573)
- [Chaumette and Hutchinson, Visual Servo Control Part II (2007)](https://doi.org/10.1109/MRA.2007.339609)

## Related Algorithms

- [Visual Servoing](/simulations/perception/visual-servoing)
- [Visual Servoing — Fixed Camera](/simulations/perception/visual-servoing-fixed)
- [Gimbal BBox Tracking](/simulations/sensors/gimbal-bbox-tracking)
