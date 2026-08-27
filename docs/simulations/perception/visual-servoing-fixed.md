<!-- Erwin Lejeune — 2026-08-26 -->
# Visual Servoing — Fixed Camera

## Problem Statement

Image-based visual servoing (IBVS) with a **strapdown** camera: the camera is
rigidly bolted to the airframe, so the image is whatever the drone is pointing
at. There is no second actuator to absorb tracking error — the only way to keep
the target in frame is to move and yaw the vehicle itself.

That constraint is what makes the fixed-camera case harder than the
[gimbal variant](/simulations/perception/visual-servoing-gimbal), and it is the
reason heading has to be servoed as a first-class loop rather than left to the
position controller.

## Model and Formulation

Three loops close on image measurements alone:

| Image error | Actuated by |
|---|---|
| horizontal bbox offset $\Delta u$ | body-lateral velocity |
| vertical bbox offset $\Delta v$ | climb rate |
| bbox size error $\Delta s$ | body-forward velocity |

The body-frame velocity command is a proportional map of the image error:

$$
\mathbf{v}_{\text{body}} = K_p \, \mathbf{e}_{\text{img}},
\qquad
\mathbf{e}_{\text{img}} = [\,s^\star - s,\; -\Delta u,\; -\Delta v\,]
$$

The lateral and vertical terms carry a negative sign because the image $u$ axis
points right and $v$ points down, while the body $y$ axis points left and $z$
points up (FLU). Apparent size $s$ stands in for range: the target is too far
when it looks too small.

## Algorithm Procedure

1. Detect the target and extract its bounding box in the image frame.
2. Form the image error against the desired centre and apparent size.
3. Flip the lateral and vertical signs into the body frame.
4. Servo heading toward the target bearing to hold it inside the field of view.
5. Emit saturated body-frame velocity commands.

## Tuning and Failure Modes

- **Check the sign convention before you touch the gains.** Both image errors
  flip sign entering the control law. Get one wrong and the drone chases the
  target *out* of frame — positive feedback that looks exactly like an unstable
  gain, and that no amount of detuning will fix.
- **The field of view is the real constraint.** With no gimbal, a narrow FOV
  turns a momentary tracking lag into a lost target. Heading has to lead, not
  follow, the lateral position loop.
- **Anchor the position setpoint to the current position.** Integrating the
  velocity command into a persistent setpoint means that once the position loop
  starts lagging, nothing in the image pulls the setpoint back.
- Gain $K_p$ too high produces oscillatory camera motion, which corrupts the
  very measurement the loop depends on.
- Occlusion stops the bounding box updating; without explicit fallback logic the
  last error is held and the drone keeps flying on stale data.

## Implementation and Execution

```bash
python -m flybots.simulations.perception.visual_servoing_fixed
```

## Evidence

![Visual Servoing — Fixed Camera](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/perception/visual_servoing_fixed/visual_servoing_fixed.gif)

## References

- [Chaumette and Hutchinson, Visual Servo Control Part I (2006)](https://doi.org/10.1109/MRA.2006.250573)
- [Chaumette and Hutchinson, Visual Servo Control Part II (2007)](https://doi.org/10.1109/MRA.2007.339609)

## Related Algorithms

- [Visual Servoing](/simulations/perception/visual-servoing)
- [Visual Servoing — Gimbal Camera](/simulations/perception/visual-servoing-gimbal)
- [Gimbal Tracking](/simulations/sensors/gimbal-tracking)
