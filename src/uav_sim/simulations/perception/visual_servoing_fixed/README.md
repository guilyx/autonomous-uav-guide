# Visual Servoing — Fixed Camera

Image-based visual servoing (IBVS) with a **strapdown** camera: the
camera is rigidly attached to the airframe, so the only way to keep the
target in frame is to move and yaw the drone itself.

The controller closes three loops on image measurements alone:

| Image error | Actuated by |
|---|---|
| horizontal bbox offset $\Delta u$ | body-lateral velocity |
| vertical bbox offset $\Delta v$ | climb rate |
| bbox size error $\Delta s$ | body-forward velocity |

Heading is servoed separately so the target stays inside the (narrow)
fixed field of view — this is what makes the fixed-camera case harder
than the [gimbal variant](../visual_servoing_gimbal/README.md).

## Key Equations

$$\mathbf{v}_{\text{body}} = K_p \, \mathbf{e}_{\text{img}},
\qquad
\mathbf{e}_{\text{img}} = [\,s^\star - s,\; -\Delta u,\; -\Delta v\,]$$

The lateral and vertical signs are negative because the image $u$ axis
points right while the body $y$ axis points left (FLU).

## Reference

F. Chaumette & S. Hutchinson, "Visual Servo Control — Part I: Basic
Approaches," IEEE Robotics & Automation Magazine, vol. 13, no. 4, 2006.
[DOI](https://doi.org/10.1109/MRA.2006.250573)

## Usage

```bash
python -m uav_sim.simulations.perception.visual_servoing_fixed
```

## Result

![visual_servoing_fixed](visual_servoing_fixed.gif)
