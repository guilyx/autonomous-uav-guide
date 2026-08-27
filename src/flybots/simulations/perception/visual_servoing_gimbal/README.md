# Visual Servoing — Gimbal Camera

Image-based visual servoing (IBVS) with a **pan-tilt gimbal**. The
gimbal owns the fast loop — it keeps the bounding box centred — and the
drone owns the slow loop: it flies so that the *gimbal* stays near its
neutral pointing angle and the target keeps its desired apparent size.

That split is the point of the demo. With a gimbal, centring the bbox no
longer tells the drone where to go (the gimbal absorbs it), so the
position command has to be built from the **gimbal angles** plus the
size error instead:

| Measurement | Actuated by |
|---|---|
| gimbal pan away from neutral | body-lateral velocity |
| gimbal tilt away from neutral | climb rate |
| bbox size error $\Delta s$ | body-forward velocity |

## Key Equations

$$\mathbf{v}_{\text{body}} =
 \big[\,k_f (s^\star - s),\; -k_\ell \, \psi_{\text{pan}},\;
       k_\ell (\theta_{\text{tilt}} - \theta^\star)\,\big]$$

with the commanded velocity rotated into the world frame by the drone
heading.

## Reference

F. Chaumette & S. Hutchinson, "Visual Servo Control — Part I: Basic
Approaches," IEEE Robotics & Automation Magazine, vol. 13, no. 4, 2006.
[DOI](https://doi.org/10.1109/MRA.2006.250573)

## Usage

```bash
python -m flybots.simulations.perception.visual_servoing_gimbal
```

## Result

![visual_servoing_gimbal](visual_servoing_gimbal.gif)
