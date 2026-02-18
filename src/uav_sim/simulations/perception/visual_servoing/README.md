# Visual Servoing

Image-based visual servoing (IBVS): the drone autonomously follows a
moving ground target by keeping its bounding box centred in the camera
frame and maintaining a desired apparent size.

## Key Equations

The image-space error drives velocity commands:

$$\mathbf{v} = K_p \cdot \mathbf{e}_{\text{img}}$$

where $\mathbf{e}_{\text{img}} = [\Delta u, \Delta v, \Delta s]$ encodes
lateral offset and size error.

## Reference

F. Chaumette & S. Hutchinson, "Visual Servo Control — Part I: Basic
Approaches," IEEE Robotics & Automation Magazine, vol. 13, no. 4, 2006.
[DOI](https://doi.org/10.1109/MRA.2006.250573)

## Usage

```bash
python -m uav_sim.simulations.perception.visual_servoing
```

## Result

![visual_servoing](visual_servoing.gif)
