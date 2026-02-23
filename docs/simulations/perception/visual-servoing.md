<!-- Erwin Lejeune — 2026-02-23 -->
# Visual Servoing

## Algorithm

Image-based visual servoing (IBVS) using bounding box centre error and size ratio to regulate approach distance and centering. The drone follows a moving target by keeping its bounding box at a desired size.

**Reference:** F. Chaumette, S. Hutchinson, "Visual Servo Control — Part I: Basic Approaches," IEEE RAM, 2006.

## Run

```bash
python -m uav_sim.simulations.perception.visual_servoing
```
