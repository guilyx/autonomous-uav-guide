# Pure Pursuit 3D

A geometric path-tracking algorithm that steers toward a look-ahead point on the path. The look-ahead distance adapts with speed, providing smooth tracking without oscillation. Simple, robust, and widely used in practice.

## Key Equations

$$\kappa = \frac{2\,\sin\alpha}{L_d}, \quad L_d = L_{\min} + k_v \|v\|$$

## Reference

R. C. Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm," CMU-RI-TR-92-01, 1992. [Link](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)

## Usage

```bash
python -m uav_sim.simulations.path_tracking.pure_pursuit
```

## Result

![pure_pursuit](pure_pursuit.gif)
