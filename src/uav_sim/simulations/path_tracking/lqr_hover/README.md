# LQR Hover Control

Stabilises the quadrotor at a hover point using a Linear Quadratic Regulator. The nonlinear dynamics are linearised about hover equilibrium and the infinite-horizon LQR gain is computed by solving the continuous-time algebraic Riccati equation.

## Key Equations

$$u = -K(x - x_{\text{ref}}) + u_{\text{hover}}, \quad K = R^{-1}B^T P, \quad A^TP + PA - PBR^{-1}B^TP + Q = 0$$

## Reference

B. D. O. Anderson, J. B. Moore, "Optimal Control: Linear Quadratic Methods," Prentice Hall, 1990.

## Usage

```bash
python -m uav_sim.simulations.path_tracking.lqr_hover
```

## Result

![lqr_hover](lqr_hover.gif)
