# Linear Quadratic Regulator (LQR)

## Theory

The quadrotor dynamics are linearised about hover to obtain $\dot{\delta x} = A\delta x + B\delta u$. The optimal gain $K$ is found by solving the continuous-time Algebraic Riccati Equation:

$$A^\top P + PA - PBR^{-1}B^\top P + Q = 0$$

yielding $K = R^{-1}B^\top P$, and control $\delta u = -K \delta x$.

## Reference

B. D. O. Anderson, J. B. Moore, "Optimal Control: Linear Quadratic Methods," Prentice-Hall, 1990.

## Simulation

```bash
uv run python simulations/lqr_hover/run.py
```

![LQR Hover](../../simulations/lqr_hover/lqr_hover.gif)

## API

```python
from quadrotor_sim.control.lqr_controller import LQRController
ctrl = LQRController(mass=0.027, gravity=9.81)
wrench = ctrl.compute(state_12, target_state_12)
```
