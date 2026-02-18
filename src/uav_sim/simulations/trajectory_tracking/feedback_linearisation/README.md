# Feedback Linearisation

Cancels the quadrotor's nonlinear dynamics through input-output linearisation, transforming the control problem into a set of decoupled double integrators. A linear PD controller then drives the linearised system.

## Key Equations

$$u = M(q)v + C(q,\dot q)\dot q + g(q), \quad v = \ddot q_d + K_d(\dot q_d - \dot q) + K_p(q_d - q)$$

## Reference

H. K. Khalil, "Nonlinear Systems," 3rd ed., Prentice Hall, 2002.

## Usage

```bash
python -m uav_sim.simulations.trajectory_tracking.feedback_linearisation
```

## Result

![feedback_linearisation](feedback_linearisation.gif)
