# Feedback Linearisation

## Theory

Exploits the differential flatness of quadrotor dynamics to cancel nonlinearities. The position tracking error dynamics become a double-integrator with chosen PD gains:

$$\ddot{e} + K_d \dot{e} + K_p e = 0$$

The desired thrust magnitude and body orientation are extracted from the computed acceleration and the desired yaw.

## Reference

D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," ICRA, 2011, Sec. IV. [DOI: 10.1109/ICRA.2011.5980409](https://doi.org/10.1109/ICRA.2011.5980409)

## Simulation

```bash
uv run python simulations/feedback_linearisation/run.py
```

![Feedback Linearisation](../../simulations/feedback_linearisation/feedback_linearisation.gif)

## API

```python
from quadrotor_sim.tracking.feedback_linearisation import FeedbackLinearisationTracker
tracker = FeedbackLinearisationTracker(mass=0.027)
wrench = tracker.compute(state, ref_pos, ref_vel, ref_acc)
```
