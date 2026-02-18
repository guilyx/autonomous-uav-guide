# Cascaded PID Hover

The foundational controller: cascaded PID loops for position, velocity, and attitude stabilise the quadrotor at a fixed hover point. Demonstrates the basic building block of multirotor control.

## Key Equations

$$u(t) = K_p e(t) + K_i \int e(\tau)d\tau + K_d \dot{e}(t)$$

## Reference

K. J. Astrom, R. M. Murray, "Feedback Systems: An Introduction for Scientists and Engineers," Princeton University Press, 2008. [Link](https://www.cds.caltech.edu/~murray/amwiki/index.php)

## Usage

```bash
python -m uav_sim.simulations.path_tracking.pid_hover
```

## Result

![pid_hover](pid_hover.gif)
