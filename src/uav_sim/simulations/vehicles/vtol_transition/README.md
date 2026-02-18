# VTOL Transition

Simulates a tiltrotor VTOL UAV transitioning between hover (rotors vertical) and forward flight (rotors horizontal). The tilt angle smoothly varies during the transition corridor.

## Key Equations

$$T_{\text{vert}} = T\cos(\alpha_{\text{tilt}}), \quad T_{\text{horiz}} = T\sin(\alpha_{\text{tilt}})$$

## Reference

P. Hartmann, C. Meyer, D. Moormann, "Unified Velocity Control and Flight State Transition of Unmanned Tilt-Wing Aircraft," JGCD, 2017. [DOI](https://doi.org/10.2514/1.G002168)

## Usage

```bash
python -m uav_sim.simulations.vehicles.vtol_transition
```

## Result

![vtol_transition](vtol_transition.gif)
