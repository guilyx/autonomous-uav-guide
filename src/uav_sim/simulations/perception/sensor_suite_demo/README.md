# Sensor Suite Demo

Showcases all sensor models simultaneously: GPS, IMU, 2D lidar, 3D lidar, camera, and range finder. The drone follows a path while each sensor's output is visualised in real time.

## Key Equations

$$z = h(x) + v, \quad v \sim \mathcal{N}(0, R)$$

## Reference

R. Siegwart, I. R. Nourbakhsh, D. Scaramuzza, "Introduction to Autonomous Mobile Robots," 2nd ed., MIT Press, 2011.

## Usage

```bash
python -m uav_sim.simulations.perception.sensor_suite_demo
```

## Result

![sensor_suite_demo](sensor_suite_demo.gif)
