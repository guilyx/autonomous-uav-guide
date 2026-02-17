# Sensor Models

## Overview

Sensor models produce noisy measurements from the true vehicle state and world.

| Sensor | Output | Key Parameters |
| --- | --- | --- |
| `IMU` | 6-vec (accel + gyro) | bias, noise std |
| `GPS` | 3-vec (position) | noise std, dropout prob |
| `Lidar2D` | N ranges | num beams, max range, noise |
| `Camera` | Pixel coords | intrinsics (fx, fy, cx, cy) |
| `RangeFinder` | 1 altitude | max range, noise |

## Reference

N. Trawny, S. I. Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation," TR, 2005.

## API

```python
from uav_sim.sensors import IMU, GPS, Lidar2D
imu = IMU(accel_noise_std=0.1, gyro_noise_std=0.01)
gps = GPS(noise_std=0.5, dropout_prob=0.05)
lidar = Lidar2D(num_beams=360, max_range=30.0)

accel_gyro = imu.sense(state)
position = gps.sense(state)
ranges = lidar.sense(state, world)
```
