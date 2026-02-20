# GPS / IMU Fusion

Combines low-rate GPS position fixes with high-rate IMU accelerometer/gyro data using an EKF. GPS dropout periods are handled gracefully by relying on IMU dead-reckoning until GPS recovers.

## Key Equations

$$\text{Predict: } \hat x_{k|k-1} = f(\hat x_{k-1}, u_k), \quad \text{Update: } \hat x_{k|k} = \hat x_{k|k-1} + K_k(z_k^{\text{GPS}} - H\hat x_{k|k-1})$$

## Reference

J. Farrell, "Aided Navigation: GPS with High Rate Sensors," McGraw-Hill, 2008.

## Usage

```bash
python -m uav_sim.simulations.estimation.gps_imu_fusion
```

## Result

![gps_imu_fusion](gps_imu_fusion.gif)
