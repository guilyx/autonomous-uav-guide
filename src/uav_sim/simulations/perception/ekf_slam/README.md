# EKF-SLAM

Simultaneous Localisation and Mapping using an Extended Kalman Filter. The state vector is augmented with landmark positions; as new landmarks are observed they are initialised, and re-observations refine both robot pose and map.

## Key Equations

$$x = [x_r, y_r, \theta_r, l_{1x}, l_{1y}, \dots, l_{nx}, l_{ny}]^T$$

## Reference

H. Durrant-Whyte, T. Bailey, "Simultaneous Localization and Mapping (SLAM): Part I," IEEE RAM, 2006. [DOI](https://doi.org/10.1109/MRA.2006.1638022)

## Usage

```bash
python -m uav_sim.simulations.perception.ekf_slam
```

## Result

![ekf_slam](ekf_slam.gif)
