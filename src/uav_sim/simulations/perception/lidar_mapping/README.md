# Lidar Mapping

Real-time 3D point cloud accumulation from a spinning lidar sensor as the drone traverses the environment. Demonstrates sensor-to-world frame transformation and incremental map building.

## Key Equations

$$p_{\text{world}} = R(\phi,\theta,\psi)\,p_{\text{body}} + t_{\text{drone}}$$

## Reference

C. Cadena et al., "Past, Present, and Future of Simultaneous Localization and Mapping," IEEE TRO, 2016. [DOI](https://doi.org/10.1109/TRO.2016.2624754)

## Usage

```bash
python -m uav_sim.simulations.perception.lidar_mapping
```

## Result

![lidar_mapping](lidar_mapping.gif)
