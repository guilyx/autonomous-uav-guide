# Occupancy Grid Mapping

Builds a 2D occupancy grid from lidar range measurements using a log-odds update rule. The drone flies a lawnmower pattern, and the map is progressively revealed as the sensor sweeps the environment. Combines ray-casting occupancy mapping with real-time point cloud visualization.

## Key Equations

$$l(m_{xy}) \leftarrow l(m_{xy}) + \log\frac{p(m_{xy} \mid z)}{1 - p(m_{xy} \mid z)} - l_0$$

## Reference

S. Thrun, "Learning Occupancy Grid Maps with Forward Sensor Models," Autonomous Robots, 2003. [DOI](https://doi.org/10.1023/A:1025584000927)

## Usage

```bash
python -m uav_sim.simulations.perception.occupancy_mapping
```

## Result

![occupancy_mapping](occupancy_mapping.gif)
