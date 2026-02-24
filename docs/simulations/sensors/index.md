<!-- Erwin Lejeune — 2026-02-24 -->
# Sensors

Sensor subsystem algorithms describe how camera geometry, gimbal control, and target extraction are integrated for robust tracking.

## Core Questions

- How should gimbal kinematics be controlled under platform motion?
- Which observation models best reduce target drift?
- How does measurement noise map into line-of-sight error?

## Algorithms

- [Gimbal Tracking](/simulations/sensors/gimbal-tracking)
- [Gimbal BBox Tracking](/simulations/sensors/gimbal-bbox-tracking)

## Prerequisites

- Camera intrinsics/extrinsics
- Servo dynamics and control loops
- Coordinate transforms between body and camera frames
