# HAWK-X 3D Mapping

**Status: scaffold with synthetic point-cloud fixture.**

Purpose: represent a local 3D environment map. Future LiDAR + IMU integration requires sensor calibration, appropriate drivers, and measurement-based validation.

## Data
See `../shared/examples/map-frame.json` for a tiny synthetic point cloud.

## Important distinctions
- A local XYZ point cloud is **not** automatically geo-referenced terrain.
- Display mapping confidence and unknown space.
- Synchronize sensor capture timestamps and coordinate frames before fusing with vision events.
- No flight planning, obstacle-avoidance control, or live SLAM is implemented.
