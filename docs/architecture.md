# HAWK-X architecture v0.1

This repository starts with three independently maintainable modules:

1. **vision-ai**: edge/onboard visual perception research, outputting detection event objects.
2. **mapping-3d**: local 3D spatial representation, outputting time-stamped point cloud/map frames.
3. **core-hq**: a human-facing 2D/3D Rescue HQ for event review.

The two onboard modules should use consistent capture timestamps and coordinate-frame metadata. A fusion service can align them only after sensor clock synchronization, camera-to-LiDAR extrinsic calibration, and location-quality checks.

```text
             ONBOARD COMPUTE (concept)
camera -> vision-ai ----+
                        +-- fusion/validation --> communications --> core-hq
LiDAR -> mapping-3d ----+
IMU ---------^
```

The flight controller remains independent with qualified human oversight and failsafe mechanisms. The present repository contains **no drone control, target-following, or flight integration**.

## Proof-of-concept milestones
- Offline detection evaluation on permitted footage
- Mapping viewer from licensed recorded LiDAR data
- Timestamps + coordinate-frame alignment
- Latency and false-alert measurements
- Operator review interface for candidate rescue alerts

## Hardware questions
The FORGE list confirms Raspberry Pi 5 16 GB + 26 TOPS accelerator for every team, and a shared stock of Jetson boards, cameras, 2D/3D LiDAR, IMU, radios, and drones. Exact model compatibility, power, mounting, and mass remain to be established.
