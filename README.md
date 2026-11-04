# HAWK-X — Onboard AI Rescue Intelligence

HAWK-X is an early-stage research prototype for search-and-rescue awareness. The project goal is to process sensor data onboard, detect people in **authorized test footage**, build a local 3D environment map, and send evidence and mapping updates to a human-operated Rescue HQ.

> **Status:** Software scaffold / simulated demo. This repository does **not** implement autonomous flight or autonomous pursuit, and no real-world rescue capabilities are currently validated.

## Architecture

```text
RGB / depth camera       LiDAR + IMU
       |                     |
  vision-ai/             mapping-3d/
       \                     /
        \                   /
         core-hq/ (data fusion + event ingestion)
                    |
            Rescue HQ web demo
               2D + 3D
```

The planned onboard perception and mapping modules share timestamps and coordinate-frame identifiers. The HQ may show an event without a geographic marker when localization is unverified.

## Repository layout

- `vision-ai/` — evaluation interface and sample detection events, no live surveillance.
- `mapping-3d/` — local 3D point-cloud data contract and demo frames.
- `core-hq/` — Rescue HQ browser demo and common integration documentation.
- `shared/` — stable JSON event schemas and example data.
- `docs/` — architecture, hardware inventory, validation checklist.

## Quick start (offline demo)

1. Clone the repository.
2. From the repository root run `python -m http.server 8000` (Python 3).
3. Open `http://localhost:8000/core-hq/`.
4. Inspect the sample detection events and the small synthetic 3D point cloud.
5. Read `docs/architecture.md` before connecting real sensors.

No API keys, cloud service, or physical hardware are needed for the demo. Sample data is intentionally synthetic and clearly labeled.

## Roadmap

1. Verify camera and compute hardware model at FORGE.
2. Benchmark offline human detection on consented recordings.
3. Test 3D mapping with licensed LiDAR sample data.
4. Integrate timestamped events and map frames through a documented interface.
5. Evaluate accuracy, false alerts, latency, and safe operator review.
6. Consider flight integration only after independent flight-safety reviews and approved supervised testing.

## Safety & attribution

The flight-control stack must remain separate from perception software, with operator override, geofencing, and appropriate fail-safe behavior. Do not claim that a simulated alert is a real rescue event. Record dataset provenance and third-party licenses before importing research repositories.

Developed as a project concept for Code Metal FORGE Hackathon.
