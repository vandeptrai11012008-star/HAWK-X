# HAWK-X Vision AI

**Status: scaffold, not a deployed detector.**

Purpose: local analysis of consented or synthetic test imagery. Planned outputs include human-detection events, confidence scores, track continuity across offline test frames, and sample snapshots for operator review.

Inputs (planned): RGB camera frames, timestamps.
Outputs: `shared/examples/detection.json` compatible events.

## Development targets
- Run repeatable evaluation on licensed/consented images.
- Measure accuracy and false-alert rate across varied conditions.
- Keep data acquisition times in the source timestamp.
- Do not treat model confidence as certainty.

No automated pursuit or autonomous flight control is included.
