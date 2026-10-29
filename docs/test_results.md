# Test Results and Evaluation Log

> All numbers in this file must come from real runs. Do not estimate.

This template is provided for recording real experimental runs of Intrusion Guard across surveillance clips under diverse real-world conditions (daylight, night, partial occlusion, varying traffic).

---

## 1. Test Environment

Fill in your actual hardware and software specifications prior to submitting:

| Parameter | Value |
|---|---|
| Processor (CPU) | TODO (e.g. AMD Ryzen 5 5600H / Intel Core i5-11400H) |
| System Memory (RAM) | TODO (e.g. 16 GB DDR4) |
| Operating System | TODO (e.g. Windows 11 Home 64-bit) |
| Python Version | TODO (e.g. Python 3.11.9) |
| Object Detection Model | TODO (e.g. yolo11n.pt) |
| Inference Resolution (`imgsz`) | TODO (e.g. 480) |
| Dwell Time Threshold (`dwell_seconds`) | TODO (e.g. 2.0s) |
| Grace Period (`grace_seconds`) | TODO (e.g. 1.0s) |
| Confidence Threshold (`confidence`) | TODO (e.g. 0.35) |

---

## 2. Experimental Results Table

### Column Definitions
- **Clip:** Filename of the test clip in `clips/`.
- **Scenario:** Specific real-world activity being tested.
- **Lighting:** Lighting environment (Daylight, Overcast, Twilight, Night / Artificial).
- **Real intrusions:** Ground-truth count of unauthorized objects that entered and stayed past dwell time.
- **Alerts fired:** Total number of alerts emitted by Intrusion Guard.
- **Correct (TP):** Legitimate intrusions correctly flagged.
- **False alerts (FP):** Erroneous alerts on allowed objects, outside objects, or spurious detections.
- **Missed (FN):** Intruders that stayed past dwell time but were not flagged.
- **First alert at (s):** Video timestamp when the initial intrusion alert fired.
- **Avg FPS:** Average processing frames per second reported in `summary.json`.
- **Notes:** Observations on tracker stability, occlusion, or detection behavior.

### Benchmark Data

| Clip | Scenario | Lighting | Real intrusions | Alerts fired | Correct | False alerts | Missed | First alert at (s) | Avg FPS | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `clip_01_person_day.mp4` | Person enters restricted zone | Daytime | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `clip_02_person_outside.mp4` | Person walks past outside zone | Daytime | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `clip_03_car_gate.mp4` | Car drives through vehicle gate | Daytime | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `clip_04_car_restricted.mp4` | Car enters pedestrian walkway | Daytime | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `clip_05_dwell_touch.mp4` | Person touches zone boundary briefly | Daytime | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `clip_06_night_surveillance.mp4` | Person enters zone at night | Night / Low-light | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `clip_07_grace_reentry.mp4` | Person leaves zone and returns | Daytime | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `clip_08_crowded_perimeter.mp4` | Multiple people and cars moving | Daytime | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

---

## 3. Findings and Failure Analysis

- **Lighting sensitivity:** TODO (Document how detection confidence and tracking stability changed between daytime and night/evening clips).
- **Occlusion and tracking stability:** TODO (Document whether ByteTrack IDs remained stable when objects were partially occluded by posts, trees, or fences).
- **Dwell time and grace period effectiveness:** TODO (Document whether the 2.0s dwell and 1.0s grace period effectively suppressed false alarms caused by momentary boundary crossing).
- **Processing speed on CPU:** TODO (Document average FPS across clips and whether real-time frame skipping was triggered).

