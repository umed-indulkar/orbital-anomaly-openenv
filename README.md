---
title: Orbital Anomaly OpenEnv
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🛰️ Orbital Anomaly OpenEnv — Version 2.0

A **spacecraft digital-twin mission-control benchmark** built with **OpenEnv, FastAPI, Docker, and Hugging Face Spaces**.

Version 2 replaces the scalar telemetry model of V1 with a full **multi-subsystem spacecraft simulator** featuring physically grounded EPS power-balance dynamics, ADCS attitude control, multi-zone thermal propagation, RF communications chain modeling, orbital context, a 13-fault latent root-cause graph with delayed cascading failures, and deterministic partial observability — all while remaining fully backward compatible with the V1 OpenEnv grader interface.

---

## 🌍 Real-World Motivation

Modern satellite operations centers face exactly these challenges every day:

- A stuck MPPT controller causes reduced solar charging → battery drain → heater shutdown in eclipse → battery temperature collapse → avionics reset cascade
- Reaction wheel saturation causes attitude drift → poor sun alignment → reduced EPS charging → thermal instability → comms degradation
- Ground station blackout windows prevent uplink during critical anomaly windows
- Active science observation windows create genuine tradeoffs: disabling the payload saves the spacecraft but wastes the imaging pass

This benchmark converts the actual causal graph of spacecraft anomaly response into a **typed, deterministic, fully reproducible OpenEnv environment**.

---

## 🎯 Benchmark Objective

The agent acts as an autonomous mission-control AI and must:

1. Interpret multi-subsystem telemetry with partial observability (sensor dropouts)
2. Identify which latent root faults are active from symptom patterns
3. Sequence corrective actions across EPS, ADCS, thermal, and comms subsystems
4. Prevent cascading subsystem failures under delayed fault propagation dynamics
5. Balance spacecraft survivability against science mission objectives during observation windows
6. Stabilise the mission within a 12-step budget

---

## 🧩 Action Space

| Action | Effect |
|---|---|
| `rotate_to_sun` | ADCS attitude correction — reduces attitude error (limited by wheel saturation), improves solar charging and antenna pointing |
| `disable_payload` | Shuts down science payload — immediately reduces payload thermal zone and power draw |
| `reboot_comms` | RF chain reset — reduces BER, packet loss, latency; may clear transponder overheating fault |
| `enter_safe_mode` | Conservative survival hold — disables payload, reduces wheel stress, cools all zones; sacrifices science |
| `switch_power_bus` | Activates redundant bus — injects battery reserve, clears bus short fault |
| `noop` | Take no action — correct choice during stable science windows |

---

## 📡 Observation Space

### V1 Backward-Compatible Fields

| Field | Type | Description |
|---|---|---|
| `battery_level` | float | Battery SOC % (same as `battery_soc`) |
| `solar_efficiency` | float | Effective solar factor [0,1] |
| `thermal_temp` | float | Payload temperature °C (same as `payload_temp`) |
| `comms_signal` | float | Composite comms quality [0,1] |
| `payload_on` | bool | Science payload active |
| `safe_mode` | bool | Safe mode enabled |
| `task_id` | str | `easy` / `medium` / `hard` |
| `mission_status` | str | `stable` / `warning` / `critical` |
| `reward` | float | Per-step reward in (0,1) |
| `done` | bool | Episode terminal flag |
| `metadata` | dict | Step, episode_id, version, obs_dropout |

### V2 EPS Telemetry

| Field | Range | Notes |
|---|---|---|
| `battery_soc` | 0–100 % | State of charge |
| `bus_voltage` | 18–28 V | Nominal 28V; sags under load |
| `panel_health` | 0–1 | Degrades in radiation zone |
| `solar_array_current` | ≥0 A | -1 = telemetry dropout |
| `charge_controller_health` | 0–1 | Reduced by MPPT fault |
| `power_bus_redundancy` | bool | Redundant bus active |

### V2 ADCS Telemetry

| Field | Range | Notes |
|---|---|---|
| `attitude_error_deg` | 0–90° | Error from sun-pointing |
| `sun_vector_alignment` | 0–1 | cos(attitude_error) |
| `reaction_wheel_momentum` | 0–1 | 1.0 = saturated |
| `wheel_saturation_level` | 0–1 | Limits manoeuvre authority |
| `gyro_bias` | deg/s | -999 = dropout (gyro fault + step cadence) |
| `star_tracker_available` | bool/None | None = sensor dropout |

### V2 Multi-Zone Thermal Telemetry

| Field | Range | Notes |
|---|---|---|
| `battery_temp` | -40 to 60°C | Safe: -5 to 35°C |
| `payload_temp` | -40 to 120°C | Safe: <75°C |
| `avionics_temp` | -40 to 100°C | Safe: <70°C; -999 = dropout |
| `radiator_efficiency` | 0–1 | Reduced by stuck radiator fault |
| `thermal_loop_health` | 0–1 | Heat pipe health |
| `heater_state` | bool | Battery heater active |

### V2 RF Communications Telemetry

| Field | Range | Notes |
|---|---|---|
| `antenna_pointing_error` | 0–60° | Driven by attitude error |
| `transmitter_power` | W | Degrades with amplifier fault |
| `bit_error_rate` | 0–1 | Good: <0.01 |
| `uplink_margin` | dB | -99 = GS blackout dropout |
| `packet_loss_ratio` | 0–1 | Good: <0.05 |
| `command_latency_ms` | ms | -1 = dropout |

### V2 Orbital Context

| Field | Type | Description |
|---|---|---|
| `sunlit` | bool | Solar charging possible |
| `eclipse_timer` | int | Steps since eclipse entry |
| `ground_station_visible` | bool | Uplink/downlink available |
| `radiation_zone` | bool | Panel degradation active |
| `observation_window_active` | bool | Science bonus window |

---

## 🔥 Latent Fault Graph

13 root-cause faults are active per task but **never directly observable**. Only telemetry symptoms are visible:

| Fault | Visible Symptoms | Delayed Cascade |
|---|---|---|
| `mppt_stuck` | Low charge_controller_health | Reduced solar charging |
| `panel_deployment_jam` | Falling panel_health | Persistent low solar input |
| `bus_short_transient` | Bus voltage sag | Avionics thermal rise |
| `battery_aging` | Unexplained SOC loss | — |
| `reaction_wheel_saturation` | High wheel_saturation_level | Attitude drift → poor solar |
| `gyro_drift` | Rising gyro_bias | Antenna mispointing after 3 steps |
| `star_tracker_dropout` | star_tracker=None | Attitude error growth |
| `radiator_valve_stuck` | Low radiator_efficiency | Avionics overheating |
| `heat_pipe_failure` | Low thermal_loop_health | Payload thermal runaway |
| `heater_relay_latch` | heater_state=False in eclipse | Battery temperature collapse |
| `transponder_overheating` | Rising BER | Avionics thermal cascade |
| `amplifier_degradation` | Falling transmitter_power | Uplink margin loss |
| `antenna_gimbal_stall` | Rising antenna_pointing_error | Comms degradation |

---

## 🎚️ Tasks

### 🟢 Easy — ADCS Misalignment + Low Battery
- **Initial state**: battery_soc=38%, attitude_error=42°, sunlit
- **Latent fault**: MPPT stuck controller
- **Agent challenge**: rotate to sun, recover battery; single fault, no eclipse

### 🟡 Medium — Thermal Overload + Active Science Window
- **Initial state**: payload_temp=68°C, radiator degraded, observation_window_active=True
- **Latent faults**: stuck radiator valve + amplifier degradation
- **Agent challenge**: thermal tradeoff against science reward; two interacting faults

### 🔴 Hard — Cascading Multi-System Failure
- **Initial state**: SOC=22%, in eclipse, GS blackout, radiation zone, star tracker down
- **Latent faults**: 7 simultaneous faults across all subsystems
- **Agent challenge**: belief-state planning with 5 dropout fields, eclipse dynamics, no GS contact

---

## 🏆 Reward Design

Dense multi-objective mission utility in strict open interval (0, 1):

```
reward = safe_map(
    0.30 × eps_score        (battery SOC + bus voltage)
  + 0.22 × thermal_score    (all 3 zones within safe limits)
  + 0.18 × adcs_score       (attitude error + wheel saturation margin)
  + 0.15 × comms_score      (BER + packet loss quality)
  + 0.15 × survivability    (penalty multiplier for critical states)
) × survivability + science_bonus
```

**Science bonus** (+0.12): payload active during `observation_window_active`  
**Survivability multiplier**: ×0.4 if SOC<10%, ×0.5 if avionics>80°C, ×0.6 if payload>90°C

---

## 🚀 Live Deployment

- **Space**: https://codequasar-orbital-anomaly-openenv.hf.space
- **API Docs**: https://codequasar-orbital-anomaly-openenv.hf.space/docs
- **OpenAPI**: https://codequasar-orbital-anomaly-openenv.hf.space/openapi.json

---

## 💻 Local Setup

```bash
# Install dependencies
uv sync

# Validate OpenEnv spec
openenv validate

# Run locally
uv run server
# → http://localhost:8000
```

---

## 🧪 Baseline Inference

```bash
python inference.py
```

Expected output (heuristic baseline, no API key):
```
[PROXY] no API key found — running heuristic baseline
[START] task=easy env=orbital_anomaly_openenv model=openai/gpt-4o-mini
[STEP] step=1 action=rotate_to_sun reward=0.42 done=false error=null
...
[END] success=true steps=8 score=0.51 rewards=0.42,0.48,...
[SUMMARY] task=easy score=0.51
[SUMMARY] task=medium score=0.44
[SUMMARY] task=hard score=0.31
[SUMMARY] overall_score=0.42
```

---

## 🐳 Docker

```bash
docker build -t orbital-anomaly-openenv .
docker run -p 8000:8000 orbital-anomaly-openenv
```

---

## 📂 Project Structure

```
orbital-anomaly-openenv/
├── __init__.py
├── client.py          ← typed OpenEnv client (V2 fields)
├── models.py          ← Pydantic Action + Observation schemas
├── inference.py       ← baseline inference script
├── openenv.yaml       ← OpenEnv spec manifest
├── Dockerfile
├── README.md
├── pyproject.toml
├── requirements.txt
├── test_reward.py     ← V2 test suite
└── server/
    ├── __init__.py
    ├── app.py         ← FastAPI server
    └── orbital_anomaly_openenv_environment.py  ← V2 digital twin
```

---

## 🧠 Why This Benchmark Matters

Most OpenEnv environments focus on discrete text tasks. This benchmark introduces:

- **Causal subsystem coupling**: actions affect multiple interconnected systems
- **Latent fault diagnosis**: agents must infer hidden root causes from symptom patterns
- **Delayed consequences**: unresolved faults cascade after 2–4 steps
- **Partial observability**: sensor dropouts require belief-state reasoning
- **Mission objective tradeoffs**: survivability vs. science throughput
- **Orbital context dynamics**: eclipse, radiation, GS windows change optimal strategy over time

This makes it immediately useful for evaluating frontier LLM agents, RL policies, and planning systems on real-world long-horizon decision making under uncertainty.

---

## ✅ Pre-Submission Checklist

```bash
openenv validate     # spec compliance
docker build .       # container builds
python test_reward.py  # all V2 tests pass
python inference.py  # baseline scores reproduce
```

Verify endpoints:
```
POST /reset  → 200
POST /step   → 200
GET  /state  → 200
```