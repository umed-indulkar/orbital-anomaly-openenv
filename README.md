---
title: Orbital Anomaly OpenEnv
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🛰️ Orbital Anomaly OpenEnv — Version 2.1

> **You are the last line of defense for a €500M spacecraft.**  
> The satellite is 400km above Earth. It has **36 decision windows** before the batteries die.  
> You cannot ask for help — the ground station is out of view.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-v0.2.3-blue)](https://github.com/huggingface/openenv)
[![HF Space](https://img.shields.io/badge/🤗_Live_Demo-Space-orange)](https://codequasar-orbital-anomaly-openenv.hf.space)
[![Colab Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/umed-indulkar/orbital-anomaly-openenv/blob/main/Orbital_Anomaly_openenv.ipynb)

A **spacecraft digital-twin mission-control benchmark** built with **OpenEnv, FastAPI, Docker, and Hugging Face Spaces**.

V2.1 upgrades: **Extended Mission Mode** (36-step long-horizon), **Multi-Agent Commander**, **13-fault Belief State** (world model), full Round 2 theme alignment.

---

## 🌍 Real-World Motivation

Modern satellite operations centers face exactly these challenges every day:

- A stuck MPPT controller reduces solar charging → battery drain → heater shutdown in eclipse → avionics reset cascade
- Reaction wheel saturation causes attitude drift → poor sun alignment → thermal instability → comms degradation
- Ground station blackout windows prevent uplink during critical anomaly windows
- Active science observation windows create genuine tradeoffs: disabling payload saves the spacecraft but wastes the imaging pass

This benchmark converts the actual **causal graph of spacecraft anomaly response** into a typed, deterministic, fully reproducible OpenEnv environment.

---

## 🎯 Round 2 Theme Alignment

| Theme | Alignment | How |
|-------|-----------|-----|
| **Theme 3.1 — World Modeling** (Primary) | ⭐⭐⭐ | 13-fault latent belief state, partial observability, causal subsystem reasoning, belief update every step |
| **Theme 2 — Long-Horizon Planning** | ⭐⭐⭐ | 36-step Extended Mission Mode, 3 anomaly phases, inter-phase state persistence |
| **Theme 1 — Multi-Agent (Fleet AI bonus)** | ⭐⭐ | MissionCommander + EPS/Thermal/Comms specialist agents, oversight architecture |

**Scaler AI Labs bonus**: Spacecraft mission operations is an enterprise workflow — physical KPIs (battery SOC, thermal margins), incident management (fault response), resource allocation (science vs survivability). Directly maps to *"complex workflows with business rule nuances in a large enterprise."*

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT LAYER                                 │
│  MissionCommanderAgent (oversight — Fleet AI bonus)                 │
│    ├── EPSSpecialistAgent      (battery + solar subsystem)           │
│    ├── ThermalSpecialistAgent  (thermal + payload management)        │
│    └── CommsSpecialistAgent    (RF chain + communications)           │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ action
┌────────────────────────▼─────────────────────────────────────────────┐
│              EXTENDED MISSION MODE — 36 STEPS                        │
│  Phase 0 (steps 01-12): EPS Crisis — Solar misalignment, battery    │
│  Phase 1 (steps 13-24): Thermal Crisis — Payload heat spike         │
│  Phase 2 (steps 25-36): Comms Crisis — RF chain degradation         │
│  [battery_soc + payload_temp CARRY OVER between phases]              │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────────┐
│              SPACECRAFT SIMULATOR V2.1                               │
│  EPS: power-balance physics (solar input, bus drain, SOC)            │
│  ADCS: cosine solar alignment, reaction wheel saturation             │
│  Thermal: 3-zone propagation (payload/avionics/battery)              │
│  RF Comms: transponder + antenna pointing chain                      │
│  Orbital: eclipse cycles, ground station windows, radiation zones    │
│  13-Fault Latent Graph: delayed cascading failures (hidden)          │
│  Partial Observability: 5 sensor dropout patterns                   │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ metadata.fault_beliefs
┌────────────────────────▼─────────────────────────────────────────────┐
│              FAULT BELIEF STATE (World Model)                        │
│  Heuristic posterior over 13 faults from observable symptoms.        │
│  Agent maintains and updates beliefs about hidden fault state.       │
│  Exposed in obs.metadata["fault_beliefs"] every step.               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Environment Specification

### Observation Space

Typed Pydantic model. Fields marked * may carry sentinel values (`-999`, `-1`, `None`) when sensors are in dropout — agent must handle missing data gracefully.

**V1 fields (grader backward compatibility)**

| Field | Range | Description |
|-------|-------|-------------|
| `battery_level` | [0, 100] | Battery SOC (%) |
| `solar_efficiency` | [0, 1] | `sun_vector × panel_health` |
| `thermal_temp` | [−20, 120] | Payload zone temperature (°C) |
| `comms_signal` | [0, 1] | `1 − 5×BER − packet_loss` |
| `payload_on` | bool | Science payload active |
| `safe_mode` | bool | Emergency safe mode engaged |
| `task_id` | str | easy / medium / hard |
| `mission_status` | str | stable / warning / critical |

**V2 Extended Telemetry** (40+ fields across EPS / ADCS / Thermal / Comms / Orbital)

| Subsystem | Key Fields |
|-----------|------------|
| EPS | `battery_soc`, `bus_voltage`, `panel_health`, `charge_controller_health`, `power_bus_redundancy` |
| ADCS | `attitude_error_deg`, `sun_vector_alignment`, `reaction_wheel_momentum`, `wheel_saturation_level`, `gyro_bias*`, `star_tracker_available*` |
| Thermal | `battery_temp`, `payload_temp`, `avionics_temp*`, `radiator_efficiency`, `thermal_loop_health` |
| Comms | `antenna_pointing_error`, `bit_error_rate`, `uplink_margin*`, `packet_loss_ratio`, `command_latency_ms*` |
| Orbital | `sunlit`, `eclipse_timer`, `ground_station_visible`, `radiation_zone`, `observation_window_active` |

**V2.1 Metadata** (in every observation)

| Key | Description |
|-----|-------------|
| `phase` | Current anomaly phase (0 / 1 / 2) |
| `phase_step` | Step within current phase (0-11) |
| `phase_scores` | Running avg reward per completed phase |
| `fault_beliefs` | Dict of top-5 fault posterior probabilities |
| `active_fault_count` | Number of latent faults active (not names) |

### Action Space

| Action | Effect |
|--------|--------|
| `rotate_to_sun` | −25° attitude error, +0.18 solar alignment |
| `disable_payload` | Off payload, −6°C thermal |
| `reboot_comms` | BER ×0.35, PLR ×0.35, restore uplink |
| `enter_safe_mode` | Emergency stabilize, disable payload |
| `switch_power_bus` | +18% SOC (primary) or +8% (degraded bus) |
| `noop` | No action |

### Reward Function

Dense multi-objective mission utility **strictly in (0, 1)**:

```python
reward = safe_map(
    0.30 * eps_score          # battery SOC + bus voltage
  + 0.22 * thermal_score      # 3-zone temperatures within limits
  + 0.18 * adcs_score         # attitude error + wheel saturation
  + 0.15 * comms_score        # BER + packet loss
  + 0.15 * survivability      # catastrophe multiplier
) * survivability
+ science_bonus(0.12)         # payload active during obs window

# Survivability: x0.4 if SOC<10%, x0.7 if SOC<20%, x0.5 if avionics>80°C
# Mapping: reward = 0.001 + raw * 0.998  (strict open interval)
```

---

## 🔄 Extended Mission Mode

```
Phase 0: EPS Crisis        Phase 1: Thermal Crisis    Phase 2: Comms Crisis
┌────────────────┐         ┌────────────────┐         ┌────────────────┐
│ Steps 01-12    │ ──────▶ │ Steps 13-24    │ ──────▶ │ Steps 25-36    │
│ SOC: 22-38%   │         │ TEMP: spikes   │         │ BER: 0.18+     │
│ Solar degraded │         │   to 79°C+     │         │ PLR: 0.45+     │
│ Fault: mppt   │         │ Fault: radiator│         │ Fault: transpdr│
└────────────────┘         └────────────────┘         └────────────────┘
         ↑─────────── battery_soc + payload_temp CARRY OVER ───────────↑
```

Poor decisions in Phase 0 → lower starting SOC in Phase 1. Thermal damage in Phase 1 → degraded comms in Phase 2. Genuine 36-step causal chain requiring long-horizon reasoning.

---

## 🤖 Multi-Agent Architecture

```python
# inference.py — MissionCommanderAgent
from inference import mission_commander_decide, compute_fault_beliefs

obs = env.reset(task_id="hard").observation
action, rationale = mission_commander_decide(obs)
# → ("switch_power_bus", "[EPS_Specialist|95%] CRITICAL: battery at minimum")

beliefs = compute_fault_beliefs(obs)
# → {"mppt_stuck": 0.78, "radiator_valve_stuck": 0.60, ...}
```

**Fleet AI bonus alignment**: Commander monitors all specialist agents, explains delegation decisions — exactly *"oversight agents that monitor, analyze, and explain the behavior of other AI agents."*

---

## 📊 Benchmark Tasks

### 🟢 Easy — EPS Crisis
Initial: `battery_soc=38%`, `attitude_error=42°`, sunlit, 1 fault (`mppt_stuck`)  
Challenge: Restore solar alignment, recover battery. Single fault, no eclipse.  
Heuristic baseline: **0.51**

### 🟡 Medium — Thermal Overload + Active Science Window
Initial: `payload_temp=68°C`, `radiator_efficiency=0.55`, `observation_window_active=True`, 2 faults  
Challenge: Thermal tradeoff against science reward, two interacting faults.  
Heuristic baseline: **0.44**

### 🔴 Hard — Cascading Multi-System Failure
Initial: `battery_soc=22%`, eclipse, GS blackout, radiation zone, 7 faults, `star_tracker=False`  
Challenge: Belief-state planning, 5 dropout fields, no ground contact.  
Heuristic baseline: **0.31**

---

## 📈 Training Results

| Task | Pre-Training | Post-Training (GRPO, 80 eps, Qwen-1.5B) | Δ |
|------|-------------|------------------------------------------|----|
| Easy (36-step) | 0.288 | 0.445 | **+54.5%** |
| Medium | 0.44 | 0.52 | **+18.2%** |
| Hard | 0.31 | 0.38 | **+22.6%** |

**What the agent learned:**
- Proactive solar alignment (step 1, not after battery crisis)
- Eclipse detection → `switch_power_bus` instead of useless `rotate_to_sun`
- Thermal cascade prevention: disable payload at 65°C, not 80°C

See full training notebook: [`Orbital_Anomaly_openenv.ipynb`](Orbital_Anomaly_openenv.ipynb)

---

## 🚀 Quickstart

```python
from client import OrbitalAnomalyOpenenvEnv
from models import OrbitalAnomalyOpenenvAction

with OrbitalAnomalyOpenenvEnv(
    base_url="https://codequasar-orbital-anomaly-openenv.hf.space"
).sync() as env:
    result = env.reset(task_id="hard")
    obs = result.observation
    print(f"Phase: {obs.metadata['phase']} | Battery: {obs.battery_soc:.1f}%")
    print(f"Fault beliefs: {obs.metadata['fault_beliefs']}")

    for step in range(36):
        action = OrbitalAnomalyOpenenvAction(action_type="rotate_to_sun")
        result = env.step(action)
        if result.done:
            break
```

```bash
# HTTP
curl -X POST https://codequasar-orbital-anomaly-openenv.hf.space/reset \
  -H "Content-Type: application/json" -d '{"task_id": "hard"}'

curl -X POST https://codequasar-orbital-anomaly-openenv.hf.space/step \
  -H "Content-Type: application/json" -d '{"action_type": "rotate_to_sun"}'

# Baseline
python inference.py
# Or with LLM:
HF_TOKEN=hf_... python inference.py
```

---

## 💻 Local Setup

```bash
git clone https://github.com/umed-indulkar/orbital-anomaly-openenv.git
cd orbital-anomaly-openenv
uv sync
openenv validate
uv run server   # → http://localhost:8000
```

---

## 🐳 Docker

```bash
docker build -t orbital-anomaly-openenv .
docker run -p 8000:8000 orbital-anomaly-openenv
```

---

## 🧪 Tests

```bash
python test_reward.py
# ✅ ALL V2 TESTS PASSED
```

---

## 📁 Repository Structure

```
orbital-anomaly-openenv/
├── server/
│   ├── app.py                               # FastAPI + OpenEnv server
│   └── orbital_anomaly_openenv_environment.py  # V2.1 simulator (THIS FILE)
├── models.py                                # Pydantic typed models
├── client.py                                # Typed Python client
├── inference.py                             # Multi-agent baseline + LLM policy
├── test_reward.py                           # Full test suite
├── Orbital_Anomaly_openenv.ipynb            # GRPO training notebook
├── openenv.yaml                             # OpenEnv manifest
├── pyproject.toml                           # Dependencies
├── Dockerfile
└── README.md
```

---

## ✅ OpenEnv Compliance

| Requirement | Status |
|-------------|--------|
| `reset()` / `step()` / `state` interface | ✅ |
| Typed Pydantic models | ✅ |
| 3+ tasks (easy/medium/hard) | ✅ |
| Rewards strictly in (0, 1) | ✅ epsilon-bounded |
| Task cycling counter | ✅ |
| Explicit task_id independent of counter | ✅ |
| `SUPPORTS_CONCURRENT_SESSIONS = True` | ✅ |
| LiteLLM proxy handshake | ✅ |
| `[START]`/`[STEP]`/`[END]` log format | ✅ |
| FastAPI + Docker + HF Spaces | ✅ |
| `openenv validate` passes | ✅ |

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🤗 Live Space | https://codequasar-orbital-anomaly-openenv.hf.space |
| 📖 API Docs | https://codequasar-orbital-anomaly-openenv.hf.space/docs |
| 📓 Training Notebook | [Colab](https://colab.research.google.com/github/umed-indulkar/orbital-anomaly-openenv/blob/main/Orbital_Anomaly_openenv.ipynb) |
| 💾 GitHub | https://github.com/umed-indulkar/orbital-anomaly-openenv |