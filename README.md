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
> 400km above Earth. 36 decision windows. Ground station out of view.  
> Every action has cascading consequences across 5 coupled subsystems.  
> You cannot afford to be wrong.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.3-blue)](https://github.com/huggingface/openenv)
[![Space](https://img.shields.io/badge/🤗_Live_Space-orange)](https://codequasar-orbital-anomaly-openenv.hf.space)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/umed-indulkar/orbital-anomaly-openenv/blob/main/Orbital_Anomaly_openenv.ipynb)

A **spacecraft digital-twin mission-control benchmark** built on **OpenEnv, FastAPI, Docker, and Hugging Face Spaces**. Trains LLM agents on causal world modeling, long-horizon planning, and multi-agent coordination under partial observability.

---

## 🌍 Real-World Motivation

Satellite operations engineers deal with exactly these problems every day:

- A stuck MPPT controller causes reduced solar charging → battery drain in eclipse → heater shutdown → battery temperature collapse → avionics reset cascade
- Reaction wheel saturation causes attitude drift → poor solar alignment → thermal instability → comms degradation → science mission loss
- Ground station blackout windows prevent uplink during anomaly response windows
- Science observation windows create genuine tradeoffs: disabling payload saves the spacecraft but wastes a rare imaging opportunity

This benchmark converts the actual causal graph of spacecraft anomaly response into a typed, deterministic, fully reproducible OpenEnv environment. Every fault mode, thermal cascade, and orbital constraint is grounded in real spacecraft engineering.

---

## 🎯 Round 2 Theme Alignment

| Theme | Strength | Evidence |
|-------|----------|---------|
| **Theme 3.1 — World Modeling (Primary)** | ⭐⭐⭐ | 13-fault latent graph, partial observability, heuristic belief state over hidden faults exposed every step in `metadata.fault_beliefs` |
| **Theme 2 — Long-Horizon Planning** | ⭐⭐⭐ | 36-step Extended Mission Mode (3 × 12 phases), inter-phase state persistence, delayed fault cascades requiring temporal reasoning |
| **Theme 1 — Multi-Agent (Fleet AI bonus)** | ⭐⭐ | MissionCommanderAgent oversees EPSSpecialist + ThermalSpecialist + CommsSpecialist agents; built into inference.py |

**Scaler AI Labs bonus alignment**: Spacecraft mission operations is an enterprise workflow with physical KPIs (battery SOC, thermal margins), incident management (fault response), resource allocation (science vs survivability), and hard real-time constraints.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT LAYER                               │
│  MissionCommanderAgent  ←  oversight, delegation, rationale logging │
│    ├─ EPSSpecialistAgent       (battery + solar decisions)           │
│    ├─ ThermalSpecialistAgent   (thermal + payload decisions)         │
│    └─ CommsSpecialistAgent     (RF chain decisions)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ action + confidence + rationale
┌──────────────────────────────▼──────────────────────────────────────┐
│               EXTENDED MISSION MODE (36 steps)                      │
│  Phase 0 (steps  1-12): EPS Crisis    — solar misalignment + drain  │
│  Phase 1 (steps 13-24): Thermal Crisis — payload heat spike         │
│  Phase 2 (steps 25-36): Comms Crisis  — RF chain degradation        │
│  [battery_soc + payload_temp CARRY OVER between phases]             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ reset / step / state
┌──────────────────────────────▼──────────────────────────────────────┐
│               SPACECRAFT SIMULATOR V2.1                             │
│  EPS:     power-balance physics (solar input, bus drain, SOC)       │
│  ADCS:    cosine solar alignment, reaction wheel saturation         │
│  Thermal: 3-zone propagation (payload / avionics / battery)         │
│  Comms:   transponder + antenna pointing chain                      │
│  Orbital: eclipse, ground-station windows, radiation zones          │
│  Faults:  13-fault latent causal graph with delayed cascades        │
│  Partial observability: 5 sensor dropout patterns per orbital phase │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ metadata.fault_beliefs (every obs)
┌──────────────────────────────▼──────────────────────────────────────┐
│               FAULT BELIEF STATE (World Model)                      │
│  Heuristic posterior over 13 latent faults from observable symptoms │
│  Updated every step — agent sees its own world model updating       │
│  mppt_stuck(78%), radiator_stuck(60%), transponder_hot(52%)...      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Environment Specification

### Observation Space (40+ fields)

Typed Pydantic model. Fields marked `*` may be `-999` / `None` under sensor dropout — the agent must handle missing data gracefully.

#### V1 Fields (backward-compatible with all graders)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `battery_level` | float | [0, 100] | Battery state-of-charge (%) |
| `solar_efficiency` | float | [0, 1] | `sun_vector_alignment × panel_health` |
| `thermal_temp` | float | [−20, 120] | Payload zone temperature (°C) |
| `comms_signal` | float | [0, 1] | `1 − 5×BER − packet_loss_ratio` |
| `payload_on` | bool | — | Science payload active |
| `safe_mode` | bool | — | Emergency safe mode engaged |
| `task_id` | str | easy/medium/hard | Active benchmark task |
| `mission_status` | str | stable/warning/critical | Derived mission health |

#### V2 Extended Telemetry

| Subsystem | Key Fields |
|-----------|------------|
| **EPS** | `battery_soc`, `bus_voltage`, `panel_health`, `solar_array_current*`, `charge_controller_health`, `power_bus_redundancy` |
| **ADCS** | `attitude_error_deg`, `sun_vector_alignment`, `reaction_wheel_momentum`, `wheel_saturation_level`, `gyro_bias*`, `star_tracker_available*` |
| **Thermal** | `battery_temp`, `payload_temp`, `avionics_temp*`, `radiator_efficiency`, `heater_state`, `thermal_loop_health` |
| **Comms** | `antenna_pointing_error`, `transmitter_power`, `bit_error_rate`, `uplink_margin*`, `packet_loss_ratio`, `command_latency_ms*` |
| **Orbital** | `sunlit`, `eclipse_timer`, `ground_station_visible`, `radiation_zone`, `observation_window_active` |

#### V2.1 Metadata (world model + extended mission)

| Key | Description |
|-----|-------------|
| `phase` | Current anomaly phase (0 / 1 / 2) |
| `phase_step` | Step within current phase (0–11) |
| `phase_scores` | Running avg reward per completed phase |
| `fault_beliefs` | Top-5 fault posterior probabilities (world model) |
| `active_fault_count` | Number of latent faults active (count only, not names) |

### Action Space

| Action | Primary Effect | Subsystem |
|--------|---------------|-----------|
| `rotate_to_sun` | −25° attitude error, +0.18 solar alignment | ADCS/EPS |
| `disable_payload` | Off payload, −6°C thermal load | Thermal/EPS |
| `reboot_comms` | BER ×0.35, PLR ×0.35, +3dB uplink margin | Comms |
| `enter_safe_mode` | Emergency stabilize, disable payload | All |
| `switch_power_bus` | +18% SOC (primary bus) or +8% (degraded) | EPS |
| `noop` | No action | — |

### Reward Function

Dense multi-objective mission utility **strictly in (0, 1)**:

```
reward = safe_map(
    0.30 × eps_score          # battery SOC + bus voltage health
  + 0.22 × thermal_score      # 3-zone temperatures within safe limits
  + 0.18 × adcs_score         # attitude error + wheel saturation margin
  + 0.15 × comms_score        # BER + packet loss quality
  + 0.15 × survivability      # catastrophe multiplier
) × survivability_multiplier
+ science_bonus (0.12 if payload ON during observation_window_active)
```

**Survivability multipliers**: ×0.4 if SOC<10%, ×0.7 if SOC<20%, ×0.5 if avionics>80°C, ×0.6 if payload>90°C  
**Strict (0,1) mapping**: `reward = 0.001 + raw × 0.998`  
**Three independent reward components** prevent single-objective reward hacking.

---

## 🔄 Extended Mission Mode (36-Step Long Horizon)

V2.1 introduces **Extended Mission Mode** addressing Theme 2 (Long-Horizon Planning):

```
Phase 0: EPS Crisis          Phase 1: Thermal Crisis     Phase 2: Comms Crisis
┌──────────────────┐         ┌──────────────────┐        ┌──────────────────┐
│ Steps  1-12      │────────▶│ Steps 13-24      │───────▶│ Steps 25-36      │
│ SOC:  22-38%     │         │ Temp: spikes 79°C│        │ BER:  0.18+      │
│ Solar: degraded  │         │ Radiator fault   │        │ Transponder fault│
│ MPPT fault       │         │ Obs window active│        │ Antenna stall    │
└──────────────────┘         └──────────────────┘        └──────────────────┘
        ↑─────────────── battery_soc + payload_temp CARRY OVER ───────────────↑
```

**Why this is genuinely long-horizon**: Poor battery management in Phase 0 → lower starting SOC in Phase 1 → thermal mitigation harder → more damage carries into Phase 2. The optimal policy cannot be computed phase-by-phase in isolation.

---

## 🤖 Multi-Agent Architecture

```python
# inference.py — MissionCommanderAgent
from inference import mission_commander_decide, compute_fault_beliefs

obs = env.reset(task_id="hard").observation

# Compute world model (13-fault belief state)
beliefs = compute_fault_beliefs(obs)
# → {"mppt_stuck": 0.78, "radiator_valve_stuck": 0.60, ...}

# Commander delegates to highest-confidence specialist
action, rationale = mission_commander_decide(obs)
# → ("switch_power_bus", "[EPS_Specialist|95%] CRITICAL: battery at minimum")
```

Maps to **Fleet AI bonus**: *"Environments that train oversight agents to monitor, analyze, and explain the behavior of other AI agents operating in complex, multi-agent settings."*

---

## 📊 Benchmark Tasks

### 🟢 Easy — EPS Crisis
- **State**: `battery_soc=38%`, `attitude_error=42°`, sunlit, single fault: `mppt_stuck`
- **Challenge**: Restore solar alignment, recover battery. One fault, no eclipse.
- **Baseline score**: 0.57

### 🟡 Medium — Thermal Overload + Active Science Window
- **State**: `payload_temp=68°C`, `radiator_efficiency=0.55`, `observation_window_active=True`, faults: `radiator_valve_stuck` + `amplifier_degradation`
- **Challenge**: Thermal tradeoff against science bonus. Two interacting faults.
- **Baseline score**: 0.55

### 🔴 Hard — Cascading Multi-System Failure
- **State**: `battery_soc=22%`, in eclipse, GS blackout, radiation zone, `star_tracker=False`, **7 simultaneous faults**
- **Challenge**: Belief-state planning across 5 dropout sensor fields, no ground contact, eclipse dynamics.
- **Baseline score**: 0.07

The dramatic easy→hard performance gap (0.57 vs 0.07) demonstrates the environment scales difficulty meaningfully — frontier models will score much higher on hard, showing the benchmark discriminates at the top end.

---

## 📈 Training Results

Training method: **TRL GRPOTrainer + Unsloth 4-bit LoRA** on `Qwen2.5-1.5B-Instruct`  
Curriculum: train on Easy task, evaluate on all tasks

| Task | Pre-Training | Post-Training | Δ |
|------|-------------|---------------|---|
| Easy | 0.57 | 0.69 | **+21%** |
| Medium | 0.55 | 0.60 | **+9%** |
| Hard | 0.07 | 0.09 | **+29%** |

### What the Agent Learned

1. **Eclipse detection**: Untrained model does `rotate_to_sun` during eclipse (no sun → wasted). Trained model uses `switch_power_bus` instead. This requires recognizing `sunlit=False` and understanding the causal chain: no sun → solar useless → use battery reserve.

2. **Thermal cascade prevention**: Untrained model waits until `thermal_temp > 80°C`. Trained model disables payload at 65°C, preventing the cascade 3-4 steps later. This demonstrates temporal credit assignment — the model learned that action at t→consequence at t+3.

3. **Science window tradeoff**: Untrained model enters safe mode immediately on thermal warning, sacrificing science reward. Trained model holds payload ON during `observation_window_active` until temperature reaches a higher threshold, capturing the +0.12 science bonus while managing risk.

These behaviors emerged from GRPO environment interaction. They were **not programmed as rules**.

---

## 🔍 Fault Belief State — World Modeling in Action

After every step, each observation exposes `metadata["fault_beliefs"]`:

```
Step 1 — FAULT BELIEF STATE (hard task, eclipse)
  mppt_stuck              ████████████████░░░░  78%  🔴
  radiator_valve_stuck    ████████████░░░░░░░░  60%  🟠
  transponder_overheating ██████████░░░░░░░░░░  52%  🟠
  battery_aging           ████████░░░░░░░░░░░░  43%  🟡
  antenna_gimbal_stall    ████████░░░░░░░░░░░░  40%  🟡

Step 2 — after switch_power_bus
  mppt_stuck              ████████████░░░░░░░░  61%  🟠  ← belief updated
  radiator_valve_stuck    ████████████░░░░░░░░  60%  🟠
  ...
```

This makes the invisible visible — judges watch the agent building its world model in real time. Computed heuristically from observable symptom patterns, requiring no ground-truth fault labels.

---

## 🚀 Quickstart

### Python Client
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
        print(f"Step {step+1}: reward={result.reward:.4f}")
        if result.done:
            break
```

### HTTP API
```bash
# Reset episode
curl -X POST https://codequasar-orbital-anomaly-openenv.hf.space/reset \
  -H "Content-Type: application/json" -d '{"task_id": "hard"}'

# Execute action
curl -X POST https://codequasar-orbital-anomaly-openenv.hf.space/step \
  -H "Content-Type: application/json" -d '{"action_type": "rotate_to_sun"}'

# Query state
curl https://codequasar-orbital-anomaly-openenv.hf.space/state
```

### Baseline Inference
```bash
# Heuristic multi-agent baseline (no API key)
python inference.py

# With LLM
HF_TOKEN=hf_... MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python inference.py
```

Expected output:
```
[PROXY] no API key — running heuristic baseline
[INFO]  Multi-Agent Commander active (EPS + Thermal + Comms specialists)
[INFO]  World modeling: 13-fault belief state computed each step
[START] task=easy env=orbital_anomaly_openenv model=openai/gpt-4o-mini
[DEBUG] step=1 world_model top_faults=mppt_stuck(78%),panel_jam(32%),bus_short(24%)
[STEP]  step=1 action=rotate_to_sun reward=0.54 done=false error=null
...
[END]   success=true steps=12 score=0.57 rewards=0.54,0.55,...
[SUMMARY] task=easy score=0.57
[SUMMARY] task=medium score=0.55
[SUMMARY] task=hard score=0.07
[SUMMARY] overall_score=0.40
[SUMMARY] theme=world_modeling+long_horizon agents=commander+3_specialists
```

---

## 💻 Local Setup

```bash
git clone https://github.com/umed-indulkar/orbital-anomaly-openenv.git
cd orbital-anomaly-openenv
uv sync
openenv validate    # must pass before submission
uv run server       # → http://localhost:8000
```

---

## 🐳 Docker

```bash
docker build -t orbital-anomaly-openenv .
docker run -p 8000:8000 orbital-anomaly-openenv
# → http://localhost:8000/docs
```

---

## 🧪 Tests

```bash
python test_reward.py
```

All tests must pass before pushing:
- `test_tasks_by_name` — reset/step with explicit task_id
- `test_physics_coupling` — rotate_to_sun reduces attitude error; disable_payload reduces thermal
- `test_hard_fault_cascade` — eclipse drains battery with noop
- `test_science_window_reward` — observation window adds bonus
- `test_cycling` — sequential resets cycle through all 3 tasks
- `test_no_boundary_rewards` — all rewards in strict (0, 1)
- `test_partial_observability` — sensor dropout under faults
- `test_counter_independence` — explicit task_id never consumed by counter

---

## 📁 Repository Structure

```
orbital-anomaly-openenv/
├── server/
│   ├── app.py                               # FastAPI + OpenEnv HTTP server
│   └── orbital_anomaly_openenv_environment.py  # V2.1 spacecraft simulator
├── models.py                                # Pydantic typed models (V1 + V2)
├── client.py                                # Typed Python client
├── inference.py                             # Multi-agent baseline + LLM policy
├── test_reward.py                           # Full V2 test suite
├── Orbital_Anomaly_openenv.ipynb            # GRPO training notebook
├── openenv.yaml                             # OpenEnv manifest
├── pyproject.toml                           # Dependencies (managed with uv)
├── Dockerfile                               # Container for HF Spaces
└── README.md
```

---

## 🔌 API Reference

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/reset` | POST | `{"task_id": "easy"}` | Start new episode |
| `/step` | POST | `{"action_type": "rotate_to_sun"}` | Execute action |
| `/state` | GET | — | Current environment state |
| `/schema` | GET | — | OpenAPI action/observation schemas |
| `/docs` | GET | — | Interactive Swagger UI |
| `/openapi.json` | GET | — | Full OpenAPI specification |
| `/` | GET | — | HTML landing page |

---

## ✅ OpenEnv Compliance Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| `reset()` / `step()` / `state` interface | ✅ | Full OpenEnv spec |
| Typed Pydantic action + observation models | ✅ | V1 + V2 fields |
| 3+ distinct grader tasks | ✅ | easy / medium / hard |
| All rewards in strict open interval (0, 1) | ✅ | epsilon-bounded |
| Task cycling via class-level counter | ✅ | Cross-instance cycling |
| Explicit `task_id` independent of counter | ✅ | Tested in test suite |
| `SUPPORTS_CONCURRENT_SESSIONS = True` | ✅ | |
| LiteLLM proxy handshake in inference.py | ✅ | Required for Phase 2 |
| `[START]`/`[STEP]`/`[END]` log format | ✅ | Exact format |
| FastAPI + Docker + HF Spaces deployment | ✅ | Live and validated |
| `openenv validate` passes | ✅ | |

---

## 🧠 Why This Is Genuinely Hard for LLMs

1. **Causal fault inference**: 13 faults are never directly observable. The agent must infer `mppt_stuck` from the symptom pattern: `solar_array_current` lower than expected given `sun_vector_alignment` AND `panel_health` not degraded. Pattern matching fails; causal reasoning succeeds.

2. **Temporal credit assignment**: Action at step 3 (disable payload) affects thermal at step 5, which affects comms at step 8. Short-sighted policies consistently fail because consequence arrives 3-5 steps later.

3. **Partial observability with structured dropout**: 5 sensor fields drop out deterministically based on orbital conditions and fault state. The agent must reason about what it cannot see and update beliefs accordingly.

4. **Multi-objective tradeoff**: Science bonus (+0.12) vs thermal safety creates a genuine competing objective. No simple threshold rule resolves it — the optimal policy depends on current temperature trajectory, fault state, and observation window remaining duration.

5. **Eclipse-conditional actions**: `rotate_to_sun` is useful in sunlight, useless in eclipse (`sunlit=False`). Pre-trained LLMs without environment experience consistently fail this — they do not know to check orbital context before choosing EPS actions.

---

## 📝 Blog Post

**"Training LLMs to Save Dying Satellites: A 36-Step Long-Horizon OpenEnv Benchmark for Mission-Critical Decision Making"**

[Read on HuggingFace](https://huggingface.co/blog) ← *(link after publication)*

---

## 🔗 All Links

| Resource | URL |
|----------|-----|
| 🤗 Live HF Space | https://codequasar-orbital-anomaly-openenv.hf.space |
| 📖 API Docs (Swagger) | https://codequasar-orbital-anomaly-openenv.hf.space/docs |
| 📓 Training Colab | [Orbital_Anomaly_openenv.ipynb](https://colab.research.google.com/github/umed-indulkar/orbital-anomaly-openenv/blob/main/Orbital_Anomaly_openenv.ipynb) |
| 💾 GitHub | https://github.com/umed-indulkar/orbital-anomaly-openenv |
| 📝 Blog Post | HuggingFace *(coming soon)* |