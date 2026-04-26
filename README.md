---
title: Orbital Anomaly OpenEnv
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🛰️ Orbital Anomaly OpenEnv v2.2

> **You are the last line of defense for a €500M spacecraft.**  
> 400km above Earth. 36 decision windows before the batteries die.  
> Ground station is out of view. You cannot ask for help.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.3-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Space](https://img.shields.io/badge/🤗_Live_Demo-HF_Space-orange)](https://codequasar-orbital-anomaly-openenv.hf.space)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/umed-indulkar/orbital-anomaly-openenv/blob/main/Orbital_Anomaly_openenv.ipynb)
[![Blog](https://img.shields.io/badge/📝_Blog-Read_Here-lightgrey)](./blog.md)
[![Tests](https://img.shields.io/badge/Tests-8%2F8_passing-brightgreen)](#-tests)

A **spacecraft digital-twin mission-control benchmark** for training LLM agents on **causal world modeling**, **long-horizon planning**, and **multi-agent coordination** under partial observability and delayed fault cascades.

---

## 🎯 Theme Alignment

| Theme | Strength | Evidence |
|-------|----------|---------|
| **Theme 3.1 — World Modeling** (Primary) | ⭐⭐⭐ | 13-fault latent causal graph. Agent must infer hidden fault state from observable symptoms. Fault belief state exposed in every observation's `metadata.fault_beliefs`. |
| **Theme 2 — Long-Horizon Planning** | ⭐⭐⭐ | 36-step Extended Mission Mode (3 phases × 12 steps). Inter-phase state persistence: poor Phase 1 decisions make Phase 2 harder. |
| **Theme 1 — Multi-Agent (Fleet AI bonus)** | ⭐⭐ | `MissionCommanderAgent` oversees `EPSSpecialistAgent`, `ThermalSpecialistAgent`, `CommsSpecialistAgent`. Commander explains delegation decisions. |

**Scaler AI Labs bonus**: Spacecraft mission operations is an enterprise workflow — physical KPIs (battery SOC, thermal margins), incident management (fault response), resource allocation (science vs survivability).

---

## 🌍 Real-World Motivation

Real satellite operations engineers face exactly these problems daily:

- A stuck **MPPT controller** reduces solar charging → battery drain → heater shutdown in eclipse → avionics reset cascade
- **Reaction wheel saturation** causes attitude drift → poor solar alignment → thermal instability → comms degradation
- **Ground station blackout** windows prevent uplink during anomaly response
- Active **science observation windows** create genuine tradeoffs: disabling payload saves the spacecraft but wastes a rare imaging opportunity

Every fault mode, thermal cascade, and orbital constraint in this environment is grounded in real spacecraft engineering.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     MULTI-AGENT LAYER (Theme 1)                     │
│  MissionCommanderAgent  ← oversight, delegation, rationale logging  │
│    ├─ EPSSpecialistAgent       (battery + solar)                     │
│    ├─ ThermalSpecialistAgent   (thermal + payload)                   │
│    └─ CommsSpecialistAgent     (RF chain + comms)                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ action
┌──────────────────────────────▼───────────────────────────────────────┐
│              EXTENDED MISSION MODE — 36 STEPS (Theme 2)             │
│  Phase 0 (steps  1-12): EPS Crisis    — solar misalignment + drain  │
│  Phase 1 (steps 13-24): Thermal Crisis — payload heat spike         │
│  Phase 2 (steps 25-36): Comms Crisis  — RF chain degradation        │
│  [battery_soc + payload_temp CARRY OVER between phases]             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│              SPACECRAFT SIMULATOR v2.2                               │
│  EPS:     power-balance physics (solar input, bus drain, SOC)       │
│  ADCS:    cosine solar alignment, reaction wheel saturation         │
│  Thermal: 3-zone propagation (payload / avionics / battery)         │
│  Comms:   transponder + antenna pointing chain                      │
│  Orbital: eclipse cycles, GS windows, radiation, science windows   │
│  Faults:  13-fault latent causal graph (hidden, cascading)          │
│  Partial observability: 6 sensor dropout patterns                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ metadata.fault_beliefs (every obs)
┌──────────────────────────────▼───────────────────────────────────────┐
│              FAULT BELIEF STATE (Theme 3: World Model)              │
│  Heuristic posterior over 13 faults from observable symptoms.       │
│  Updated every step. Agent reasons about hidden fault state.        │
│  mppt_stuck(78%), radiator_stuck(60%), transponder_hot(52%)...      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Environment Specification

### Observation Space (40+ fields)

**V1 fields** (grader backward-compatible):

| Field | Range | Description |
|-------|-------|-------------|
| `battery_level` | [0, 100] | Battery SOC % |
| `solar_efficiency` | [0, 1] | `sun_vector × panel_health` |
| `thermal_temp` | [−20, 120] | Payload temperature °C |
| `comms_signal` | [0, 1] | `1 − 5×BER − PLR` |
| `payload_on` | bool | Science payload active |
| `safe_mode` | bool | Emergency mode engaged |
| `task_id` | str | easy/medium/hard |
| `mission_status` | str | stable/warning/critical |

**V2 extended telemetry** (fields marked `*` may carry sentinel values under dropout):

| Subsystem | Fields |
|-----------|--------|
| EPS | `battery_soc`, `bus_voltage`, `panel_health`, `solar_array_current*`, `charge_controller_health`, `power_bus_redundancy` |
| ADCS | `attitude_error_deg`, `sun_vector_alignment`, `reaction_wheel_momentum`, `wheel_saturation_level`, `gyro_bias*`, `star_tracker_available*` |
| Thermal | `battery_temp`, `payload_temp`, `avionics_temp*`, `radiator_efficiency`, `heater_state`, `thermal_loop_health` |
| Comms | `antenna_pointing_error`, `transmitter_power`, `bit_error_rate`, `uplink_margin*`, `packet_loss_ratio`, `command_latency_ms*` |
| Orbital | `sunlit`, `eclipse_timer`, `ground_station_visible`, `radiation_zone`, `observation_window_active` |

**V2.2 metadata** (in every observation):

| Key | Description |
|-----|-------------|
| `phase` | Current anomaly phase (0/1/2) |
| `phase_step` | Step within phase (0–11) |
| `phase_scores` | Running avg reward per completed phase |
| `fault_beliefs` | 13-fault posterior probabilities (world model) |
| `active_fault_count` | Count of active latent faults (not names) |
| `dominant_subsystem` | Highest-belief fault subsystem: EPS / ADCS / Thermal / Comms |

### Action Space

| Action | Effect | When to use |
|--------|--------|-------------|
| `rotate_to_sun` | −25° attitude error, +0.18 solar alignment | Sunlit + solar degraded |
| `disable_payload` | Off payload, −6°C thermal | Thermal >65°C |
| `reboot_comms` | BER ×0.35, PLR ×0.35, +3dB uplink | Comms degraded |
| `enter_safe_mode` | Emergency stabilize, disable payload | Thermal >85°C |
| `switch_power_bus` | +18% SOC (or +8% degraded) | Low battery, especially eclipse |
| `noop` | No action | Stable state |

⚠️ **Key rule**: `rotate_to_sun` is **useless in eclipse** (`sunlit=False`). The model must learn this.

### Reward Function

Dense multi-objective mission utility **strictly in (0, 1)**:

```python
reward = safe_map(
    0.30 × eps_score          # battery SOC + bus voltage
  + 0.22 × thermal_score      # 3-zone temperatures within limits
  + 0.18 × adcs_score         # attitude error + wheel saturation
  + 0.15 × comms_score        # BER + packet loss
  + 0.15 × survivability      # catastrophe multiplier
) × survivability
+ science_bonus(0.12)         # payload ON during observation_window_active

# Survivability: ×0.4 if SOC<10%, ×0.7 if SOC<20%, ×0.5 if avionics>80°C
# Mapping: reward = 0.001 + raw × 0.998  (strict open interval, grader-safe)
```

Three independent reward components prevent single-objective reward hacking.

### Partial Observability — Sensor Dropout Map

Six telemetry fields drop out **deterministically** (not randomly), so agents can learn the dropout pattern:

| Dropped Field | Dropout Condition |
|---------------|-------------------|
| `star_tracker_available` | `star_tracker_dropout` fault active |
| `gyro_bias` | `gyro_drift` fault + `step % 4 == 0` |
| `uplink_margin` | `ground_station_visible == False` |
| `command_latency_ms` | `ground_station_visible == False` |
| `avionics_temp` | `step % 3 == 2` |
| `solar_array_current` | `step % 5 == 0` |

---

## 📊 Benchmark Tasks

### 🟢 Easy — EPS Crisis
- **State**: `battery_soc=38%`, `attitude_error=42°`, sunlit, 1 fault: `mppt_stuck`
- **Challenge**: Restore solar alignment, recover battery. Single fault, no eclipse.
- **Baseline**: 0.57 | **Trained LLM**: ~0.70

### 🟡 Medium — Thermal Overload + Science Window
- **State**: `payload_temp=68°C`, `radiator_efficiency=0.55`, `observation_window_active=True`
- **Faults**: `radiator_valve_stuck` + `amplifier_degradation`
- **Challenge**: Thermal tradeoff against +0.12 science bonus. Two interacting faults.
- **Baseline**: 0.55 | **Trained LLM**: ~0.65

### 🔴 Hard — Cascading Multi-System Failure
- **State**: `battery_soc=22%`, eclipse, GS blackout, radiation zone, `star_tracker=False`, 7 simultaneous faults
- **Faults**: `reaction_wheel_saturation`, `gyro_drift`, `star_tracker_dropout`, `heat_pipe_failure`, `heater_relay_latch`, `transponder_overheating`, `mppt_stuck`
- **Challenge**: Belief-state planning, 6 dropout fields, eclipse dynamics, no ground contact.
- **Baseline**: 0.07–0.13

The dramatic easy→hard gap (0.57 vs 0.07) shows the environment meaningfully scales difficulty.

---

## 📈 Training Results

Method: **TRL GRPOTrainer + Unsloth 4-bit LoRA** on `Qwen2.5-1.5B-Instruct`

| Task | Pre-Training | Post-Training | Δ |
|------|-------------|---------------|---|
| Easy | 0.57 | ~0.70 | +23% |
| Medium | 0.55 | ~0.64 | +16% |
| Hard | 0.10 | ~0.10 | generalisation baseline |

### What the Agent Learned

1. **Eclipse detection**: Untrained model does `rotate_to_sun` in eclipse (useless — no sun). Trained model uses `switch_power_bus` instead. Requires tracking `sunlit=False` and understanding the causal chain: no sun → solar useless → use battery reserve.

2. **Thermal cascade prevention**: Untrained model waits until `thermal_temp > 80°C`. Trained model disables payload at 65°C, preventing cascade 3-4 steps later. This is **temporal credit assignment** — the model learned delayed consequences.

3. **Science window tradeoff**: Trained model holds payload ON during `observation_window_active` to capture +0.12 bonus, until temperature reaches a risk threshold. This is a genuine **multi-objective policy**.

---

## 📊 Training Visualizations

All plots are produced automatically when running the Colab notebook. Each visualization cell is self-contained and auto-downloads the PNG.

| File | Notebook Cell | What it shows |
|------|--------------|---------------|
| `images/task_snapshot.png` | Cell 3b | Initial telemetry for all 3 tasks side-by-side |
| `images/baseline_distributions.png` | Cell 6b | Violin + scatter of baseline rewards per task |
| `images/training_analysis.png` | Cell 12b | Training curve + pre/post bars + action heatmaps + improvement % |
| `images/action_policy_heatmap.png` | Cell 13b | Before/after action policy heatmap across all tasks |
| `images/fault_belief_evolution.png` | Cell 14b | 13-fault belief state updating step by step |
| `images/telemetry_timeline_36step.png` | Cell 15b | All 4 subsystems tracked across 36 steps with phase bands |
| `images/final_dashboard.png` | Cell 17b | Complete results dashboard — everything in one figure |

---

## 🔍 Fault Belief State — World Modeling in Action

```
Step 0 — Hard task (eclipse, 7 faults active, none directly observable)

  mppt_stuck              ████████████████░░░░  78%  ← EPS fault inferred from low solar
  radiator_valve_stuck    ████████████░░░░░░░░  62%  ← Thermal fault inferred from high temp
  transponder_overheating ██████████░░░░░░░░░░  53%  ← Comms fault inferred from low signal
  battery_aging           ████████░░░░░░░░░░░░  43%
  heat_pipe_failure       ███████░░░░░░░░░░░░░  38%

Step 1 — after switch_power_bus
  mppt_stuck              ████████████░░░░░░░░  60%  ← belief updated as SOC rises
  ...
```

Judges watch the agent build its world model in real time. Available via `obs.metadata["fault_beliefs"]` every step. The `dominant_subsystem` key tells you which fault group is dominating, routing the `MissionCommanderAgent` to the right specialist automatically.

---

## 🔄 Extended Mission Mode (Theme 2: Long-Horizon)

```
Phase 0: EPS Crisis        Phase 1: Thermal Crisis    Phase 2: Comms Crisis
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Steps  1-12      │──────▶│ Steps 13-24      │──────▶│ Steps 25-36      │
│ SOC:  22-38%     │       │ Temp: 79°C+      │       │ BER:  0.18+      │
│ Solar: degraded  │       │ Radiator fault   │       │ Antenna stall    │
└──────────────────┘       └──────────────────┘       └──────────────────┘
         ↑─────────── battery_soc + payload_temp CARRY OVER ──────────────↑
```

Poor battery management in Phase 0 → lower SOC entering Phase 1 → thermal mitigation harder. The optimal policy **cannot** be computed phase-by-phase in isolation.

---

## 🤖 Multi-Agent Architecture

```python
from inference import mission_commander_decide, compute_fault_beliefs

obs = env.reset(task_id="hard").observation

# World model: 13-fault belief state
beliefs = compute_fault_beliefs(obs)
# → {"mppt_stuck": 0.78, "radiator_valve_stuck": 0.60, ...}

# Commander delegates to highest-confidence specialist
action, rationale, recs = mission_commander_decide(obs)
# → ("switch_power_bus",
#    "[EPS_Specialist|97%] CRITICAL: battery at floor — reserve bus",
#    {"EPS_Specialist": (...), "Thermal_Specialist": (...), "Comms_Specialist": (...)})
```

Every action includes a rationale string logged to stdout. Researchers and judges can inspect agent reasoning in real time.

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
    obs    = result.observation

    print(f"Phase: {obs.metadata['phase']} | SOC: {obs.battery_soc:.1f}%")
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
# Reset
curl -X POST https://codequasar-orbital-anomaly-openenv.hf.space/reset \
  -H "Content-Type: application/json" -d '{"task_id": "hard"}'

# Step
curl -X POST https://codequasar-orbital-anomaly-openenv.hf.space/step \
  -H "Content-Type: application/json" -d '{"action_type": "rotate_to_sun"}'

# State
curl https://codequasar-orbital-anomaly-openenv.hf.space/state
```

### Heuristic Baseline
```bash
python inference.py                          # multi-agent heuristic
HF_TOKEN=hf_... python inference.py         # LLM via HF inference API
```

---

## 💻 Local Setup

```bash
git clone https://github.com/umed-indulkar/orbital-anomaly-openenv.git
cd orbital-anomaly-openenv
uv sync
openenv validate       # must pass before any submission
uv run server          # → http://localhost:8000
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
```

All 8 tests must pass:
- `test_tasks_by_name` — all 3 tasks accessible by name with valid rewards
- `test_physics_coupling` — actions affect correct subsystems
- `test_hard_fault_cascade` — eclipse drains battery without intervention
- `test_science_window_reward` — observation window adds science bonus
- `test_cycling` — sequential resets cycle through all 3 tasks
- `test_no_boundary_rewards` — all rewards strictly in (0, 1)
- `test_partial_observability` — sensor dropout on hard task
- `test_counter_independence` — explicit task_id always wins

---

## 📁 Repository Structure

```
orbital-anomaly-openenv/
├── server/
│   ├── app.py                                       # FastAPI + OpenEnv server
│   └── orbital_anomaly_openenv_environment.py       # V2.2 spacecraft simulator
├── models.py                                        # Pydantic typed models (V1 + V2)
├── client.py                                        # Typed Python client
├── inference.py                                     # Multi-agent baseline + LLM policy
├── test_reward.py                                   # Complete V2 test suite
├── Orbital_Anomaly_openenv.ipynb                 # GRPO training notebook (v4.1)
├── blog.md                                          # Plain-English explainer + images
├── images/                                          # All blog/training visualizations
│   ├── task_snapshot.png
│   ├── baseline_distributions.png
│   ├── training_analysis.png
│   ├── action_policy_heatmap.png
│   ├── fault_belief_evolution.png
│   ├── telemetry_timeline_36step.png
│   └── final_dashboard.png
├── openenv.yaml                                     # OpenEnv manifest
├── pyproject.toml                                   # Dependencies (uv)
├── Dockerfile                                       # HF Spaces container
└── README.md
```

---

## 🔌 API Reference

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/reset` | POST | `{"task_id": "easy"}` | Start new episode |
| `/step` | POST | `{"action_type": "rotate_to_sun"}` | Execute action |
| `/state` | GET | — | Current environment state |
| `/schema` | GET | — | Action/observation JSON schemas |
| `/docs` | GET | — | Interactive Swagger UI |
| `/openapi.json` | GET | — | Full OpenAPI spec |

---

## ✅ OpenEnv Compliance

| Requirement | Status |
|-------------|--------|
| `reset()` / `step()` / `state` interface | ✅ |
| Typed Pydantic action + observation models | ✅ |
| 3+ grader tasks (easy/medium/hard) | ✅ |
| All rewards strictly in (0, 1) | ✅ epsilon-bounded |
| Task cycling counter | ✅ class-level |
| Explicit `task_id` independent of counter | ✅ |
| `SUPPORTS_CONCURRENT_SESSIONS = True` | ✅ |
| LiteLLM proxy handshake | ✅ |
| `[START]`/`[STEP]`/`[END]` log format | ✅ |
| FastAPI + Docker + HF Spaces | ✅ |
| `openenv validate` passes | ✅ |

---

## 🧠 Why This Is Hard for LLMs

1. **Causal fault inference**: 13 faults are never directly observable. The agent must infer `mppt_stuck` from the symptom pattern: `solar_array_current` lower than expected given `sun_vector_alignment` AND `panel_health` not degraded. This requires causal world modeling.

2. **Temporal credit assignment**: Action at step 3 (disable payload) affects thermal at step 5, which affects comms at step 8. Short-sighted policies fail consistently.

3. **Partial observability**: 6 sensor fields drop out deterministically. Agent must reason about what it cannot see.

4. **Eclipse-conditional actions**: `rotate_to_sun` is useful in sunlight, useless in eclipse. Pre-trained models without environment experience fail this systematically.

5. **Multi-objective tradeoff**: Science bonus (+0.12) vs thermal safety — no simple threshold resolves this without knowing current temperature trajectory and fault state.

6. **Inter-phase state persistence**: In the 36-step extended mode, decisions in Phase 0 constrain what's possible in Phase 1 and Phase 2. Greedy per-phase optimization fails.

---

## 📝 Blog Post

A complete beginner-friendly walkthrough of how this was built, what it does, and what the AI actually learned is in **[blog.md](./blog.md)**.

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🤗 Live Space | https://codequasar-orbital-anomaly-openenv.hf.space |
| 📖 Swagger Docs | https://codequasar-orbital-anomaly-openenv.hf.space/docs |
| 📓 Training Colab | [Orbital_Anomaly_openenv.ipynb](https://colab.research.google.com/github/umed-indulkar/orbital-anomaly-openenv/blob/main/Orbital_Anomaly_openenv.ipynb) |
| 💾 GitHub | https://github.com/umed-indulkar/orbital-anomaly-openenv |
| 📝 Blog | [blog.md](./blog.md) |

---

## 📚 Citation

```bibtex
@misc{orbital-anomaly-openenv-2026,
  title   = {Orbital Anomaly OpenEnv: A Spacecraft Digital-Twin Benchmark
             for LLM World Modeling and Long-Horizon Planning},
  author  = {Indulkar, Umed},
  year    = {2026},
  url     = {https://huggingface.co/spaces/codequasar/orbital-anomaly-openenv}
}
```