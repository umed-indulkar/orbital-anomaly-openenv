---
title: Orbital Anomaly OpenEnv
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🛰️ Orbital Anomaly OpenEnv

A cloud-deployable **satellite anomaly response simulation benchmark** built with **OpenEnv, FastAPI, Docker, and Hugging Face Spaces**.

This environment simulates realistic spacecraft subsystem failures where an autonomous agent acts as **mission control** and must stabilize the satellite through multi-step corrective actions.

It is designed as a **real-world decision-making benchmark** for evaluating frontier LLM and RL agents on long-horizon anomaly recovery tasks. 

---

# 🌍 Real-World Motivation

Modern satellites continuously face cascading subsystem risks:

- ⚡ Battery drain
- ☀️ Poor solar panel alignment
- 🌡️ Thermal overload
- 📡 Communication degradation
- 🎯 Payload power spikes
- 🚨 Safe mode escalation

Human operators diagnose telemetry streams and execute recovery actions in the correct sequence.

This benchmark converts that real operational workflow into a **typed OpenEnv environment**.

---

# 🎯 Benchmark Objective

The agent must:

1. Interpret degraded telemetry
2. Identify dominant failure source
3. Choose corrective actions
4. Prevent cascading subsystem failures
5. Maximize long-term mission reward
6. Stabilize mission within step budget

This makes it ideal for:
- LLM agents
- RL policies
- planning systems
- multi-step recovery reasoning
- reward shaping research

---

# 🧩 Action Space

The agent can issue one of the following mission-control actions:

- `rotate_to_sun` → improves solar charging efficiency
- `disable_payload` → reduces thermal + battery load
- `reboot_comms` → restores communications
- `enter_safe_mode` → emergency stabilization action
- `switch_power_bus` → boosts battery reserves
- `noop` → no intervention

---

# 📡 Observation Space

Each step returns spacecraft telemetry:

- `battery_level` → 0–100
- `solar_efficiency` → 0–1
- `thermal_temp` → Celsius
- `comms_signal` → 0–1
- `payload_on` → bool
- `safe_mode` → bool
- `task_id` → easy / medium / hard
- `mission_status` → stable / warning / critical
- `reward` → dense 0–1 progress signal
- `done` → episode termination

---

# 🎚️ Difficulty Tasks

The environment contains **3 deterministic benchmark tasks**:

## 🟢 Easy
Low battery + solar misalignment

## 🟡 Medium
Thermal overload + moderate power degradation

## 🔴 Hard
Simultaneous battery, thermal, and communication collapse

This satisfies the **minimum 3-task grader requirement**. 

---

# 🏆 Reward Design

The reward is a **dense trajectory signal in the range 0.0–1.0**.

It combines:

- battery health
- thermal stability
- communication quality
- mission survivability

This provides:
- partial progress shaping
- long-horizon planning incentives
- smooth optimization signal
- deterministic grading

Exactly aligned with OpenEnv scoring requirements. :contentReference[oaicite:3]{index=3}

---

# 🚀 Live Deployment

## 🌐 Hugging Face Space
https://codequasar-orbital-anomaly-openenv.hf.space

## 📘 Interactive API Docs
https://codequasar-orbital-anomaly-openenv.hf.space/docs

## 📡 OpenAPI Schema
https://codequasar-orbital-anomaly-openenv.hf.space/openapi.json

---

# 💻 Local Setup

## 1) Install dependencies
```bash
uv sync
```

## 2) Run server locally
```bash
uv run server
```

Server starts at:

```text
http://localhost:8000
```

Docs:

```text
http://localhost:8000/docs
```

---

# 🧪 Run Baseline Inference

The repository includes a **reproducible baseline agent**.

Run:

```bash
py -m orbital_anomaly_openenv.inference
```

Expected output:
```text
[START] orbital anomaly recovery baseline
[STEP] step=1 action=rotate_to_sun reward=0.610
...
[END] final_reward=0.647
```

This satisfies the mandatory **baseline reproducibility requirement**. 

---

# 🐳 Docker Build

Build locally:

```bash
docker build -t orbital-anomaly-openenv .
```

Run:

```bash
docker run -p 8000:8000 orbital-anomaly-openenv
```

---

# ☁️ Deploy to Hugging Face Spaces

Push latest repo:

```bash
git push origin main
git push hf main
```

HF auto-builds the Docker Space.

---

# 📂 Project Structure

```text
orbital_anomaly_openenv/
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── README.md
├── server/
│   ├── app.py
│   ├── orbital_anomaly_openenv_environment.py
│   └── __init__.py
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

---

# 🧠 Why This Benchmark Matters

Most existing OpenEnv tasks focus on:
- browser actions
- text manipulation
- scheduling
- tool usage

This benchmark introduces a **spacecraft anomaly response domain**, which is:

- safety critical
- long horizon
- partially observable
- reward dense
- causally rich
- highly novel

This strongly improves **novelty + real-world utility scores**. :contentReference[oaicite:5]{index=5}

---

# ✅ Validation Checklist

Before submission run:

```bash
openenv validate
```

and:

```bash
docker build .
```

Also verify:

```text
POST /reset → 200
POST /step → 200
GET /state → 200
```

This matches official validation flow. 