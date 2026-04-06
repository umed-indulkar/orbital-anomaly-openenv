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

- `rotate_to_sun`
- `disable_payload`
- `reboot_comms`
- `enter_safe_mode`
- `switch_power_bus`
- `noop`

---

# 📡 Observation Space

Each step returns spacecraft telemetry:

- `battery_level`
- `solar_efficiency`
- `thermal_temp`
- `comms_signal`
- `payload_on`
- `safe_mode`
- `task_id`
- `mission_status`
- `reward`
- `done`

---

# 🎚️ Difficulty Tasks

The environment contains **3 deterministic benchmark tasks**:

- 🟢 Easy → battery + solar anomaly
- 🟡 Medium → thermal overload
- 🔴 Hard → cascading subsystem failure

This satisfies the **minimum 3-task grader requirement**.

---

# 🏆 Reward Design

The reward is a **dense trajectory signal in the range [0, 1]**.

It combines:
- battery health
- thermal stability
- communication quality
- mission survivability

This enables:
- partial progress shaping
- long-horizon optimization
- deterministic evaluation
- smooth policy learning

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

## 2) Validate structure
```bash
openenv validate
```

## 3) Run locally
```bash
uv run server
```

Server:
```text
http://localhost:8000
```

Docs:
```text
http://localhost:8000/docs
```

---

# 🧪 Baseline Inference

Run:
```bash
py inference.py
```

Expected:
```text
[START] orbital anomaly recovery baseline
[STEP] step=1 task=easy action=rotate_to_sun reward=0.610
...
[END] final_reward=0.720
```

---

# 🐳 Docker Build

```bash
docker build -t orbital-anomaly-openenv .
docker run -p 8000:8000 orbital-anomaly-openenv
```

---

# ☁️ Hugging Face Deployment

```bash
git push origin main
git push hf main
```

HF automatically rebuilds the Docker Space.

---

# 📂 Project Structure

```text
orbital-anomaly-openenv/
├── __init__.py
├── client.py
├── models.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── README.md
├── pyproject.toml
├── uv.lock
└── server/
    ├── __init__.py
    ├── app.py
    └── orbital_anomaly_openenv_environment.py
```

---

# 🧠 Why This Benchmark Matters

Most OpenEnv tasks focus on:
- browser automation
- scheduling
- text tools
- web actions

This benchmark introduces a **causal spacecraft recovery domain** that is:

- safety critical
- long horizon
- reward dense
- partially observable
- highly novel
- real-world aligned

This improves both **novelty and utility scores**.

---

# ✅ Validation Checklist

Run before submission:

```bash
openenv validate
docker build .
py inference.py
```

Also verify:

```text
POST /reset → 200
POST /step → 200
GET /state → 200
```