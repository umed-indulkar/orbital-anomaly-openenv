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

A cloud-deployable **satellite anomaly response simulation benchmark** built with **OpenEnv, FastAPI, and Hugging Face Spaces**.

This environment simulates realistic spacecraft subsystem failures across:

- ⚡ Power / battery
- ☀️ Solar charging efficiency
- 🌡️ Thermal control
- 📡 Communications
- 🎯 Payload operations
- 🚨 Safe mode escalation

Agents must diagnose telemetry and perform **multi-step recovery strategies** to stabilize the mission.

---

# 🚀 Live Demo

## 🌐 Hugging Face Space
https://codequasar-orbital-anomaly-openenv.hf.space

## 📘 Interactive API Docs
https://codequasar-orbital-anomaly-openenv.hf.space/docs

---

# 🎯 Benchmark Goal

The objective is to test whether an agent can:

1. Interpret degraded satellite telemetry  
2. Identify the dominant subsystem anomaly  
3. Select the correct recovery action  
4. Avoid cascading failures  
5. Maximize long-term mission reward  

This makes it a **decision-making benchmark for autonomous agents**, not just a simulator.

---

# 🛰️ Example Initial Observation

After `POST /reset`:

```json
{
  "battery_level": 42,
  "solar_efficiency": 0.25,
  "thermal_temp": 48,
  "comms_signal": 0.95,
  "payload_on": true,
  "safe_mode": false,
  "task_id": "easy",
  "mission_status": "warning"
}