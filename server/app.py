# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations
from typing import Optional
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

from models import OrbitalAnomalyOpenenvAction, OrbitalAnomalyOpenenvObservation
from server.orbital_anomaly_openenv_environment import OrbitalAnomalyOpenenvEnvironment
from inference import compute_fault_beliefs, dominant_subsystem, top_faults_str, mission_commander_decide

app = create_app(
    OrbitalAnomalyOpenenvEnvironment,
    OrbitalAnomalyOpenenvAction,
    OrbitalAnomalyOpenenvObservation,
    env_name="orbital_anomaly_openenv",
    max_concurrent_envs=8,
)


@app.post("/reset", include_in_schema=False)
async def reset_with_task(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id: Optional[str] = body.get("task_id") if isinstance(body, dict) else None
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env
    obs = env.reset(task_id=task_id)
    meta    = obs.metadata or {}
    beliefs = meta.get("fault_beliefs") or compute_fault_beliefs(obs)
    dom     = meta.get("dominant_subsystem") or dominant_subsystem(beliefs)
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "world_model": {
            "dominant_subsystem": dom,
            "top_faults": top_faults_str(beliefs, 3),
            "fault_beliefs": beliefs,
            "phase": meta.get("phase", 0),
            "phase_step": meta.get("phase_step", 0),
        },
    })


@app.post("/step", include_in_schema=False)
async def step_with_rationale(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    action_type: str = (body.get("action_type", "noop") if isinstance(body, dict) else "noop")
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env
    obs = env.step(OrbitalAnomalyOpenenvAction(action_type=action_type))
    meta    = obs.metadata or {}
    beliefs = meta.get("fault_beliefs") or compute_fault_beliefs(obs)
    dom     = meta.get("dominant_subsystem") or dominant_subsystem(beliefs)
    action_rec, rationale, specialist_recs = mission_commander_decide(obs)
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "agent_decision": {
            "recommended_action": action_rec,
            "rationale": rationale,
            "specialists": {
                name: {"action": r[0], "confidence": round(r[1], 3), "reason": r[2]}
                for name, r in specialist_recs.items()
            },
        },
        "world_model": {
            "dominant_subsystem": dom,
            "top_faults": top_faults_str(beliefs, 3),
            "fault_beliefs": beliefs,
            "phase": meta.get("phase", 0),
            "phase_step": meta.get("phase_step", 0),
        },
    })


@app.post("/run_episode", include_in_schema=False)
async def run_full_episode(request: Request) -> JSONResponse:
    """Run a complete episode autonomously and return all steps at once."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id: Optional[str] = body.get("task_id", "easy") if isinstance(body, dict) else "easy"
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env
    obs = env.reset(task_id=task_id)

    steps = []
    total_reward = 0.0

    for step_num in range(12):
        if env._check_done():
            break
        meta    = obs.metadata or {}
        beliefs = meta.get("fault_beliefs") or compute_fault_beliefs(obs)
        dom     = meta.get("dominant_subsystem") or dominant_subsystem(beliefs)
        action, rationale, specialist_recs = mission_commander_decide(obs)
        obs = env.step(OrbitalAnomalyOpenenvAction(action_type=action))
        reward = float(obs.reward or 0.001)
        total_reward += reward
        meta2   = obs.metadata or {}
        beliefs2 = meta2.get("fault_beliefs") or compute_fault_beliefs(obs)

        steps.append({
            "step":      step_num + 1,
            "action":    action,
            "rationale": rationale,
            "reward":    round(reward, 4),
            "done":      obs.done,
            "telemetry": {
                "battery_soc":      round(obs.battery_soc or 0, 1),
                "solar_efficiency": round((obs.solar_efficiency or 0) * 100, 1),
                "thermal_temp":     round(obs.thermal_temp or 0, 1),
                "comms_signal":     round((obs.comms_signal or 0) * 100, 1),
                "sunlit":           getattr(obs, "sunlit", True),
                "payload_on":       obs.payload_on,
                "mission_status":   obs.mission_status or "stable",
            },
            "world_model": {
                "dominant_subsystem": dom,
                "top_faults": top_faults_str(beliefs, 3),
                "fault_beliefs": beliefs2,
                "phase": meta2.get("phase", 0),
            },
            "specialists": {
                name: {"action": r[0], "confidence": round(r[1], 3), "reason": r[2]}
                for name, r in specialist_recs.items()
            },
        })
        if obs.done:
            break

    return JSONResponse(content={
        "task_id":      task_id,
        "total_steps":  len(steps),
        "avg_reward":   round(total_reward / max(len(steps), 1), 4),
        "final_status": obs.mission_status or "stable",
        "steps":        steps,
    })


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orbital Anomaly — Mission Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<style>
:root {
  --void: #020508;
  --deep: #050d14;
  --panel: rgba(8,20,35,0.85);
  --border: rgba(0,212,255,0.12);
  --border-hot: rgba(0,212,255,0.4);
  --cyan: #00d4ff;
  --cyan-dim: rgba(0,212,255,0.15);
  --green: #00ff88;
  --green-dim: rgba(0,255,136,0.12);
  --amber: #ffb800;
  --red: #ff3d3d;
  --text: #c8dce8;
  --muted: #3a5568;
  --font-mono: 'Space Mono', monospace;
  --font-display: 'Orbitron', monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--void);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 12px;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── STAR FIELD ── */
.stars {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.6) 0%, transparent 100%),
    radial-gradient(1px 1px at 80% 10%, rgba(255,255,255,0.4) 0%, transparent 100%),
    radial-gradient(1px 1px at 60% 70%, rgba(255,255,255,0.5) 0%, transparent 100%),
    radial-gradient(1px 1px at 40% 50%, rgba(255,255,255,0.3) 0%, transparent 100%),
    radial-gradient(1px 1px at 90% 60%, rgba(255,255,255,0.6) 0%, transparent 100%),
    radial-gradient(1px 1px at 10% 80%, rgba(255,255,255,0.4) 0%, transparent 100%),
    radial-gradient(1px 1px at 50% 20%, rgba(255,255,255,0.5) 0%, transparent 100%),
    radial-gradient(1px 1px at 70% 90%, rgba(255,255,255,0.3) 0%, transparent 100%),
    radial-gradient(1px 1px at 30% 60%, rgba(255,255,255,0.4) 0%, transparent 100%),
    radial-gradient(1px 1px at 85% 40%, rgba(255,255,255,0.5) 0%, transparent 100%);
}
.stars::after {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 50% 0%, rgba(0,40,80,0.4) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 20% 100%, rgba(0,20,50,0.3) 0%, transparent 60%);
}

/* ── LAYOUT ── */
.shell {
  position: relative; z-index: 1;
  min-height: 100vh;
  display: flex; flex-direction: column;
}

/* ── HEADER ── */
header {
  padding: 20px 32px 16px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  background: rgba(2,5,8,0.7);
  backdrop-filter: blur(12px);
}
.logo-text {
  font-family: var(--font-display);
  font-size: 15px; font-weight: 900; letter-spacing: 3px;
  color: var(--cyan);
  text-shadow: 0 0 20px rgba(0,212,255,0.4);
}
.logo-sub { font-size: 10px; color: var(--muted); letter-spacing: 2px; margin-top: 2px; }
.header-links { display: flex; gap: 16px; }
.header-link {
  font-size: 10px; color: var(--muted); text-decoration: none;
  letter-spacing: 1px; padding: 4px 10px;
  border: 1px solid var(--border); border-radius: 3px;
  transition: all .2s;
}
.header-link:hover { color: var(--cyan); border-color: var(--border-hot); }

/* ── MAIN CONTENT ── */
main {
  flex: 1; display: grid;
  grid-template-columns: 1fr 520px;
  grid-template-rows: 1fr;
  gap: 0;
  max-height: calc(100vh - 65px);
}

/* ── LEFT: SATELLITE SCENE ── */
.sat-scene {
  position: relative; display: flex;
  flex-direction: column; align-items: center; justify-content: center;
  padding: 40px;
  border-right: 1px solid var(--border);
  overflow: hidden;
}

/* Earth glow */
.earth {
  position: absolute; bottom: -180px; left: 50%;
  transform: translateX(-50%);
  width: 420px; height: 420px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 35%,
    #1a4a7a 0%, #0d2d52 30%, #081a30 60%, #040d18 100%);
  box-shadow:
    0 0 60px rgba(20,100,200,0.3),
    0 0 120px rgba(10,60,140,0.15),
    inset 0 0 60px rgba(0,0,0,0.5);
}
.earth::after {
  content: '';
  position: absolute; inset: -2px;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 25%,
    rgba(100,180,255,0.1) 0%, transparent 50%);
}
.earth-atmo {
  position: absolute; bottom: -210px; left: 50%;
  transform: translateX(-50%);
  width: 480px; height: 480px;
  border-radius: 50%;
  background: radial-gradient(circle,
    transparent 78%, rgba(50,140,255,0.08) 83%, rgba(30,100,200,0.04) 88%, transparent 93%);
  animation: atmo-pulse 4s ease-in-out infinite;
}
@keyframes atmo-pulse { 0%,100%{opacity:.6} 50%{opacity:1} }

/* Orbit path */
.orbit-path {
  position: absolute;
  width: 500px; height: 160px;
  bottom: 30px; left: 50%;
  transform: translateX(-50%);
  border: 1px dashed rgba(0,212,255,0.15);
  border-radius: 50%;
  pointer-events: none;
}

/* Satellite */
.sat-wrapper {
  position: relative;
  animation: orbit 12s linear infinite;
  transform-origin: 0 80px;
  margin-bottom: 60px;
}
@keyframes orbit {
  0%   { transform: rotate(-40deg)  translateX(250px) rotate(40deg); }
  100% { transform: rotate(320deg) translateX(250px) rotate(-320deg); }
}
.sat-wrapper.crisis { animation-duration: 6s; }

.satellite {
  position: relative; width: 60px; height: 30px;
  display: flex; align-items: center; justify-content: center;
  filter: drop-shadow(0 0 8px rgba(0,212,255,0.5));
}
.sat-body {
  width: 24px; height: 18px;
  background: linear-gradient(135deg, #1e3a5a, #0d2040);
  border: 1px solid rgba(0,212,255,0.4);
  border-radius: 3px;
  position: relative;
  box-shadow: 0 0 10px rgba(0,212,255,0.2);
}
.sat-body::after {
  content: '';
  position: absolute; top: 50%; left: 50%;
  transform: translate(-50%,-50%);
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--cyan);
  opacity: .6;
  box-shadow: 0 0 6px var(--cyan);
}
.sat-panel {
  width: 20px; height: 10px;
  background: linear-gradient(90deg, #1a3a6a, #2a5090);
  border: 1px solid rgba(100,160,255,0.3);
  border-radius: 1px;
  position: absolute;
}
.sat-panel.left  { right: calc(100% + 4px); top: 4px; }
.sat-panel.right { left:  calc(100% + 4px); top: 4px; }
.sat-panel.misaligned { transform: rotate(35deg); opacity: .5; }

.sat-antenna {
  position: absolute; top: -10px; left: 50%;
  width: 1px; height: 10px;
  background: linear-gradient(to top, rgba(0,212,255,0.6), transparent);
  transform: translateX(-50%);
}
.sat-signal {
  position: absolute; top: -20px; left: 50%;
  transform: translateX(-50%);
  width: 12px; height: 12px;
  border: 1px solid rgba(0,212,255,0.4);
  border-radius: 50%;
  animation: signal-pulse 2s ease-out infinite;
}
.sat-signal::after {
  content: '';
  position: absolute; inset: 3px;
  border: 1px solid rgba(0,212,255,0.6);
  border-radius: 50%;
  animation: signal-pulse 2s ease-out infinite .4s;
}
@keyframes signal-pulse { 0%{opacity:.8;transform:scale(1)} 100%{opacity:0;transform:scale(2.5)} }

/* Status glow */
.sat-glow {
  position: absolute; inset: -15px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,212,255,0.15) 0%, transparent 70%);
  animation: glow-pulse 3s ease-in-out infinite;
}
.sat-glow.critical { background: radial-gradient(circle, rgba(255,61,61,0.25) 0%, transparent 70%); }
.sat-glow.warning  { background: radial-gradient(circle, rgba(255,184,0,0.2) 0%, transparent 70%); }
@keyframes glow-pulse { 0%,100%{opacity:.5;transform:scale(1)} 50%{opacity:1;transform:scale(1.1)} }

/* Scene info */
.scene-title {
  font-family: var(--font-display);
  font-size: 22px; font-weight: 900; letter-spacing: 4px;
  color: var(--cyan);
  text-shadow: 0 0 30px rgba(0,212,255,0.3);
  text-align: center; margin-bottom: 6px;
  position: relative; z-index: 2;
}
.scene-sub {
  font-size: 10px; color: var(--muted); letter-spacing: 2px;
  text-align: center; margin-bottom: 40px;
  position: relative; z-index: 2;
}

/* Task selector */
.task-grid {
  display: flex; gap: 12px; margin-bottom: 32px;
  position: relative; z-index: 2;
}
.task-card {
  flex: 1; padding: 14px 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px; cursor: pointer;
  transition: all .25s; text-align: center;
  backdrop-filter: blur(8px);
}
.task-card:hover { border-color: var(--border-hot); background: var(--cyan-dim); }
.task-card.selected { border-color: var(--cyan); background: var(--cyan-dim); box-shadow: 0 0 20px rgba(0,212,255,0.15); }
.task-card.selected.hard { border-color: var(--red); background: rgba(255,61,61,0.08); box-shadow: 0 0 20px rgba(255,61,61,0.15); }
.task-card.selected.medium { border-color: var(--amber); background: rgba(255,184,0,0.08); box-shadow: 0 0 20px rgba(255,184,0,0.15); }
.task-icon { font-size: 22px; margin-bottom: 6px; }
.task-name { font-family: var(--font-display); font-size: 9px; letter-spacing: 2px; font-weight: 700; }
.task-desc { font-size: 9px; color: var(--muted); margin-top: 4px; line-height: 1.4; }
.task-stats { display: flex; gap: 6px; justify-content: center; margin-top: 8px; flex-wrap: wrap; }
.task-tag { font-size: 9px; padding: 1px 6px; border-radius: 2px; }
.tag-easy   { background: rgba(0,255,136,0.12); color: var(--green); }
.tag-medium { background: rgba(255,184,0,0.12); color: var(--amber); }
.tag-hard   { background: rgba(255,61,61,0.12); color: var(--red); }

/* Launch button */
.launch-btn {
  font-family: var(--font-display);
  font-size: 12px; font-weight: 700; letter-spacing: 3px;
  padding: 14px 40px;
  background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,100,160,0.2));
  border: 1px solid var(--cyan);
  color: var(--cyan); cursor: pointer;
  border-radius: 4px;
  transition: all .2s;
  text-transform: uppercase;
  box-shadow: 0 0 20px rgba(0,212,255,0.1);
  position: relative; z-index: 2;
}
.launch-btn:hover { background: rgba(0,212,255,0.2); box-shadow: 0 0 30px rgba(0,212,255,0.25); transform: translateY(-1px); }
.launch-btn:active { transform: translateY(0); }
.launch-btn:disabled { opacity: .4; cursor: not-allowed; transform: none; }
.launch-btn.running {
  border-color: var(--amber); color: var(--amber);
  background: rgba(255,184,0,0.1);
  animation: btn-pulse .8s ease-in-out infinite;
}
@keyframes btn-pulse { 0%,100%{box-shadow:0 0 20px rgba(255,184,0,0.15)} 50%{box-shadow:0 0 35px rgba(255,184,0,0.35)} }

/* ── RIGHT: RESULTS PANEL ── */
.results-panel {
  display: flex; flex-direction: column;
  background: rgba(2,6,10,0.6);
  backdrop-filter: blur(12px);
  overflow: hidden;
}

/* Score header */
.score-bar {
  padding: 20px 24px 16px;
  border-bottom: 1px solid var(--border);
  display: grid; grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
}
.score-metric { text-align: center; }
.score-label { font-size: 9px; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }
.score-value { font-family: var(--font-display); font-size: 22px; font-weight: 700; }
.score-value.cyan  { color: var(--cyan); text-shadow: 0 0 15px rgba(0,212,255,0.4); }
.score-value.green { color: var(--green); text-shadow: 0 0 15px rgba(0,255,136,0.3); }
.score-value.amber { color: var(--amber); text-shadow: 0 0 15px rgba(255,184,0,0.3); }
.score-value.red   { color: var(--red); text-shadow: 0 0 15px rgba(255,61,61,0.3); }

/* Reward chart */
.chart-wrap { padding: 16px 24px 8px; border-bottom: 1px solid var(--border); }
.chart-label { font-size: 9px; color: var(--muted); letter-spacing: 2px; margin-bottom: 8px; }
svg.chart { width: 100%; height: 64px; }

/* Step feed */
.step-feed {
  flex: 1; overflow-y: auto;
  padding: 12px 24px;
  display: flex; flex-direction: column; gap: 6px;
}
.step-feed::-webkit-scrollbar { width: 3px; }
.step-feed::-webkit-scrollbar-track { background: transparent; }
.step-feed::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 2px; }

.step-item {
  display: grid; grid-template-columns: 28px 1fr auto;
  gap: 10px; align-items: start;
  padding: 10px 12px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.04);
  border-radius: 4px;
  opacity: 0; transform: translateY(8px);
  animation: step-in .3s ease forwards;
  border-left: 2px solid transparent;
}
@keyframes step-in { to { opacity: 1; transform: none; } }
.step-item.high  { border-left-color: var(--green); }
.step-item.mid   { border-left-color: var(--cyan); }
.step-item.low   { border-left-color: var(--amber); }
.step-item.crit  { border-left-color: var(--red); }

.step-num {
  font-family: var(--font-display); font-size: 10px; font-weight: 700;
  color: var(--muted); padding-top: 1px;
}
.step-body {}
.step-action {
  font-family: var(--font-display); font-size: 10px; font-weight: 700;
  color: var(--cyan); margin-bottom: 3px; letter-spacing: 1px;
}
.step-rationale { font-size: 10px; color: var(--muted); line-height: 1.5; }
.step-tele { font-size: 10px; color: rgba(200,220,232,0.5); margin-top: 3px; }
.step-reward {
  font-family: var(--font-display); font-size: 13px; font-weight: 700;
  padding-top: 1px;
}

/* Fault belief mini-bars */
.fault-mini { margin-top: 6px; display: flex; flex-direction: column; gap: 2px; }
.fault-mini-row { display: flex; align-items: center; gap: 6px; }
.fault-mini-name { font-size: 9px; color: var(--muted); width: 130px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fault-mini-bar { flex: 1; height: 3px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; }
.fault-mini-fill { height: 100%; border-radius: 2px; }
.fault-mini-pct { font-size: 9px; width: 28px; text-align: right; flex-shrink: 0; }

/* Final verdict */
.verdict {
  padding: 16px 24px;
  border-top: 1px solid var(--border);
  display: none;
}
.verdict.visible { display: block; }
.verdict-title {
  font-family: var(--font-display); font-size: 11px;
  letter-spacing: 2px; color: var(--muted); margin-bottom: 10px;
}
.verdict-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.verdict-label { font-size: 11px; color: var(--text); }
.verdict-val { font-family: var(--font-display); font-size: 11px; font-weight: 700; }

/* Idle state */
.idle-msg {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 12px; padding: 40px;
  text-align: center;
}
.idle-icon { font-size: 36px; opacity: .3; }
.idle-text { font-size: 11px; color: var(--muted); line-height: 1.7; letter-spacing: 1px; }

/* Action icon map */
.action-icons { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
.a-badge { font-size: 9px; padding: 2px 6px; border-radius: 2px; border: 1px solid; }
</style>
</head>
<body>

<div class="stars"></div>
<div class="shell">

<header>
  <div>
    <div class="logo-text">ORBITAL ANOMALY</div>
    <div class="logo-sub">MISSION CONTROL · OPENENV v2.2</div>
  </div>
  <div class="header-links">
    <a href="/docs" class="header-link">API DOCS</a>
    <a href="/state" class="header-link">RAW STATE</a>
    <a href="https://github.com/umed-indulkar/orbital-anomaly-openenv" class="header-link" target="_blank">GITHUB</a>
  </div>
</header>

<main>

  <!-- LEFT: Scene + Controls -->
  <div class="sat-scene">
    <div class="earth"></div>
    <div class="earth-atmo"></div>
    <div class="orbit-path"></div>

    <div class="sat-wrapper" id="sat-wrapper">
      <div class="satellite">
        <div class="sat-glow" id="sat-glow"></div>
        <div class="sat-panel left" id="panel-left"></div>
        <div class="sat-body"></div>
        <div class="sat-panel right" id="panel-right"></div>
        <div class="sat-antenna"></div>
        <div class="sat-signal" id="sat-signal"></div>
      </div>
    </div>

    <div class="scene-title">SELECT MISSION</div>
    <div class="scene-sub" id="scene-sub">CHOOSE A CRISIS SCENARIO · AGENT WILL RESPOND</div>

    <div class="task-grid">
      <div class="task-card selected easy" onclick="selectTask('easy', this)">
        <div class="task-icon">🟢</div>
        <div class="task-name">EASY</div>
        <div class="task-desc">EPS Crisis<br>Single fault · Sunlit</div>
        <div class="task-stats">
          <span class="task-tag tag-easy">SOC 38%</span>
          <span class="task-tag tag-easy">1 FAULT</span>
        </div>
      </div>
      <div class="task-card medium" onclick="selectTask('medium', this)">
        <div class="task-icon">🟡</div>
        <div class="task-name">MEDIUM</div>
        <div class="task-desc">Thermal + Science<br>Dual fault · Tradeoff</div>
        <div class="task-stats">
          <span class="task-tag tag-medium">TEMP 68°C</span>
          <span class="task-tag tag-medium">2 FAULTS</span>
        </div>
      </div>
      <div class="task-card hard" onclick="selectTask('hard', this)">
        <div class="task-icon">🔴</div>
        <div class="task-name">HARD</div>
        <div class="task-desc">Cascade Failure<br>Eclipse · 7 faults</div>
        <div class="task-stats">
          <span class="task-tag tag-hard">SOC 22%</span>
          <span class="task-tag tag-hard">7 FAULTS</span>
        </div>
      </div>
    </div>

    <button class="launch-btn" id="launch-btn" onclick="launchMission()">
      ▶ LAUNCH MISSION
    </button>
  </div>

  <!-- RIGHT: Results -->
  <div class="results-panel">

    <div class="score-bar" id="score-bar">
      <div class="score-metric">
        <div class="score-label">AVG REWARD</div>
        <div class="score-value cyan" id="sc-avg">—</div>
      </div>
      <div class="score-metric">
        <div class="score-label">STEPS</div>
        <div class="score-value cyan" id="sc-steps">—</div>
      </div>
      <div class="score-metric">
        <div class="score-label">FINAL STATUS</div>
        <div class="score-value cyan" id="sc-status">—</div>
      </div>
    </div>

    <div class="chart-wrap">
      <div class="chart-label">REWARD TIMELINE</div>
      <svg class="chart" viewBox="0 0 460 64" preserveAspectRatio="none">
        <defs>
          <linearGradient id="rg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="rgba(0,212,255,0.3)"/>
            <stop offset="100%" stop-color="rgba(0,212,255,0)"/>
          </linearGradient>
        </defs>
        <line x1="0" y1="46" x2="460" y2="46" stroke="rgba(255,184,0,0.2)" stroke-width="1" stroke-dasharray="4"/>
        <text x="4" y="44" font-size="8" fill="rgba(255,184,0,0.4)" font-family="Space Mono">0.45</text>
        <path id="chart-area" fill="url(#rg)" d=""/>
        <polyline id="chart-line" fill="none" stroke="#00d4ff" stroke-width="1.5"
          stroke-linecap="round" stroke-linejoin="round" points=""/>
      </svg>
    </div>

    <div id="feed-container" style="flex:1;display:flex;flex-direction:column;overflow:hidden">
      <div class="idle-msg" id="idle-msg">
        <div class="idle-icon">🛰️</div>
        <div class="idle-text">
          SELECT A MISSION SCENARIO<br>AND LAUNCH THE AGENT<br><br>
          THE AI WILL EXECUTE 12 DECISION STEPS<br>
          USING THE MULTI-AGENT COMMANDER<br>
          + EPS · THERMAL · COMMS SPECIALISTS
        </div>
      </div>
      <div class="step-feed" id="step-feed" style="display:none"></div>
    </div>

    <div class="verdict" id="verdict">
      <div class="verdict-title">MISSION DEBRIEF</div>
      <div class="verdict-row">
        <span class="verdict-label">Task</span>
        <span class="verdict-val cyan" id="vd-task">—</span>
      </div>
      <div class="verdict-row">
        <span class="verdict-label">Avg Reward</span>
        <span class="verdict-val" id="vd-avg">—</span>
      </div>
      <div class="verdict-row">
        <span class="verdict-label">Steps Taken</span>
        <span class="verdict-val cyan" id="vd-steps">—</span>
      </div>
      <div class="verdict-row">
        <span class="verdict-label">Final Status</span>
        <span class="verdict-val" id="vd-status">—</span>
      </div>
    </div>

  </div>
</main>
</div>

<script>
let selectedTask = 'easy';
let running = false;

const ACTION_ICONS = {
  rotate_to_sun:    {icon:'☀️', color:'#00d4ff'},
  disable_payload:  {icon:'📦', color:'#8b5cf6'},
  reboot_comms:     {icon:'📡', color:'#00d4ff'},
  enter_safe_mode:  {icon:'🛡️', color:'#ff3d3d'},
  switch_power_bus: {icon:'🔋', color:'#00ff88'},
  noop:             {icon:'⏸️', color:'#3a5568'},
};

function selectTask(task, el) {
  selectedTask = task;
  document.querySelectorAll('.task-card').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
  // Update satellite visual
  const glow = document.getElementById('sat-glow');
  const sub  = document.getElementById('scene-sub');
  glow.className = 'sat-glow';
  if (task === 'hard') {
    glow.classList.add('critical');
    sub.textContent = 'ECLIPSE · 7 FAULTS · BATTERY 22% · GS BLACKOUT';
    document.getElementById('sat-wrapper').classList.add('crisis');
    document.getElementById('sat-signal').style.display = 'none';
    document.getElementById('panel-left').classList.add('misaligned');
    document.getElementById('panel-right').classList.add('misaligned');
  } else if (task === 'medium') {
    glow.classList.add('warning');
    sub.textContent = 'THERMAL CRISIS · 2 FAULTS · SCIENCE WINDOW ACTIVE';
    document.getElementById('sat-wrapper').classList.remove('crisis');
    document.getElementById('sat-signal').style.display = '';
    document.getElementById('panel-left').classList.remove('misaligned');
    document.getElementById('panel-right').classList.remove('misaligned');
  } else {
    sub.textContent = 'EPS CRISIS · 1 FAULT · MPPT STUCK';
    document.getElementById('sat-wrapper').classList.remove('crisis');
    document.getElementById('sat-signal').style.display = '';
    document.getElementById('panel-left').classList.remove('misaligned');
    document.getElementById('panel-right').classList.remove('misaligned');
  }
}

function faultColor(p) {
  return p > 0.7 ? '#ff3d3d' : p > 0.4 ? '#ffb800' : '#00d4ff';
}

function rewardClass(r) {
  return r >= 0.7 ? 'high' : r >= 0.5 ? 'mid' : r >= 0.35 ? 'low' : 'crit';
}

function rewardColor(r) {
  return r >= 0.7 ? '#00ff88' : r >= 0.5 ? '#00d4ff' : r >= 0.35 ? '#ffb800' : '#ff3d3d';
}

function statusColor(st) {
  const s = (st||'').toLowerCase();
  return s === 'critical' ? 'red' : s === 'warning' ? 'amber' : 'green';
}

function buildFaultBars(beliefs) {
  if (!beliefs) return '';
  const sorted = Object.entries(beliefs).sort((a,b)=>b[1]-a[1]).slice(0, 5);
  return `<div class="fault-mini">` + sorted.map(([name, prob]) =>
    `<div class="fault-mini-row">
      <span class="fault-mini-name">${name}</span>
      <div class="fault-mini-bar"><div class="fault-mini-fill" style="width:${(prob*100).toFixed(0)}%;background:${faultColor(prob)}"></div></div>
      <span class="fault-mini-pct" style="color:${faultColor(prob)}">${Math.round(prob*100)}%</span>
    </div>`).join('') + `</div>`;
}

function updateChart(rewards) {
  if (!rewards.length) return;
  const W = 460, H = 64, pad = 4;
  const n = rewards.length;
  const pts = rewards.map((r, i) => {
    const x = n === 1 ? W/2 : (i / (n-1)) * W;
    const y = H - pad - (Math.max(0, Math.min(1, r)) * (H - pad*2));
    return [x.toFixed(1), y.toFixed(1)];
  });
  const line = pts.map(p => p.join(',')).join(' ');
  const area = `M${pts[0][0]},${H} ` + pts.map(p => `L${p[0]},${p[1]}`).join(' ') + ` L${pts[pts.length-1][0]},${H} Z`;
  document.getElementById('chart-line').setAttribute('points', line);
  document.getElementById('chart-area').setAttribute('d', area);
}

async function launchMission() {
  if (running) return;
  running = true;

  const btn  = document.getElementById('launch-btn');
  const feed = document.getElementById('step-feed');
  const idle = document.getElementById('idle-msg');

  btn.textContent = '⟳ EXECUTING...';
  btn.className = 'launch-btn running';
  btn.disabled = true;

  // Reset UI
  feed.innerHTML = '';
  feed.style.display = 'flex';
  idle.style.display = 'none';
  document.getElementById('verdict').classList.remove('visible');
  document.getElementById('sc-avg').textContent   = '—';
  document.getElementById('sc-steps').textContent = '—';
  document.getElementById('sc-status').textContent= '—';
  document.getElementById('sc-avg').className = 'score-value cyan';
  document.getElementById('sc-status').className = 'score-value cyan';
  document.getElementById('chart-line').setAttribute('points','');
  document.getElementById('chart-area').setAttribute('d','');

  try {
    const res  = await fetch('/run_episode', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({task_id: selectedTask})
    });
    const data = await res.json();
    const rewards = data.steps.map(s => s.reward);

    // Animate steps in with delay
    for (let i = 0; i < data.steps.length; i++) {
      await new Promise(r => setTimeout(r, 120));
      const s    = data.steps[i];
      const ai   = ACTION_ICONS[s.action] || {icon:'⚡',color:'#00d4ff'};
      const cls  = rewardClass(s.reward);
      const col  = rewardColor(s.reward);
      const t    = s.telemetry || {};
      const wm   = s.world_model || {};
      const sunlit = t.sunlit !== undefined ? t.sunlit : true;

      const el = document.createElement('div');
      el.className = `step-item ${cls}`;
      el.style.animationDelay = '0ms';
      el.innerHTML = `
        <div class="step-num">${String(s.step).padStart(2,'0')}</div>
        <div class="step-body">
          <div class="step-action">${ai.icon} ${s.action.toUpperCase()}</div>
          <div class="step-rationale">${s.rationale}</div>
          <div class="step-tele">
            BAT ${t.battery_soc}% · SOL ${t.solar_efficiency}% · TEMP ${t.thermal_temp}°C · COMMS ${t.comms_signal}%
            ${!sunlit ? ' · <span style="color:#ffb800">ECLIPSE</span>' : ''}
            · <span style="color:${t.mission_status==='critical'?'#ff3d3d':t.mission_status==='warning'?'#ffb800':'#00ff88'}">${(t.mission_status||'stable').toUpperCase()}</span>
          </div>
          ${buildFaultBars(wm.fault_beliefs)}
        </div>
        <div class="step-reward" style="color:${col}">${s.reward.toFixed(3)}</div>`;
      feed.appendChild(el);
      feed.scrollTop = feed.scrollHeight;

      // Update live scores
      const sofar = rewards.slice(0, i+1);
      const avg = sofar.reduce((a,b)=>a+b,0)/sofar.length;
      document.getElementById('sc-avg').textContent   = avg.toFixed(3);
      document.getElementById('sc-steps').textContent = `${i+1}/${data.steps.length}`;
      document.getElementById('sc-status').textContent = (t.mission_status||'stable').toUpperCase();
      document.getElementById('sc-avg').className    = `score-value ${rewardClass(avg)==='high'?'green':rewardClass(avg)==='mid'?'cyan':'amber'}`;
      document.getElementById('sc-status').className = `score-value ${statusColor(t.mission_status)}`;
      updateChart(sofar);
    }

    // Verdict
    const finalAvg = rewards.reduce((a,b)=>a+b,0)/rewards.length;
    document.getElementById('vd-task').textContent  = data.task_id.toUpperCase();
    document.getElementById('vd-avg').textContent   = finalAvg.toFixed(4);
    document.getElementById('vd-avg').style.color   = rewardColor(finalAvg);
    document.getElementById('vd-steps').textContent = data.total_steps;
    document.getElementById('vd-status').textContent= (data.final_status||'stable').toUpperCase();
    document.getElementById('vd-status').style.color= statusColor(data.final_status)==='green'?'#00ff88':statusColor(data.final_status)==='amber'?'#ffb800':'#ff3d3d';
    document.getElementById('verdict').classList.add('visible');
    updateChart(rewards);

  } catch(e) {
    const el = document.createElement('div');
    el.style.cssText='color:#ff3d3d;padding:16px;font-size:11px';
    el.textContent = 'Error: ' + e.message;
    feed.appendChild(el);
  }

  running = false;
  btn.textContent = '▶ RUN AGAIN';
  btn.className = 'launch-btn';
  btn.disabled = false;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return DASHBOARD_HTML


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()