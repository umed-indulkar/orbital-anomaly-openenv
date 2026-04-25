# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations
import traceback
from typing import Optional
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

from models import OrbitalAnomalyOpenenvAction, OrbitalAnomalyOpenenvObservation
from server.orbital_anomaly_openenv_environment import OrbitalAnomalyOpenenvEnvironment
from inference import (
    compute_fault_beliefs, dominant_subsystem,
    top_faults_str, mission_commander_decide,
)

app = create_app(
    OrbitalAnomalyOpenenvEnvironment,
    OrbitalAnomalyOpenenvAction,
    OrbitalAnomalyOpenenvObservation,
    env_name="orbital_anomaly_openenv",
    max_concurrent_envs=8,
)


def _world_model(obs, env_obj=None):
    meta    = obs.metadata or {}
    beliefs = meta.get("fault_beliefs") or compute_fault_beliefs(obs)
    dom     = meta.get("dominant_subsystem") or dominant_subsystem(beliefs)
    return {
        "dominant_subsystem": dom,
        "top_faults":         top_faults_str(beliefs, 3),
        "fault_beliefs":      beliefs,
        "phase":              meta.get("phase", 0),
        "phase_step":         meta.get("phase_step", 0),
    }


@app.post("/reset", include_in_schema=False)
async def reset_with_task(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id = body.get("task_id") if isinstance(body, dict) else None
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env
    obs = env.reset(task_id=task_id)
    return JSONResponse({"observation": obs.model_dump(),
                         "reward": obs.reward, "done": obs.done,
                         "world_model": _world_model(obs)})


@app.post("/step", include_in_schema=False)
async def step_with_rationale(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    action_type = body.get("action_type", "noop") if isinstance(body, dict) else "noop"
    env: OrbitalAnomalyOpenenvEnvironment = request.app.state.env
    obs = env.step(OrbitalAnomalyOpenenvAction(action_type=action_type))
    action_rec, rationale, recs = mission_commander_decide(obs)
    return JSONResponse({
        "observation": obs.model_dump(),
        "reward": obs.reward, "done": obs.done,
        "agent_decision": {
            "recommended_action": action_rec, "rationale": rationale,
            "specialists": {
                n: {"action": r[0], "confidence": round(r[1], 3), "reason": r[2]}
                for n, r in recs.items()
            },
        },
        "world_model": _world_model(obs),
    })


@app.post("/run_episode", include_in_schema=False)
async def run_full_episode(request: Request) -> JSONResponse:
    """Run a complete autonomous episode; return all steps at once."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id = (body.get("task_id", "easy") if isinstance(body, dict) else "easy")

    try:
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
            action, rationale, recs = mission_commander_decide(obs)

            obs = env.step(OrbitalAnomalyOpenenvAction(action_type=action))
            reward = float(obs.reward or 0.001)
            total_reward += reward

            meta2    = obs.metadata or {}
            beliefs2 = meta2.get("fault_beliefs") or compute_fault_beliefs(obs)

            steps.append({
                "step":      step_num + 1,
                "action":    action,
                "rationale": rationale,
                "reward":    round(reward, 4),
                "done":      bool(obs.done),
                "telemetry": {
                    "battery_soc":      round(float(obs.battery_soc or 0), 1),
                    "solar_efficiency": round(float((obs.solar_efficiency or 0) * 100), 1),
                    "thermal_temp":     round(float(obs.thermal_temp or 0), 1),
                    "comms_signal":     round(float((obs.comms_signal or 0) * 100), 1),
                    "sunlit":           bool(getattr(obs, "sunlit", True)),
                    "payload_on":       bool(obs.payload_on),
                    "safe_mode":        bool(obs.safe_mode),
                    "mission_status":   str(obs.mission_status or "stable"),
                    "gs_visible":       bool(getattr(obs, "ground_station_visible", True)),
                },
                "world_model": {
                    "dominant_subsystem": dom,
                    "top_faults":         top_faults_str(beliefs, 3),
                    "fault_beliefs":      {k: round(float(v), 3) for k, v in beliefs2.items()},
                    "phase":              int(meta2.get("phase", 0)),
                },
                "specialists": {
                    n: {"action": r[0], "confidence": round(float(r[1]), 3), "reason": str(r[2])}
                    for n, r in recs.items()
                },
            })
            if obs.done:
                break

        avg = round(total_reward / max(len(steps), 1), 4)
        return JSONResponse({
            "task_id":      task_id,
            "total_steps":  len(steps),
            "avg_reward":   avg,
            "final_status": str(obs.mission_status or "stable"),
            "steps":        steps,
        })

    except Exception:
        tb = traceback.format_exc()
        print("[run_episode ERROR]", tb, flush=True)
        return JSONResponse({"error": "Episode failed", "detail": tb}, status_code=500)


# ── Dashboard HTML ─────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orbital Anomaly — Mission Control</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<style>
:root{
  --void:#020508;--deep:#050d14;--panel:rgba(6,16,28,0.9);
  --border:rgba(0,200,255,0.12);--border-hot:rgba(0,200,255,0.45);
  --cyan:#00d4ff;--cyan-dim:rgba(0,212,255,0.08);
  --green:#00ff88;--amber:#ffb800;--red:#ff3d3d;--purple:#a855f7;
  --text:#b8ccd8;--muted:#2d4a5a;
  --mono:'Space Mono',monospace;--display:'Orbitron',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{background:var(--void);color:var(--text);font-family:var(--mono);font-size:12px}

/* Stars */
.stars{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.star{position:absolute;border-radius:50%;background:#fff;animation:twinkle var(--d,3s) var(--delay,0s) ease-in-out infinite}
@keyframes twinkle{0%,100%{opacity:var(--min,.1)}50%{opacity:var(--max,.8)}}

/* Layout */
.shell{position:relative;z-index:1;height:100vh;display:flex;flex-direction:column}
header{
  flex-shrink:0;padding:14px 28px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:rgba(2,5,8,0.8);backdrop-filter:blur(16px);
}
.logo-text{font-family:var(--display);font-size:14px;font-weight:900;letter-spacing:3px;color:var(--cyan);text-shadow:0 0 20px rgba(0,212,255,0.35)}
.logo-sub{font-size:9px;color:var(--muted);letter-spacing:2px;margin-top:2px}
.header-right{display:flex;gap:10px;align-items:center}
.hlink{font-size:9px;color:var(--muted);text-decoration:none;letter-spacing:1px;padding:3px 10px;border:1px solid var(--border);border-radius:2px;transition:.2s}
.hlink:hover{color:var(--cyan);border-color:var(--border-hot)}

main{flex:1;display:grid;grid-template-columns:1fr 500px;min-height:0}

/* ─── LEFT ─── */
.left{
  position:relative;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  border-right:1px solid var(--border);
  overflow:hidden;padding:20px 32px;gap:20px;
}

/* Scanline overlay */
.left::before{
  content:'';position:absolute;inset:0;pointer-events:none;z-index:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.04) 2px,rgba(0,0,0,0.04) 4px);
}

/* Earth */
.earth-wrap{position:absolute;bottom:-220px;left:50%;transform:translateX(-50%);z-index:1}
.earth{
  width:460px;height:460px;border-radius:50%;
  background:radial-gradient(circle at 32% 30%,#1e5a9a,#0d3060 35%,#071a38 65%,#030d1c);
  box-shadow:0 0 80px rgba(20,100,220,0.25),0 0 160px rgba(10,60,160,0.1),inset 0 0 80px rgba(0,0,0,0.6);
}
.earth::after{
  content:'';position:absolute;inset:-6px;border-radius:50%;
  background:radial-gradient(circle,transparent 79%,rgba(60,160,255,0.07) 84%,rgba(30,100,200,0.03) 90%,transparent 94%);
  animation:atmo 5s ease-in-out infinite;
}
@keyframes atmo{0%,100%{opacity:.5}50%{opacity:1}}

/* Orbit ring */
.orbit-ring{
  position:absolute;z-index:2;
  width:520px;height:180px;
  bottom:60px;left:50%;transform:translateX(-50%);
  border:1px dashed rgba(0,200,255,0.18);border-radius:50%;
  pointer-events:none;
}

/* Satellite on orbit */
.sat-orbit{
  position:absolute;bottom:60px;left:50%;z-index:3;
  width:0;height:0;
  animation:sat-around 14s linear infinite;
}
.sat-orbit.fast{animation-duration:7s}
@keyframes sat-around{
  from{transform:rotate(-30deg) translateX(260px) rotate(30deg)}
  to  {transform:rotate(330deg) translateX(260px) rotate(-330deg)}
}
.sat{
  position:absolute;left:-30px;top:-16px;
  width:60px;height:32px;display:flex;align-items:center;
  filter:drop-shadow(0 0 6px rgba(0,212,255,0.6));
}
.sat-body{
  width:22px;height:16px;flex-shrink:0;
  background:linear-gradient(135deg,#1e3a5a,#0d2040);
  border:1px solid rgba(0,212,255,0.5);border-radius:2px;position:relative;
  box-shadow:0 0 8px rgba(0,212,255,0.2);
}
.sat-body::after{
  content:'';position:absolute;inset:3px;border-radius:50%;
  background:var(--cyan);opacity:.5;box-shadow:0 0 5px var(--cyan);
}
.wing{
  width:18px;height:9px;flex-shrink:0;
  background:linear-gradient(90deg,#1a3a72,#2a5090);
  border:1px solid rgba(80,140,255,0.35);border-radius:1px;
  transition:transform .6s,opacity .6s;
}
.wing.mis{transform:rotate(38deg);opacity:.4}
.sig-ring{
  position:absolute;top:-18px;left:8px;
  width:10px;height:10px;border-radius:50%;
  border:1px solid rgba(0,212,255,0.5);
  animation:ring-expand 1.8s ease-out infinite;
}
.sig-ring::after{
  content:'';position:absolute;inset:3px;border-radius:50%;
  border:1px solid rgba(0,212,255,0.7);
  animation:ring-expand 1.8s ease-out .5s infinite;
}
@keyframes ring-expand{0%{opacity:.9;transform:scale(1)}100%{opacity:0;transform:scale(2.8)}}
.sat-glow{
  position:absolute;inset:-20px;border-radius:50%;
  background:radial-gradient(circle,rgba(0,212,255,0.18),transparent 70%);
  animation:glow 3s ease-in-out infinite;
}
.sat-glow.warn{background:radial-gradient(circle,rgba(255,184,0,0.22),transparent 70%)}
.sat-glow.crit{background:radial-gradient(circle,rgba(255,61,61,0.28),transparent 70%);animation-duration:1.2s}
@keyframes glow{0%,100%{opacity:.5;transform:scale(1)}50%{opacity:1;transform:scale(1.12)}}

/* Scene text */
.scene-header{position:relative;z-index:4;text-align:center}
.scene-title{font-family:var(--display);font-size:20px;font-weight:900;letter-spacing:5px;color:var(--cyan);text-shadow:0 0 25px rgba(0,212,255,0.3);margin-bottom:4px}
.scene-sub{font-size:9px;color:var(--muted);letter-spacing:2px}

/* Task cards */
.task-grid{display:flex;gap:10px;position:relative;z-index:4;width:100%;max-width:580px}
.task-card{
  flex:1;padding:14px 10px;
  background:rgba(6,16,28,0.7);border:1px solid var(--border);
  border-radius:6px;cursor:pointer;text-align:center;
  transition:all .22s;backdrop-filter:blur(8px);
}
.task-card:hover{border-color:var(--border-hot);background:var(--cyan-dim)}
.task-card.sel-easy  {border-color:var(--cyan);background:rgba(0,212,255,0.07);box-shadow:0 0 18px rgba(0,212,255,0.12)}
.task-card.sel-medium{border-color:var(--amber);background:rgba(255,184,0,0.07);box-shadow:0 0 18px rgba(255,184,0,0.12)}
.task-card.sel-hard  {border-color:var(--red);background:rgba(255,61,61,0.07);box-shadow:0 0 18px rgba(255,61,61,0.12)}
.t-icon{font-size:20px;margin-bottom:5px}
.t-name{font-family:var(--display);font-size:9px;font-weight:700;letter-spacing:2px;margin-bottom:3px}
.t-desc{font-size:9px;color:var(--muted);line-height:1.5;margin-bottom:6px}
.t-tags{display:flex;gap:4px;justify-content:center;flex-wrap:wrap}
.ttag{font-size:9px;padding:1px 6px;border-radius:2px;border:1px solid}
.ttag-c{color:var(--cyan);border-color:rgba(0,212,255,.3);background:rgba(0,212,255,.07)}
.ttag-a{color:var(--amber);border-color:rgba(255,184,0,.3);background:rgba(255,184,0,.07)}
.ttag-r{color:var(--red);border-color:rgba(255,61,61,.3);background:rgba(255,61,61,.07)}

/* Stat strip */
.stat-strip{
  display:flex;gap:16px;position:relative;z-index:4;
  padding:10px 16px;background:rgba(0,0,0,0.3);
  border:1px solid var(--border);border-radius:4px;width:100%;max-width:580px;
}
.stat-item{flex:1;text-align:center}
.stat-lbl{font-size:8px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:3px}
.stat-val{font-family:var(--display);font-size:13px;font-weight:700;color:var(--cyan)}

/* Launch */
.launch-btn{
  font-family:var(--display);font-size:11px;font-weight:700;letter-spacing:3px;
  padding:13px 44px;
  background:linear-gradient(135deg,rgba(0,212,255,.12),rgba(0,80,140,.18));
  border:1px solid var(--cyan);color:var(--cyan);cursor:pointer;
  border-radius:3px;transition:all .2s;text-transform:uppercase;
  box-shadow:0 0 18px rgba(0,212,255,.1);position:relative;z-index:4;
}
.launch-btn:hover{background:rgba(0,212,255,.2);box-shadow:0 0 28px rgba(0,212,255,.25);transform:translateY(-1px)}
.launch-btn:active{transform:none}
.launch-btn:disabled{opacity:.35;cursor:not-allowed;transform:none}
.launch-btn.running{border-color:var(--amber);color:var(--amber);background:rgba(255,184,0,.08);animation:bpulse .9s ease-in-out infinite}
@keyframes bpulse{0%,100%{box-shadow:0 0 16px rgba(255,184,0,.1)}50%{box-shadow:0 0 32px rgba(255,184,0,.35)}}

/* ─── RIGHT ─── */
.right{display:flex;flex-direction:column;min-height:0;background:rgba(2,6,10,0.55);backdrop-filter:blur(14px)}

/* Score bar */
.score-bar{
  flex-shrink:0;display:grid;grid-template-columns:1fr 1fr 1fr;
  border-bottom:1px solid var(--border);
}
.score-cell{padding:14px 16px;text-align:center;border-right:1px solid var(--border)}
.score-cell:last-child{border-right:none}
.sc-lbl{font-size:8px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:5px}
.sc-val{font-family:var(--display);font-size:20px;font-weight:700;color:var(--cyan)}
.sc-val.green{color:var(--green);text-shadow:0 0 12px rgba(0,255,136,.3)}
.sc-val.amber{color:var(--amber);text-shadow:0 0 12px rgba(255,184,0,.3)}
.sc-val.red  {color:var(--red);text-shadow:0 0 12px rgba(255,61,61,.3)}

/* Chart */
.chart-wrap{flex-shrink:0;padding:12px 20px 8px;border-bottom:1px solid var(--border)}
.chart-lbl{font-size:8px;color:var(--muted);letter-spacing:2px;margin-bottom:6px}
svg.rchart{width:100%;height:56px;display:block}

/* World model ticker */
.wm-ticker{
  flex-shrink:0;padding:6px 20px;
  border-bottom:1px solid var(--border);
  background:rgba(0,0,0,0.2);
  font-size:9px;color:var(--muted);letter-spacing:1px;
  white-space:nowrap;overflow:hidden;
}
.wm-ticker span{color:var(--cyan)}

/* Feed */
.feed-wrap{flex:1;overflow-y:auto;padding:10px 16px;display:flex;flex-direction:column;gap:5px}
.feed-wrap::-webkit-scrollbar{width:2px}
.feed-wrap::-webkit-scrollbar-thumb{background:var(--muted);border-radius:1px}

.idle{
  flex:1;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:16px;padding:32px;text-align:center;
}
.idle-icon{font-size:40px;opacity:.2;animation:float 4s ease-in-out infinite}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
.idle-txt{font-size:10px;color:var(--muted);line-height:1.9;letter-spacing:1px}
.idle-hint{
  font-size:9px;color:var(--muted);padding:8px 16px;
  border:1px solid var(--border);border-radius:2px;letter-spacing:1px;
}

/* Step card */
.step-card{
  padding:9px 12px;
  background:rgba(255,255,255,0.018);
  border:1px solid rgba(255,255,255,0.05);
  border-left:2px solid transparent;
  border-radius:3px;
  opacity:0;transform:translateY(6px);
  animation:sin .28s ease forwards;
  display:grid;grid-template-columns:26px 1fr auto;gap:8px;align-items:start;
}
@keyframes sin{to{opacity:1;transform:none}}
.step-card.lv-high{border-left-color:var(--green)}
.step-card.lv-mid {border-left-color:var(--cyan)}
.step-card.lv-low {border-left-color:var(--amber)}
.step-card.lv-crit{border-left-color:var(--red)}
.sn{font-family:var(--display);font-size:9px;color:var(--muted);padding-top:2px}
.sb{}
.sa{font-family:var(--display);font-size:10px;font-weight:700;color:var(--cyan);margin-bottom:2px;letter-spacing:.5px}
.sr-text{font-size:10px;color:#4a6a7a;line-height:1.5;margin-bottom:3px}
.st{font-size:9px;color:rgba(180,210,225,.35);margin-bottom:4px}

/* Fault mini-bars */
.fbars{display:flex;flex-direction:column;gap:2px;margin-top:3px}
.fbar-row{display:grid;grid-template-columns:120px 1fr 26px;align-items:center;gap:5px}
.fn{font-size:9px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fb{height:3px;background:rgba(255,255,255,.05);border-radius:1px;overflow:hidden}
.ff{height:100%;border-radius:1px;transition:width .5s}
.fp{font-size:8px;text-align:right}

/* Specialist mini grid */
.specs{display:flex;gap:5px;margin-top:4px;flex-wrap:wrap}
.spec-pill{font-size:9px;padding:1px 7px;border-radius:2px;border:1px solid;letter-spacing:.3px}

.srw{font-family:var(--display);font-size:12px;font-weight:700;padding-top:2px}

/* Verdict */
.verdict{flex-shrink:0;border-top:1px solid var(--border);padding:12px 20px;display:none}
.verdict.show{display:block}
.vd-title{font-family:var(--display);font-size:9px;letter-spacing:2px;color:var(--muted);margin-bottom:8px}
.vd-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.vd-cell{padding:7px 10px;background:rgba(0,0,0,.25);border:1px solid var(--border);border-radius:2px}
.vd-lbl{font-size:8px;color:var(--muted);letter-spacing:1.5px;margin-bottom:2px}
.vd-val{font-family:var(--display);font-size:12px;font-weight:700}
</style>
</head>
<body>
<div class="stars" id="stars"></div>
<div class="shell">

<header>
  <div>
    <div class="logo-text">ORBITAL ANOMALY</div>
    <div class="logo-sub">MISSION CONTROL · OPENENV v2.2</div>
  </div>
  <div class="header-right">
    <a href="/docs" class="hlink">API DOCS</a>
    <a href="/state" class="hlink">RAW STATE</a>
    <a href="https://github.com/umed-indulkar/orbital-anomaly-openenv" class="hlink" target="_blank">GITHUB</a>
  </div>
</header>

<main>
<!-- LEFT -->
<div class="left">
  <!-- Earth + Satellite -->
  <div class="earth-wrap"><div class="earth"></div></div>
  <div class="orbit-ring"></div>
  <div class="sat-orbit" id="sat-orbit">
    <div class="sat">
      <div class="sat-glow" id="sat-glow"></div>
      <div class="wing" id="wing-l"></div>
      <div class="sat-body"></div>
      <div class="wing" id="wing-r"></div>
      <div class="sig-ring" id="sig-ring"></div>
    </div>
  </div>

  <!-- Title -->
  <div class="scene-header" style="margin-bottom:4px">
    <div class="scene-title" id="scene-title">SELECT MISSION</div>
    <div class="scene-sub" id="scene-sub">CHOOSE A CRISIS SCENARIO · AI AGENT WILL RESPOND</div>
  </div>

  <!-- Live stats strip (shown during/after run) -->
  <div class="stat-strip" id="stat-strip" style="display:none">
    <div class="stat-item"><div class="stat-lbl">Battery</div><div class="stat-val" id="ls-bat">—</div></div>
    <div class="stat-item"><div class="stat-lbl">Thermal</div><div class="stat-val" id="ls-temp">—</div></div>
    <div class="stat-item"><div class="stat-lbl">Comms</div><div class="stat-val" id="ls-comms">—</div></div>
    <div class="stat-item"><div class="stat-lbl">Phase</div><div class="stat-val" id="ls-phase">—</div></div>
    <div class="stat-item"><div class="stat-lbl">Dominant fault</div><div class="stat-val" id="ls-dom" style="font-size:9px">—</div></div>
  </div>

  <!-- Task cards -->
  <div class="task-grid">
    <div class="task-card sel-easy" id="tc-easy" onclick="selectTask('easy')">
      <div class="t-icon">🟢</div>
      <div class="t-name">EASY</div>
      <div class="t-desc">EPS Crisis<br>Single fault · Sunlit</div>
      <div class="t-tags">
        <span class="ttag ttag-c">SOC 38%</span>
        <span class="ttag ttag-c">1 FAULT</span>
        <span class="ttag ttag-c">MPPT</span>
      </div>
    </div>
    <div class="task-card" id="tc-medium" onclick="selectTask('medium')">
      <div class="t-icon">🟡</div>
      <div class="t-name">MEDIUM</div>
      <div class="t-desc">Thermal + Science<br>Dual fault · Tradeoff</div>
      <div class="t-tags">
        <span class="ttag ttag-a">68°C</span>
        <span class="ttag ttag-a">2 FAULTS</span>
        <span class="ttag ttag-a">+0.12 BONUS</span>
      </div>
    </div>
    <div class="task-card" id="tc-hard" onclick="selectTask('hard')">
      <div class="t-icon">🔴</div>
      <div class="t-name">HARD</div>
      <div class="t-desc">Cascade Failure<br>Eclipse · 7 faults</div>
      <div class="t-tags">
        <span class="ttag ttag-r">SOC 22%</span>
        <span class="ttag ttag-r">ECLIPSE</span>
        <span class="ttag ttag-r">7 FAULTS</span>
      </div>
    </div>
  </div>

  <button class="launch-btn" id="launch-btn" onclick="launchMission()">▶ LAUNCH MISSION</button>
</div>

<!-- RIGHT -->
<div class="right">
  <!-- Score bar -->
  <div class="score-bar">
    <div class="score-cell"><div class="sc-lbl">AVG REWARD</div><div class="sc-val" id="sc-avg">—</div></div>
    <div class="score-cell"><div class="sc-lbl">STEPS</div><div class="sc-val" id="sc-steps">—</div></div>
    <div class="score-cell"><div class="sc-lbl">MISSION STATUS</div><div class="sc-val" id="sc-status">—</div></div>
  </div>

  <!-- Chart -->
  <div class="chart-wrap">
    <div class="chart-lbl">REWARD TIMELINE — AGENT PERFORMANCE</div>
    <svg class="rchart" viewBox="0 0 460 56" preserveAspectRatio="none">
      <defs>
        <linearGradient id="rgrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="rgba(0,212,255,.25)"/>
          <stop offset="100%" stop-color="rgba(0,212,255,0)"/>
        </linearGradient>
      </defs>
      <line x1="0" y1="40" x2="460" y2="40" stroke="rgba(255,184,0,.2)" stroke-width="1" stroke-dasharray="3"/>
      <text x="4" y="38" font-size="8" fill="rgba(255,184,0,.4)" font-family="Space Mono,monospace">0.45</text>
      <path id="chart-area" fill="url(#rgrad)" d=""/>
      <polyline id="chart-line" fill="none" stroke="#00d4ff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" points=""/>
    </svg>
  </div>

  <!-- World model ticker -->
  <div class="wm-ticker" id="wm-ticker">WORLD MODEL — <span>awaiting mission launch</span></div>

  <!-- Feed / Idle -->
  <div class="feed-wrap" id="feed-wrap">
    <div class="idle" id="idle-el">
      <div class="idle-icon">🛰️</div>
      <div class="idle-txt">
        SELECT A MISSION SCENARIO ABOVE<br>
        THE MULTI-AGENT AI WILL EXECUTE 12 DECISION STEPS<br><br>
        COMMANDER → EPS · THERMAL · COMMS SPECIALISTS<br>
        WORLD MODEL: 13-FAULT BELIEF STATE PER STEP<br>
        LONG-HORIZON: 36-STEP EXTENDED MISSION MODE
      </div>
      <div class="idle-hint">THEME 3: WORLD MODELING · THEME 2: LONG-HORIZON PLANNING</div>
    </div>
  </div>

  <!-- Verdict -->
  <div class="verdict" id="verdict">
    <div class="vd-title">MISSION DEBRIEF</div>
    <div class="vd-grid">
      <div class="vd-cell"><div class="vd-lbl">TASK</div><div class="vd-val" id="vd-task" style="color:var(--cyan)">—</div></div>
      <div class="vd-cell"><div class="vd-lbl">AVG REWARD</div><div class="vd-val" id="vd-avg">—</div></div>
      <div class="vd-cell"><div class="vd-lbl">STEPS TAKEN</div><div class="vd-val" style="color:var(--cyan)" id="vd-steps">—</div></div>
      <div class="vd-cell"><div class="vd-lbl">FINAL STATUS</div><div class="vd-val" id="vd-fin">—</div></div>
    </div>
  </div>
</div>
</main>
</div>

<script>
// ── Stars ──────────────────────────────────────────────────────────────────────
(function(){
  const c=document.getElementById('stars');
  for(let i=0;i<120;i++){
    const s=document.createElement('div');
    s.className='star';
    const sz=Math.random()<.85?1:Math.random()<.7?1.5:2;
    s.style.cssText=`width:${sz}px;height:${sz}px;left:${Math.random()*100}%;top:${Math.random()*100}%;`+
      `--d:${2+Math.random()*4}s;--delay:${Math.random()*4}s;`+
      `--min:${.05+Math.random()*.1};--max:${.4+Math.random()*.6}`;
    c.appendChild(s);
  }
})();

// ── State ──────────────────────────────────────────────────────────────────────
let selTask='easy', running=false;
const rewards=[];

const ICONS={rotate_to_sun:'☀️',disable_payload:'📦',reboot_comms:'📡',enter_safe_mode:'🛡️',switch_power_bus:'🔋',noop:'⏸️'};
const SPEC_COLORS={EPS_Specialist:'rgba(0,212,255,.25)',Thermal_Specialist:'rgba(168,85,247,.25)',Comms_Specialist:'rgba(0,255,136,.2)'};
const SPEC_BORDER={EPS_Specialist:'rgba(0,212,255,.5)',Thermal_Specialist:'rgba(168,85,247,.5)',Comms_Specialist:'rgba(0,255,136,.4)'};

function rColor(r){return r>=.7?'var(--green)':r>=.5?'var(--cyan)':r>=.35?'var(--amber)':'var(--red)'}
function rClass(r){return r>=.7?'lv-high':r>=.5?'lv-mid':r>=.35?'lv-low':'lv-crit'}
function fColor(p){return p>.7?'var(--red)':p>.4?'var(--amber)':'var(--cyan)'}
function statusColor(st){const s=(st||'').toLowerCase();return s==='critical'?'var(--red)':s==='warning'?'var(--amber)':'var(--green)'}

function selectTask(task){
  selTask=task;
  ['easy','medium','hard'].forEach(t=>{
    const el=document.getElementById('tc-'+t);
    el.className='task-card'+(t===task?' sel-'+t:'');
  });
  // Satellite visual changes
  const glow=document.getElementById('sat-glow');
  const orbit=document.getElementById('sat-orbit');
  const sig=document.getElementById('sig-ring');
  const wl=document.getElementById('wing-l');
  const wr=document.getElementById('wing-r');
  const sub=document.getElementById('scene-sub');
  glow.className='sat-glow';
  orbit.className='sat-orbit';
  wl.classList.remove('mis'); wr.classList.remove('mis');
  sig.style.display='';
  if(task==='hard'){
    glow.classList.add('crit');
    orbit.classList.add('fast');
    wl.classList.add('mis'); wr.classList.add('mis');
    sig.style.display='none';
    sub.textContent='ECLIPSE · 7 FAULTS · SOC 22% · GS BLACKOUT · RADIATION ZONE';
  } else if(task==='medium'){
    glow.classList.add('warn');
    sub.textContent='THERMAL CRISIS · 2 FAULTS · PAYLOAD TEMP 68°C · SCIENCE WINDOW ACTIVE';
  } else {
    sub.textContent='EPS CRISIS · 1 FAULT (MPPT STUCK) · SOC 38% · SUNLIT';
  }
}

function updateChart(){
  if(!rewards.length)return;
  const W=460,H=56,pad=4,n=rewards.length;
  const pts=rewards.map((r,i)=>{
    const x=n===1?W/2:(i/(n-1))*W;
    const y=H-pad-(Math.max(0,Math.min(1,r))*(H-pad*2));
    return [x.toFixed(1),y.toFixed(1)];
  });
  const line=pts.map(p=>p.join(',')).join(' ');
  const area=`M${pts[0][0]},${H} `+pts.map(p=>`L${p[0]},${p[1]}`).join(' ')+` L${pts[pts.length-1][0]},${H} Z`;
  document.getElementById('chart-line').setAttribute('points',line);
  document.getElementById('chart-area').setAttribute('d',area);
}

function buildFaultBars(beliefs){
  if(!beliefs)return'';
  const top=Object.entries(beliefs).sort((a,b)=>b[1]-a[1]).slice(0,5);
  return`<div class="fbars">`+top.map(([n,p])=>
    `<div class="fbar-row">
      <span class="fn">${n}</span>
      <div class="fb"><div class="ff" style="width:${(p*100).toFixed(0)}%;background:${fColor(p)}"></div></div>
      <span class="fp" style="color:${fColor(p)}">${Math.round(p*100)}%</span>
    </div>`).join('')+`</div>`;
}

function buildSpecs(specs){
  if(!specs)return'';
  return`<div class="specs">`+Object.entries(specs).map(([name,s])=>{
    const short=name.replace('_Specialist','');
    const bg=SPEC_COLORS[name]||'rgba(255,255,255,.05)';
    const bc=SPEC_BORDER[name]||'rgba(255,255,255,.15)';
    return`<span class="spec-pill" style="background:${bg};border-color:${bc};color:var(--text)">${short}: ${s.action} (${Math.round(s.confidence*100)}%)</span>`;
  }).join('')+`</div>`;
}

async function launchMission(){
  if(running)return;
  running=true;
  rewards.length=0;

  const btn=document.getElementById('launch-btn');
  const feed=document.getElementById('feed-wrap');
  btn.textContent='⟳ EXECUTING...';
  btn.className='launch-btn running';
  btn.disabled=true;

  // Reset right panel
  feed.innerHTML='';
  document.getElementById('idle-el')&&(document.getElementById('idle-el').style.display='none');
  document.getElementById('verdict').classList.remove('show');
  document.getElementById('sc-avg').textContent='—';
  document.getElementById('sc-steps').textContent='—';
  document.getElementById('sc-status').textContent='—';
  document.getElementById('sc-avg').className='sc-val';
  document.getElementById('sc-status').className='sc-val';
  document.getElementById('chart-line').setAttribute('points','');
  document.getElementById('chart-area').setAttribute('d','');
  document.getElementById('wm-ticker').innerHTML='WORLD MODEL — <span style="color:var(--amber)">MISSION IN PROGRESS...</span>';
  document.getElementById('stat-strip').style.display='flex';

  try{
    const res=await fetch('/run_episode',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({task_id:selTask})
    });
    if(!res.ok){
      const txt=await res.text();
      throw new Error(`Server error ${res.status}: ${txt.slice(0,120)}`);
    }
    const data=await res.json();
    if(data.error){throw new Error(data.error+': '+data.detail)}

    for(let i=0;i<data.steps.length;i++){
      await new Promise(r=>setTimeout(r,110));
      const s=data.steps[i];
      const t=s.telemetry||{};
      const wm=s.world_model||{};
      const icon=ICONS[s.action]||'⚡';
      const cls=rClass(s.reward);
      const col=rColor(s.reward);

      // Update left live stats
      document.getElementById('ls-bat').textContent=t.battery_soc+'%';
      document.getElementById('ls-bat').style.color=t.battery_soc<20?'var(--red)':t.battery_soc<40?'var(--amber)':'var(--cyan)';
      document.getElementById('ls-temp').textContent=t.thermal_temp+'°C';
      document.getElementById('ls-temp').style.color=t.thermal_temp>85?'var(--red)':t.thermal_temp>70?'var(--amber)':'var(--cyan)';
      document.getElementById('ls-comms').textContent=t.comms_signal+'%';
      document.getElementById('ls-comms').style.color=t.comms_signal<30?'var(--red)':t.comms_signal<60?'var(--amber)':'var(--cyan)';
      document.getElementById('ls-phase').textContent='P'+(wm.phase+1)+'/3';
      document.getElementById('ls-dom').textContent=(wm.dominant_subsystem||'—').toUpperCase();

      // Update satellite glow based on status
      const glow=document.getElementById('sat-glow');
      if(t.mission_status==='critical'){glow.className='sat-glow crit'}
      else if(t.mission_status==='warning'){glow.className='sat-glow warn'}
      else{glow.className='sat-glow'}

      // World model ticker
      document.getElementById('wm-ticker').innerHTML=
        `WORLD MODEL — Dominant: <span>${(wm.dominant_subsystem||'?').toUpperCase()}</span> &nbsp;|&nbsp; `+
        `Top faults: <span>${wm.top_faults||'—'}</span> &nbsp;|&nbsp; `+
        `Phase: <span>${wm.phase+1}/3</span>`+
        (!t.sunlit?' &nbsp;|&nbsp; <span style="color:var(--amber)">🌑 ECLIPSE</span>':'')+
        (!t.gs_visible?' &nbsp;|&nbsp; <span style="color:var(--red)">GS BLACKOUT</span>':'');

      // Add step card
      const el=document.createElement('div');
      el.className=`step-card ${cls}`;
      el.innerHTML=
        `<div class="sn">${String(s.step).padStart(2,'0')}</div>`+
        `<div class="sb">`+
          `<div class="sa">${icon} ${s.action.toUpperCase()}</div>`+
          `<div class="sr-text">${s.rationale}</div>`+
          `<div class="st">BAT ${t.battery_soc}% · TEMP ${t.thermal_temp}°C · COMMS ${t.comms_signal}%`+
          `${!t.sunlit?' · <span style="color:var(--amber)">ECLIPSE</span>':''}`+
          `${t.safe_mode?' · <span style="color:var(--red)">SAFE MODE</span>':''}`+
          ` · <span style="color:${statusColor(t.mission_status)}">${(t.mission_status||'stable').toUpperCase()}</span></div>`+
          buildFaultBars(wm.fault_beliefs)+
          buildSpecs(s.specialists)+
        `</div>`+
        `<div class="srw" style="color:${col}">${s.reward.toFixed(3)}</div>`;
      feed.appendChild(el);
      feed.scrollTop=feed.scrollHeight;

      // Update score bar
      rewards.push(s.reward);
      const avg=rewards.reduce((a,b)=>a+b,0)/rewards.length;
      document.getElementById('sc-avg').textContent=avg.toFixed(3);
      document.getElementById('sc-avg').className='sc-val '+(avg>=.7?'green':avg>=.5?'':'amber');
      document.getElementById('sc-steps').textContent=`${i+1} / ${data.steps.length}`;
      document.getElementById('sc-status').textContent=(t.mission_status||'stable').toUpperCase();
      document.getElementById('sc-status').className='sc-val '+(t.mission_status==='critical'?'red':t.mission_status==='warning'?'amber':'green');
      updateChart();
    }

    // Verdict
    const finalAvg=rewards.reduce((a,b)=>a+b,0)/rewards.length;
    document.getElementById('vd-task').textContent=data.task_id.toUpperCase();
    document.getElementById('vd-avg').textContent=finalAvg.toFixed(4);
    document.getElementById('vd-avg').style.color=rColor(finalAvg);
    document.getElementById('vd-steps').textContent=data.total_steps;
    document.getElementById('vd-fin').textContent=(data.final_status||'stable').toUpperCase();
    document.getElementById('vd-fin').style.color=statusColor(data.final_status);
    document.getElementById('verdict').classList.add('show');
    document.getElementById('wm-ticker').innerHTML='WORLD MODEL — <span style="color:var(--green)">MISSION COMPLETE · AVG REWARD: '+finalAvg.toFixed(4)+'</span>';

  }catch(e){
    const el=document.createElement('div');
    el.style.cssText='color:var(--red);padding:16px;font-size:11px;background:rgba(255,61,61,.08);border:1px solid rgba(255,61,61,.2);border-radius:3px;margin:8px';
    el.textContent='Error: '+e.message;
    feed.appendChild(el);
  }

  running=false;
  btn.textContent='▶ RUN AGAIN';
  btn.className='launch-btn';
  btn.disabled=false;
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