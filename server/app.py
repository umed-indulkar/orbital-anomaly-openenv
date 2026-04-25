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


def _world_model(obs):
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


# ── /reset override — creates a fresh env per request (OpenEnv pattern) ───────
@app.post("/reset", include_in_schema=False)
async def reset_with_task(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_id = body.get("task_id") if isinstance(body, dict) else None
    # KEY FIX: instantiate directly — OpenEnv never stores env on app.state
    env = OrbitalAnomalyOpenenvEnvironment()
    obs = env.reset(task_id=task_id)
    return JSONResponse({"observation": obs.model_dump(),
                         "reward": obs.reward, "done": obs.done,
                         "world_model": _world_model(obs)})


# ── /step override — stateless, creates fresh env ─────────────────────────────
@app.post("/step", include_in_schema=False)
async def step_with_rationale(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    action_type = body.get("action_type", "noop") if isinstance(body, dict) else "noop"
    # KEY FIX: same pattern — fresh env
    env = OrbitalAnomalyOpenenvEnvironment()
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
    if task_id not in ("easy", "medium", "hard"):
        task_id = "easy"

    try:
        # KEY FIX: instantiate directly — never use request.app.state.env
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset(task_id=task_id)
        steps = []
        total_reward = 0.0

        # Safe heuristic: reads from obs object
        def safe_heuristic(o):
            try:
                bat    = float(o.battery_soc or 50)
                sol    = float(o.solar_efficiency or 0.5)
                temp   = float(o.thermal_temp or 40)
                ber    = float(o.bit_error_rate or 0.01)
                plr    = float(o.packet_loss_ratio or 0.05)
                comms  = max(0.0, min(1.0, 1.0 - ber * 5.0 - plr))
                sunlit = bool(getattr(o, "sunlit", True))
                payl   = bool(o.payload_on) if o.payload_on is not None else True
                if bat < 12:                          return "switch_power_bus", "[EPS|99%] CRITICAL: battery at floor"
                if bat < 30 and not sunlit:           return "switch_power_bus", "[EPS|94%] Eclipse + low battery — solar useless"
                if bat < 20 and sol < 0.35:           return "rotate_to_sun",   "[EPS|91%] CRITICAL: battery low, solar misaligned"
                if temp > 84:                         return "enter_safe_mode",  "[Thermal|98%] CRITICAL: thermal cascade imminent"
                if temp > 74 and payl:                return "disable_payload",  "[Thermal|91%] CRITICAL: payload heat critical"
                if bat < 38 and not sunlit:           return "switch_power_bus", "[EPS|82%] Eclipse: activate power reserve"
                if bat < 35:                          return "rotate_to_sun",    "[EPS|78%] Battery warning — realign solar"
                if comms < 0.22 and bat > 25:         return "reboot_comms",     "[Comms|97%] CRITICAL: link near loss"
                if sol < 0.42 and sunlit:             return "rotate_to_sun",    "[EPS|65%] Solar suboptimal — realign"
                if comms < 0.50:                      return "reboot_comms",     "[Comms|55%] Comms degraded"
                if temp > 62 and payl:                return "disable_payload",  "[Thermal|58%] Proactive thermal management"
                if sol < 0.70 and sunlit:             return "rotate_to_sun",    "[EPS|40%] Solar improvement possible"
                return "noop", "[Commander] All systems nominal"
            except Exception:
                return "noop", "[Commander] Heuristic error — holding"

        def safe_beliefs(o):
            try:
                b = max(0.0, min(1.0, float(o.battery_soc or 50) / 100.0))
                s = max(0.0, min(1.0, float(o.solar_efficiency or 0.5)))
                t = max(0.0, min(1.0, (float(o.thermal_temp or 40) - 20) / 80))
                c = max(0.0, min(1.0, float(o.comms_signal or 0.5)))
                w = max(0.0, min(1.0, float(o.wheel_saturation_level or 0.0)))
                r = max(0.0, min(1.0, 1.0 - float(o.radiator_efficiency or 1.0)))
                clip = lambda x: round(max(0.0, min(1.0, x)), 3)
                return {
                    "mppt_stuck":               clip((1-s)*0.9 + (1-b)*0.3),
                    "panel_deployment_jam":     clip((1-s)*0.8),
                    "bus_short_transient":      clip((1-b)*0.6 + t*0.2),
                    "battery_aging":            clip((1-b)*0.5),
                    "reaction_wheel_saturation":clip(w*0.9 + (1-s)*0.2),
                    "gyro_drift":               clip((1-s)*0.35 + w*0.1),
                    "star_tracker_dropout":     clip((1-s)*0.4),
                    "radiator_valve_stuck":     clip(r*0.7 + t*0.5),
                    "heat_pipe_failure":        clip(t*0.75 + (1-b)*0.1),
                    "heater_relay_latch":       clip(t*0.5 + (1-b)*0.2),
                    "transponder_overheating":  clip((1-c)*0.8 + t*0.3),
                    "amplifier_degradation":    clip((1-c)*0.65),
                    "antenna_gimbal_stall":     clip((1-c)*0.55 + (1-s)*0.15),
                }
            except Exception:
                return {}

        def safe_dom(beliefs):
            try:
                groups = {
                    "EPS":     ["mppt_stuck","panel_deployment_jam","bus_short_transient","battery_aging"],
                    "ADCS":    ["reaction_wheel_saturation","gyro_drift","star_tracker_dropout"],
                    "Thermal": ["radiator_valve_stuck","heat_pipe_failure","heater_relay_latch"],
                    "Comms":   ["transponder_overheating","amplifier_degradation","antenna_gimbal_stall"],
                }
                scores = {g: sum(beliefs.get(f,0) for f in fs)/len(fs) for g,fs in groups.items()}
                return max(scores, key=scores.get)
            except Exception:
                return "EPS"

        def sf(val, default=0.0):
            try: return round(float(val or default), 1)
            except: return round(default, 1)

        MAX_STEPS = 12

        for step_num in range(MAX_STEPS):
            if bool(obs.done):
                break

            action, rationale = safe_heuristic(obs)

            try:
                _, _, spec_recs = mission_commander_decide(obs)
                specialists = {
                    n: {"action": r[0], "confidence": round(r[1], 3), "reason": r[2]}
                    for n, r in spec_recs.items()
                }
            except Exception:
                specialists = {
                    "EPS_Specialist":     {"action": action, "confidence": 0.85, "reason": rationale},
                    "Thermal_Specialist": {"action": "noop", "confidence": 0.12, "reason": "Thermal nominal"},
                    "Comms_Specialist":   {"action": "noop", "confidence": 0.10, "reason": "Comms nominal"},
                }

            try:
                obs = env.step(OrbitalAnomalyOpenenvAction(action_type=action))
            except Exception as e:
                print(f"[step ERROR] step={step_num} {e}", flush=True)
                break

            reward = float(obs.reward or 0.001)
            total_reward += reward

            meta2    = obs.metadata or {}
            beliefs2 = meta2.get("fault_beliefs") or safe_beliefs(obs)
            dom      = meta2.get("dominant_subsystem") or safe_dom(beliefs2)

            steps.append({
                "step":      step_num + 1,
                "action":    action,
                "rationale": rationale,
                "reward":    round(reward, 4),
                "done":      bool(obs.done),
                "telemetry": {
                    "battery_soc":      sf(obs.battery_soc, 50),
                    "solar_efficiency": round(sf(obs.solar_efficiency, 0.5) * 100, 1),
                    "thermal_temp":     sf(obs.thermal_temp, 40),
                    "comms_signal":     round(sf(obs.comms_signal, 0.5) * 100, 1),
                    "sunlit":           bool(getattr(obs, "sunlit", True)),
                    "payload_on":       bool(obs.payload_on) if obs.payload_on is not None else True,
                    "safe_mode":        bool(obs.safe_mode) if obs.safe_mode is not None else False,
                    "mission_status":   str(obs.mission_status or "stable"),
                    "gs_visible":       bool(getattr(obs, "ground_station_visible", True)),
                    "attitude_error":   sf(getattr(obs, "attitude_error_deg", 0), 0),
                    "wheel_sat":        round(sf(getattr(obs, "wheel_saturation_level", 0), 0) * 100, 1),
                },
                "world_model": {
                    "dominant_subsystem": dom,
                    "top_faults": ", ".join(
                        f"{k}({round(v*100)}%)"
                        for k,v in sorted(beliefs2.items(), key=lambda x:-x[1])[:3]
                    ) if beliefs2 else "unknown",
                    "fault_beliefs": {k: round(float(v), 3) for k,v in beliefs2.items()},
                    "phase": int(meta2.get("phase", 0)),
                    "phase_step": int(meta2.get("phase_step", step_num)),
                },
                "specialists": specialists,
            })

            if bool(obs.done):
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
        print("[run_episode FATAL]", tb, flush=True)
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
.stars{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.star{position:absolute;border-radius:50%;background:#fff;animation:twinkle var(--d,3s) var(--delay,0s) ease-in-out infinite}
@keyframes twinkle{0%,100%{opacity:var(--min,.1)}50%{opacity:var(--max,.8)}}
.shell{position:relative;z-index:1;height:100vh;display:flex;flex-direction:column}
header{flex-shrink:0;padding:12px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:rgba(2,5,8,0.85);backdrop-filter:blur(16px);}
.logo-block{display:flex;align-items:center;gap:14px}
.logo-text{font-family:var(--display);font-size:13px;font-weight:900;letter-spacing:3px;color:var(--cyan);text-shadow:0 0 20px rgba(0,212,255,0.35)}
.logo-sub{font-size:8px;color:var(--muted);letter-spacing:2px;margin-top:2px}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse-dot 2s ease-in-out infinite;flex-shrink:0}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)}}
.header-center{display:flex;gap:20px;align-items:center}
.hstat{text-align:center}
.hstat-lbl{font-size:7px;color:var(--muted);letter-spacing:1.5px;margin-bottom:1px}
.hstat-val{font-family:var(--display);font-size:10px;color:var(--cyan)}
.header-right{display:flex;gap:8px;align-items:center}
.hlink{font-size:8px;color:var(--muted);text-decoration:none;letter-spacing:1px;padding:3px 10px;border:1px solid var(--border);border-radius:2px;transition:.2s}
.hlink:hover{color:var(--cyan);border-color:var(--border-hot)}
.hlink.hl-green{border-color:rgba(0,255,136,.25);color:var(--green)}
.hlink.hl-green:hover{border-color:var(--green);box-shadow:0 0 8px rgba(0,255,136,.2)}
main{flex:1;display:grid;grid-template-columns:1fr 520px;min-height:0}
.left{position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;border-right:1px solid var(--border);overflow:hidden;padding:16px 28px;gap:14px;}
.left::before{content:'';position:absolute;inset:0;pointer-events:none;z-index:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.04) 2px,rgba(0,0,0,0.04) 4px);}
.earth-wrap{position:absolute;bottom:-220px;left:50%;transform:translateX(-50%);z-index:1}
.earth{width:460px;height:460px;border-radius:50%;background:radial-gradient(circle at 32% 30%,#1e5a9a,#0d3060 35%,#071a38 65%,#030d1c);box-shadow:0 0 80px rgba(20,100,220,0.25),0 0 160px rgba(10,60,160,0.1),inset 0 0 80px rgba(0,0,0,0.6);}
.earth::after{content:'';position:absolute;inset:-6px;border-radius:50%;background:radial-gradient(circle,transparent 79%,rgba(60,160,255,0.07) 84%,rgba(30,100,200,0.03) 90%,transparent 94%);animation:atmo 5s ease-in-out infinite;}
@keyframes atmo{0%,100%{opacity:.5}50%{opacity:1}}
.orbit-ring{position:absolute;z-index:2;width:520px;height:180px;bottom:60px;left:50%;transform:translateX(-50%);border:1px dashed rgba(0,200,255,0.18);border-radius:50%;pointer-events:none;}
.sat-orbit{position:absolute;bottom:60px;left:50%;z-index:3;width:0;height:0;animation:sat-around 14s linear infinite;}
.sat-orbit.fast{animation-duration:7s}
@keyframes sat-around{from{transform:rotate(-30deg) translateX(260px) rotate(30deg)}to{transform:rotate(330deg) translateX(260px) rotate(-330deg)}}
.sat{position:absolute;left:-30px;top:-16px;width:60px;height:32px;display:flex;align-items:center;filter:drop-shadow(0 0 6px rgba(0,212,255,0.6));}
.sat-body{width:22px;height:16px;flex-shrink:0;background:linear-gradient(135deg,#1e3a5a,#0d2040);border:1px solid rgba(0,212,255,0.5);border-radius:2px;position:relative;box-shadow:0 0 8px rgba(0,212,255,0.2);}
.sat-body::after{content:'';position:absolute;inset:3px;border-radius:50%;background:var(--cyan);opacity:.5;box-shadow:0 0 5px var(--cyan);}
.wing{width:18px;height:9px;flex-shrink:0;background:linear-gradient(90deg,#1a3a72,#2a5090);border:1px solid rgba(80,140,255,0.35);border-radius:1px;transition:transform .6s,opacity .6s;}
.wing.mis{transform:rotate(38deg);opacity:.4}
.sig-ring{position:absolute;top:-18px;left:8px;width:10px;height:10px;border-radius:50%;border:1px solid rgba(0,212,255,0.5);animation:ring-expand 1.8s ease-out infinite;}
.sig-ring::after{content:'';position:absolute;inset:3px;border-radius:50%;border:1px solid rgba(0,212,255,0.7);animation:ring-expand 1.8s ease-out .5s infinite;}
@keyframes ring-expand{0%{opacity:.9;transform:scale(1)}100%{opacity:0;transform:scale(2.8)}}
.sat-glow{position:absolute;inset:-20px;border-radius:50%;background:radial-gradient(circle,rgba(0,212,255,0.18),transparent 70%);animation:glow 3s ease-in-out infinite;}
.sat-glow.warn{background:radial-gradient(circle,rgba(255,184,0,0.22),transparent 70%)}
.sat-glow.crit{background:radial-gradient(circle,rgba(255,61,61,0.28),transparent 70%);animation-duration:1.2s}
@keyframes glow{0%,100%{opacity:.5;transform:scale(1)}50%{opacity:1;transform:scale(1.12)}}
.scene-header{position:relative;z-index:4;text-align:center}
.scene-title{font-family:var(--display);font-size:18px;font-weight:900;letter-spacing:5px;color:var(--cyan);text-shadow:0 0 25px rgba(0,212,255,0.3);margin-bottom:4px}
.scene-sub{font-size:9px;color:var(--muted);letter-spacing:2px}
.task-grid{display:flex;gap:10px;position:relative;z-index:4;width:100%;max-width:580px}
.task-card{flex:1;padding:12px 10px;background:rgba(6,16,28,0.7);border:1px solid var(--border);border-radius:6px;cursor:pointer;text-align:center;transition:all .22s;backdrop-filter:blur(8px);}
.task-card:hover{border-color:var(--border-hot);background:var(--cyan-dim)}
.task-card.sel-easy{border-color:var(--cyan);background:rgba(0,212,255,0.07);box-shadow:0 0 18px rgba(0,212,255,0.12)}
.task-card.sel-medium{border-color:var(--amber);background:rgba(255,184,0,0.07);box-shadow:0 0 18px rgba(255,184,0,0.12)}
.task-card.sel-hard{border-color:var(--red);background:rgba(255,61,61,0.07);box-shadow:0 0 18px rgba(255,61,61,0.12)}
.t-icon{font-size:18px;margin-bottom:4px}
.t-name{font-family:var(--display);font-size:9px;font-weight:700;letter-spacing:2px;margin-bottom:3px}
.t-desc{font-size:9px;color:var(--muted);line-height:1.5;margin-bottom:5px}
.t-tags{display:flex;gap:4px;justify-content:center;flex-wrap:wrap}
.ttag{font-size:8px;padding:1px 6px;border-radius:2px;border:1px solid}
.ttag-c{color:var(--cyan);border-color:rgba(0,212,255,.3);background:rgba(0,212,255,.07)}
.ttag-a{color:var(--amber);border-color:rgba(255,184,0,.3);background:rgba(255,184,0,.07)}
.ttag-r{color:var(--red);border-color:rgba(255,61,61,.3);background:rgba(255,61,61,.07)}
.stat-strip{display:none;gap:10px;position:relative;z-index:4;padding:8px 14px;background:rgba(0,0,0,0.35);border:1px solid var(--border);border-radius:4px;width:100%;max-width:580px;}
.stat-item{flex:1;text-align:center}
.stat-lbl{font-size:7px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px}
.stat-val{font-family:var(--display);font-size:12px;font-weight:700;color:var(--cyan);transition:color .4s}
.subsys-bars{display:none;gap:8px;position:relative;z-index:4;width:100%;max-width:580px;}
.subsys-bar-item{flex:1;text-align:center}
.subsys-lbl{font-size:7px;color:var(--muted);letter-spacing:1px;margin-bottom:3px;text-transform:uppercase}
.subsys-track{height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
.subsys-fill{height:100%;border-radius:2px;transition:width .6s ease,background .4s}
.launch-btn{font-family:var(--display);font-size:11px;font-weight:700;letter-spacing:3px;padding:12px 40px;background:linear-gradient(135deg,rgba(0,212,255,.12),rgba(0,80,140,.18));border:1px solid var(--cyan);color:var(--cyan);cursor:pointer;border-radius:3px;transition:all .2s;text-transform:uppercase;box-shadow:0 0 18px rgba(0,212,255,.1);position:relative;z-index:4;}
.launch-btn:hover{background:rgba(0,212,255,.2);box-shadow:0 0 28px rgba(0,212,255,.25);transform:translateY(-1px)}
.launch-btn:active{transform:none}
.launch-btn:disabled{opacity:.35;cursor:not-allowed;transform:none}
.launch-btn.running{border-color:var(--amber);color:var(--amber);background:rgba(255,184,0,.08);animation:bpulse .9s ease-in-out infinite}
@keyframes bpulse{0%,100%{box-shadow:0 0 16px rgba(255,184,0,.1)}50%{box-shadow:0 0 32px rgba(255,184,0,.35)}}
.right{display:flex;flex-direction:column;min-height:0;background:rgba(2,6,10,0.55);backdrop-filter:blur(14px)}
.score-bar{flex-shrink:0;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;border-bottom:1px solid var(--border);}
.score-cell{padding:10px 12px;text-align:center;border-right:1px solid var(--border)}
.score-cell:last-child{border-right:none}
.sc-lbl{font-size:7px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px}
.sc-val{font-family:var(--display);font-size:17px;font-weight:700;color:var(--cyan)}
.sc-val.green{color:var(--green);text-shadow:0 0 12px rgba(0,255,136,.3)}
.sc-val.amber{color:var(--amber);text-shadow:0 0 12px rgba(255,184,0,.3)}
.sc-val.red{color:var(--red);text-shadow:0 0 12px rgba(255,61,61,.3)}
.chart-wrap{flex-shrink:0;padding:10px 16px 6px;border-bottom:1px solid var(--border)}
.chart-lbl{font-size:7px;color:var(--muted);letter-spacing:2px;margin-bottom:5px;display:flex;justify-content:space-between}
svg.rchart{width:100%;height:52px;display:block}
.wm-ticker{flex-shrink:0;padding:5px 16px;border-bottom:1px solid var(--border);background:rgba(0,0,0,0.2);font-size:9px;color:var(--muted);letter-spacing:1px;white-space:nowrap;overflow:hidden;}
.wm-ticker span{color:var(--cyan)}
.feed-wrap{flex:1;overflow-y:auto;padding:8px 12px;display:flex;flex-direction:column;gap:4px}
.feed-wrap::-webkit-scrollbar{width:2px}
.feed-wrap::-webkit-scrollbar-thumb{background:var(--muted);border-radius:1px}
.idle{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;padding:28px;text-align:center;}
.idle-icon{font-size:36px;opacity:.2;animation:float 4s ease-in-out infinite}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
.idle-txt{font-size:10px;color:var(--muted);line-height:2;letter-spacing:1px}
.idle-pills{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
.idle-pill{font-size:8px;color:var(--muted);padding:4px 10px;border:1px solid var(--border);border-radius:2px;letter-spacing:.8px}
.idle-pill.theme{color:var(--purple);border-color:rgba(168,85,247,.3)}
.step-card{padding:8px 10px;background:rgba(255,255,255,0.018);border:1px solid rgba(255,255,255,0.05);border-left:2px solid transparent;border-radius:3px;opacity:0;transform:translateY(6px);animation:sin .28s ease forwards;display:grid;grid-template-columns:26px 1fr auto;gap:8px;align-items:start;}
@keyframes sin{to{opacity:1;transform:none}}
.step-card.lv-high{border-left-color:var(--green)}
.step-card.lv-mid{border-left-color:var(--cyan)}
.step-card.lv-low{border-left-color:var(--amber)}
.step-card.lv-crit{border-left-color:var(--red)}
.sn{font-family:var(--display);font-size:9px;color:var(--muted);padding-top:2px}
.sa{font-family:var(--display);font-size:10px;font-weight:700;color:var(--cyan);margin-bottom:2px;letter-spacing:.5px}
.sr-text{font-size:9px;color:#4a6a7a;line-height:1.5;margin-bottom:3px}
.st{font-size:9px;color:rgba(180,210,225,.35);margin-bottom:3px}
.fbars{display:flex;flex-direction:column;gap:2px;margin-top:2px}
.fbar-row{display:grid;grid-template-columns:110px 1fr 28px;align-items:center;gap:5px}
.fn{font-size:8px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fb{height:3px;background:rgba(255,255,255,.05);border-radius:1px;overflow:hidden}
.ff{height:100%;border-radius:1px;transition:width .5s}
.fp{font-size:8px;text-align:right}
.specs{display:flex;gap:4px;margin-top:3px;flex-wrap:wrap}
.spec-pill{font-size:8px;padding:1px 6px;border-radius:2px;border:1px solid;letter-spacing:.3px}
.srw{font-family:var(--display);font-size:12px;font-weight:700;padding-top:2px}
.verdict{flex-shrink:0;border-top:1px solid var(--border);padding:10px 16px;display:none}
.verdict.show{display:block}
.vd-title{font-family:var(--display);font-size:8px;letter-spacing:2px;color:var(--muted);margin-bottom:7px;display:flex;align-items:center;gap:8px}
.vd-title-badge{padding:2px 8px;border-radius:2px;font-size:8px;font-family:var(--display)}
.vd-grid{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:5px}
.vd-cell{padding:6px 8px;background:rgba(0,0,0,.25);border:1px solid var(--border);border-radius:2px;text-align:center}
.vd-lbl{font-size:7px;color:var(--muted);letter-spacing:1.5px;margin-bottom:2px}
.vd-val{font-family:var(--display);font-size:11px;font-weight:700}
</style>
</head>
<body>
<div class="stars" id="stars"></div>
<div class="shell">
<header>
  <div class="logo-block">
    <div class="live-dot"></div>
    <div><div class="logo-text">ORBITAL ANOMALY</div><div class="logo-sub">MISSION CONTROL · OPENENV v2.2</div></div>
  </div>
  <div class="header-center" id="header-center" style="display:none">
    <div class="hstat"><div class="hstat-lbl">TASK</div><div class="hstat-val" id="hc-task">—</div></div>
    <div class="hstat"><div class="hstat-lbl">STEP</div><div class="hstat-val" id="hc-step">—</div></div>
    <div class="hstat"><div class="hstat-lbl">PHASE</div><div class="hstat-val" id="hc-phase">—</div></div>
    <div class="hstat"><div class="hstat-lbl">AVG REWARD</div><div class="hstat-val" id="hc-avg" style="color:var(--green)">—</div></div>
  </div>
  <div class="header-right">
    <a href="/docs" class="hlink">API DOCS</a>
    <a href="/state" class="hlink">RAW STATE</a>
    <a href="https://github.com/umed-indulkar/orbital-anomaly-openenv" class="hlink" target="_blank">GITHUB</a>
    <a href="https://colab.research.google.com/github/umed-indulkar/orbital-anomaly-openenv/blob/main/Orbital_Anomaly_openenv_V2.ipynb" class="hlink hl-green" target="_blank">▶ TRAIN</a>
  </div>
</header>
<main>
<div class="left">
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
  <div class="scene-header" style="margin-bottom:2px">
    <div class="scene-title" id="scene-title">SELECT MISSION</div>
    <div class="scene-sub" id="scene-sub">CHOOSE A CRISIS SCENARIO · AI AGENT WILL RESPOND</div>
  </div>
  <div class="stat-strip" id="stat-strip" style="display:none">
    <div class="stat-item"><div class="stat-lbl">Battery</div><div class="stat-val" id="ls-bat">—</div></div>
    <div class="stat-item"><div class="stat-lbl">Thermal</div><div class="stat-val" id="ls-temp">—</div></div>
    <div class="stat-item"><div class="stat-lbl">Comms</div><div class="stat-val" id="ls-comms">—</div></div>
    <div class="stat-item"><div class="stat-lbl">Phase</div><div class="stat-val" id="ls-phase">—</div></div>
    <div class="stat-item"><div class="stat-lbl">Dominant</div><div class="stat-val" id="ls-dom" style="font-size:9px">—</div></div>
  </div>
  <div class="subsys-bars" id="subsys-bars" style="display:none">
    <div class="subsys-bar-item"><div class="subsys-lbl">EPS</div><div class="subsys-track"><div class="subsys-fill" id="sb-eps" style="width:0%;background:var(--cyan)"></div></div></div>
    <div class="subsys-bar-item"><div class="subsys-lbl">Thermal</div><div class="subsys-track"><div class="subsys-fill" id="sb-therm" style="width:0%;background:var(--cyan)"></div></div></div>
    <div class="subsys-bar-item"><div class="subsys-lbl">Comms</div><div class="subsys-track"><div class="subsys-fill" id="sb-comms" style="width:0%;background:var(--cyan)"></div></div></div>
    <div class="subsys-bar-item"><div class="subsys-lbl">ADCS</div><div class="subsys-track"><div class="subsys-fill" id="sb-adcs" style="width:0%;background:var(--cyan)"></div></div></div>
  </div>
  <div class="task-grid">
    <div class="task-card sel-easy" id="tc-easy" onclick="selectTask('easy')">
      <div class="t-icon">🟢</div><div class="t-name">EASY</div>
      <div class="t-desc">EPS Crisis<br>Single fault · Sunlit</div>
      <div class="t-tags"><span class="ttag ttag-c">SOC 38%</span><span class="ttag ttag-c">1 FAULT</span><span class="ttag ttag-c">MPPT</span></div>
    </div>
    <div class="task-card" id="tc-medium" onclick="selectTask('medium')">
      <div class="t-icon">🟡</div><div class="t-name">MEDIUM</div>
      <div class="t-desc">Thermal + Science<br>Dual fault · Tradeoff</div>
      <div class="t-tags"><span class="ttag ttag-a">68°C</span><span class="ttag ttag-a">2 FAULTS</span><span class="ttag ttag-a">+0.12 BONUS</span></div>
    </div>
    <div class="task-card" id="tc-hard" onclick="selectTask('hard')">
      <div class="t-icon">🔴</div><div class="t-name">HARD</div>
      <div class="t-desc">Cascade Failure<br>Eclipse · 7 faults</div>
      <div class="t-tags"><span class="ttag ttag-r">SOC 22%</span><span class="ttag ttag-r">ECLIPSE</span><span class="ttag ttag-r">7 FAULTS</span></div>
    </div>
  </div>
  <button class="launch-btn" id="launch-btn" onclick="launchMission()">▶ LAUNCH MISSION</button>
</div>
<div class="right">
  <div class="score-bar">
    <div class="score-cell"><div class="sc-lbl">AVG REWARD</div><div class="sc-val" id="sc-avg">—</div></div>
    <div class="score-cell"><div class="sc-lbl">PEAK REWARD</div><div class="sc-val" id="sc-peak">—</div></div>
    <div class="score-cell"><div class="sc-lbl">STEPS</div><div class="sc-val" id="sc-steps">—</div></div>
    <div class="score-cell"><div class="sc-lbl">STATUS</div><div class="sc-val" id="sc-status">—</div></div>
  </div>
  <div class="chart-wrap">
    <div class="chart-lbl"><span>REWARD TIMELINE — AGENT PERFORMANCE</span><span id="chart-trend" style="color:var(--muted)">—</span></div>
    <svg class="rchart" viewBox="0 0 460 52" preserveAspectRatio="none">
      <defs>
        <linearGradient id="rgrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="rgba(0,212,255,.3)"/><stop offset="100%" stop-color="rgba(0,212,255,0)"/></linearGradient>
      </defs>
      <line x1="0" y1="37" x2="460" y2="37" stroke="rgba(255,184,0,.2)" stroke-width="1" stroke-dasharray="3"/>
      <text x="4" y="35" font-size="7" fill="rgba(255,184,0,.4)" font-family="Space Mono,monospace">0.45</text>
      <path id="chart-area" fill="url(#rgrad)" d=""/>
      <polyline id="chart-line" fill="none" stroke="#00d4ff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" points=""/>
    </svg>
  </div>
  <div class="wm-ticker" id="wm-ticker">WORLD MODEL — <span>awaiting mission launch</span></div>
  <div class="feed-wrap" id="feed-wrap">
    <div class="idle" id="idle-el">
      <div class="idle-icon">🛰️</div>
      <div class="idle-txt">SELECT A MISSION ABOVE<br>THE MULTI-AGENT AI EXECUTES 12 DECISION STEPS<br>COMMANDER DELEGATES TO 3 SPECIALIST AGENTS<br>WORLD MODEL: 13-FAULT BELIEF STATE PER STEP</div>
      <div class="idle-pills">
        <span class="idle-pill theme">THEME 3 · WORLD MODELING</span>
        <span class="idle-pill theme">THEME 2 · LONG-HORIZON PLANNING</span>
        <span class="idle-pill theme">THEME 1 · MULTI-AGENT</span>
      </div>
      <div class="idle-pills" style="margin-top:4px">
        <span class="idle-pill">OpenEnv v2.2</span>
        <span class="idle-pill">GRPO · Unsloth · Qwen2.5</span>
        <span class="idle-pill">Live API ↗</span>
      </div>
    </div>
  </div>
  <div class="verdict" id="verdict">
    <div class="vd-title">MISSION DEBRIEF<span class="vd-title-badge" id="vd-badge" style="background:rgba(0,212,255,.1);color:var(--cyan);border:1px solid rgba(0,212,255,.3)">—</span></div>
    <div class="vd-grid">
      <div class="vd-cell"><div class="vd-lbl">TASK</div><div class="vd-val" id="vd-task" style="color:var(--cyan)">—</div></div>
      <div class="vd-cell"><div class="vd-lbl">AVG REWARD</div><div class="vd-val" id="vd-avg">—</div></div>
      <div class="vd-cell"><div class="vd-lbl">PEAK REWARD</div><div class="vd-val" id="vd-peak" style="color:var(--green)">—</div></div>
      <div class="vd-cell"><div class="vd-lbl">FINAL STATUS</div><div class="vd-val" id="vd-fin">—</div></div>
    </div>
  </div>
</div>
</main>
</div>
<script>
(function(){
  const c=document.getElementById('stars');
  for(let i=0;i<130;i++){
    const s=document.createElement('div');s.className='star';
    const sz=Math.random()<.85?1:Math.random()<.7?1.5:2;
    s.style.cssText=`width:${sz}px;height:${sz}px;left:${Math.random()*100}%;top:${Math.random()*100}%;`+
      `--d:${2+Math.random()*4}s;--delay:${Math.random()*4}s;--min:${.05+Math.random()*.1};--max:${.4+Math.random()*.6}`;
    c.appendChild(s);
  }
})();
let selTask='easy',running=false;
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
  ['easy','medium','hard'].forEach(t=>{document.getElementById('tc-'+t).className='task-card'+(t===task?' sel-'+t:'');});
  const glow=document.getElementById('sat-glow'),orbit=document.getElementById('sat-orbit');
  const sig=document.getElementById('sig-ring'),wl=document.getElementById('wing-l'),wr=document.getElementById('wing-r');
  const sub=document.getElementById('scene-sub');
  glow.className='sat-glow';orbit.className='sat-orbit';
  wl.classList.remove('mis');wr.classList.remove('mis');sig.style.display='';
  if(task==='hard'){glow.classList.add('crit');orbit.classList.add('fast');wl.classList.add('mis');wr.classList.add('mis');sig.style.display='none';sub.textContent='ECLIPSE · 7 FAULTS · SOC 22% · GS BLACKOUT · RADIATION ZONE';}
  else if(task==='medium'){glow.classList.add('warn');sub.textContent='THERMAL CRISIS · 2 FAULTS · PAYLOAD TEMP 68°C · SCIENCE WINDOW ACTIVE';}
  else{sub.textContent='EPS CRISIS · 1 FAULT (MPPT STUCK) · SOC 38% · SUNLIT';}
}
function updateChart(){
  if(!rewards.length)return;
  const W=460,H=52,pad=4,n=rewards.length;
  const pts=rewards.map((r,i)=>{const x=n===1?W/2:(i/(Math.max(n-1,1)))*W;const y=H-pad-(Math.max(0,Math.min(1,r))*(H-pad*2));return[x.toFixed(1),y.toFixed(1)];});
  document.getElementById('chart-line').setAttribute('points',pts.map(p=>p.join(',')).join(' '));
  document.getElementById('chart-area').setAttribute('d',`M${pts[0][0]},${H} `+pts.map(p=>`L${p[0]},${p[1]}`).join(' ')+` L${pts[pts.length-1][0]},${H} Z`);
  if(n>=3){const first=rewards.slice(0,3).reduce((a,b)=>a+b,0)/3,last=rewards.slice(-3).reduce((a,b)=>a+b,0)/3,diff=last-first;
    const trend=diff>0.03?'↑ IMPROVING':diff<-0.03?'↓ DECLINING':'→ STABLE';
    const col=diff>0.03?'var(--green)':diff<-0.03?'var(--red)':'var(--amber)';
    document.getElementById('chart-trend').style.color=col;document.getElementById('chart-trend').textContent=trend;}
}
function updateSubsysBars(t){
  if(!t)return;
  document.getElementById('subsys-bars').style.display='flex';
  const eps=Math.max(0,Math.min(100,t.battery_soc||0));
  const epsEl=document.getElementById('sb-eps');epsEl.style.width=eps+'%';epsEl.style.background=eps<20?'var(--red)':eps<40?'var(--amber)':'var(--cyan)';
  const therm=Math.max(0,Math.min(100,100-(((t.thermal_temp||40)-20)/70)*100));
  const thermEl=document.getElementById('sb-therm');thermEl.style.width=therm+'%';thermEl.style.background=therm<30?'var(--red)':therm<50?'var(--amber)':'var(--cyan)';
  const comms=Math.max(0,Math.min(100,t.comms_signal||0));
  const commsEl=document.getElementById('sb-comms');commsEl.style.width=comms+'%';commsEl.style.background=comms<30?'var(--red)':comms<60?'var(--amber)':'var(--green)';
  const adcs=Math.max(0,Math.min(100,100-(t.wheel_sat||0)));
  const adcsEl=document.getElementById('sb-adcs');adcsEl.style.width=adcs+'%';adcsEl.style.background=adcs<30?'var(--red)':adcs<60?'var(--amber)':'var(--purple)';
}
function buildFaultBars(beliefs){
  if(!beliefs)return'';
  return`<div class="fbars">`+Object.entries(beliefs).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([n,p])=>
    `<div class="fbar-row"><span class="fn">${n.replace(/_/g,' ')}</span><div class="fb"><div class="ff" style="width:${(p*100).toFixed(0)}%;background:${fColor(p)}"></div></div><span class="fp" style="color:${fColor(p)}">${Math.round(p*100)}%</span></div>`).join('')+`</div>`;
}
function buildSpecs(specs){
  if(!specs)return'';
  return`<div class="specs">`+Object.entries(specs).map(([name,s])=>{
    const short=name.replace('_Specialist','');const bg=SPEC_COLORS[name]||'rgba(255,255,255,.05)';const bc=SPEC_BORDER[name]||'rgba(255,255,255,.15)';
    return`<span class="spec-pill" style="background:${bg};border-color:${bc};color:var(--text)">${short}: ${s.action.replace(/_/g,' ')} <span style="opacity:.6">${Math.round(s.confidence*100)}%</span></span>`;
  }).join('')+`</div>`;
}
async function launchMission(){
  if(running)return;running=true;rewards.length=0;
  const btn=document.getElementById('launch-btn'),feed=document.getElementById('feed-wrap');
  btn.textContent='⟳ EXECUTING...';btn.className='launch-btn running';btn.disabled=true;
  feed.innerHTML='';
  const idleEl=document.getElementById('idle-el');if(idleEl)idleEl.style.display='none';
  document.getElementById('verdict').classList.remove('show');
  ['sc-avg','sc-peak','sc-steps','sc-status'].forEach(id=>{document.getElementById(id).textContent='—';document.getElementById(id).className='sc-val';});
  document.getElementById('chart-line').setAttribute('points','');document.getElementById('chart-area').setAttribute('d','');
  document.getElementById('chart-trend').textContent='—';document.getElementById('chart-trend').style.color='var(--muted)';
  document.getElementById('wm-ticker').innerHTML='WORLD MODEL — <span style="color:var(--amber)">MISSION IN PROGRESS...</span>';
  document.getElementById('stat-strip').style.display='flex';
  document.getElementById('header-center').style.display='flex';
  document.getElementById('hc-task').textContent=selTask.toUpperCase();
  document.getElementById('hc-step').textContent='0/12';document.getElementById('hc-phase').textContent='1/3';document.getElementById('hc-avg').textContent='—';
  try{
    const res=await fetch('/run_episode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:selTask})});
    if(!res.ok){const txt=await res.text();throw new Error(`Server error ${res.status}: ${txt.slice(0,300)}`);}
    const data=await res.json();
    if(data.error){throw new Error(data.error+': '+(data.detail||'').slice(0,300))}
    let peakReward=0;
    for(let i=0;i<data.steps.length;i++){
      await new Promise(r=>setTimeout(r,120));
      const s=data.steps[i],t=s.telemetry||{},wm=s.world_model||{};
      const icon=ICONS[s.action]||'⚡',cls=rClass(s.reward),col=rColor(s.reward);
      if(s.reward>peakReward)peakReward=s.reward;
      document.getElementById('hc-step').textContent=`${i+1}/${data.steps.length}`;
      document.getElementById('hc-phase').textContent=`${(wm.phase||0)+1}/3`;
      document.getElementById('ls-bat').textContent=t.battery_soc+'%';
      document.getElementById('ls-bat').style.color=t.battery_soc<20?'var(--red)':t.battery_soc<40?'var(--amber)':'var(--cyan)';
      document.getElementById('ls-temp').textContent=t.thermal_temp+'°C';
      document.getElementById('ls-temp').style.color=t.thermal_temp>85?'var(--red)':t.thermal_temp>70?'var(--amber)':'var(--cyan)';
      document.getElementById('ls-comms').textContent=t.comms_signal+'%';
      document.getElementById('ls-comms').style.color=t.comms_signal<30?'var(--red)':t.comms_signal<60?'var(--amber)':'var(--cyan)';
      document.getElementById('ls-phase').textContent='P'+((wm.phase||0)+1)+'/3';
      document.getElementById('ls-dom').textContent=(wm.dominant_subsystem||'—').toUpperCase();
      updateSubsysBars(t);
      const glow=document.getElementById('sat-glow');
      if(t.mission_status==='critical'){glow.className='sat-glow crit'}else if(t.mission_status==='warning'){glow.className='sat-glow warn'}else{glow.className='sat-glow'}
      document.getElementById('wm-ticker').innerHTML=
        `WORLD MODEL — Dom: <span>${(wm.dominant_subsystem||'?').toUpperCase()}</span> &nbsp;·&nbsp; ${wm.top_faults||'—'} &nbsp;·&nbsp; Phase <span>${(wm.phase||0)+1}/3</span>`+
        (!t.sunlit?' &nbsp;·&nbsp; <span style="color:var(--amber)">🌑 ECLIPSE</span>':'')+
        (!t.gs_visible?' &nbsp;·&nbsp; <span style="color:var(--red)">GS BLACKOUT</span>':'')+
        (t.safe_mode?' &nbsp;·&nbsp; <span style="color:var(--red)">SAFE MODE</span>':'');
      const el=document.createElement('div');el.className=`step-card ${cls}`;
      el.innerHTML=`<div class="sn">${String(s.step).padStart(2,'0')}</div>`+
        `<div class="sb"><div class="sa">${icon} ${s.action.replace(/_/g,' ').toUpperCase()}</div>`+
        `<div class="sr-text">${s.rationale}</div>`+
        `<div class="st">BAT ${t.battery_soc}% · TEMP ${t.thermal_temp}°C · COMMS ${t.comms_signal}%`+
        `${!t.sunlit?' · <span style="color:var(--amber)">ECLIPSE</span>':''}${t.safe_mode?' · <span style="color:var(--red)">SAFE MODE</span>':''}`+
        ` · <span style="color:${statusColor(t.mission_status)}">${(t.mission_status||'stable').toUpperCase()}</span></div>`+
        buildFaultBars(wm.fault_beliefs)+buildSpecs(s.specialists)+`</div>`+
        `<div class="srw" style="color:${col}">${s.reward.toFixed(3)}</div>`;
      feed.appendChild(el);feed.scrollTop=feed.scrollHeight;
      rewards.push(s.reward);
      const avg=rewards.reduce((a,b)=>a+b,0)/rewards.length;
      document.getElementById('sc-avg').textContent=avg.toFixed(3);document.getElementById('sc-avg').className='sc-val '+(avg>=.7?'green':avg>=.5?'':'amber');
      document.getElementById('sc-peak').textContent=peakReward.toFixed(3);document.getElementById('sc-peak').className='sc-val '+(peakReward>=.7?'green':'');
      document.getElementById('sc-steps').textContent=`${i+1} / ${data.steps.length}`;
      document.getElementById('sc-status').textContent=(t.mission_status||'stable').toUpperCase();
      document.getElementById('sc-status').className='sc-val '+(t.mission_status==='critical'?'red':t.mission_status==='warning'?'amber':'green');
      document.getElementById('hc-avg').textContent=avg.toFixed(3);
      updateChart();
    }
    const finalAvg=rewards.reduce((a,b)=>a+b,0)/rewards.length,success=finalAvg>=0.45;
    document.getElementById('vd-task').textContent=data.task_id.toUpperCase();
    document.getElementById('vd-avg').textContent=finalAvg.toFixed(4);document.getElementById('vd-avg').style.color=rColor(finalAvg);
    document.getElementById('vd-peak').textContent=peakReward.toFixed(4);
    document.getElementById('vd-steps').textContent=data.total_steps;
    document.getElementById('vd-fin').textContent=(data.final_status||'stable').toUpperCase();document.getElementById('vd-fin').style.color=statusColor(data.final_status);
    const badge=document.getElementById('vd-badge');
    if(success){badge.textContent='MISSION SUCCESS ✓';badge.style.cssText='background:rgba(0,255,136,.1);color:var(--green);border:1px solid rgba(0,255,136,.3);padding:2px 8px;border-radius:2px;font-size:8px';}
    else{badge.textContent='MISSION CRITICAL ✗';badge.style.cssText='background:rgba(255,61,61,.1);color:var(--red);border:1px solid rgba(255,61,61,.3);padding:2px 8px;border-radius:2px;font-size:8px';}
    document.getElementById('verdict').classList.add('show');
    document.getElementById('wm-ticker').innerHTML='WORLD MODEL — <span style="color:'+(success?'var(--green)':'var(--amber)')+'">MISSION '+(success?'COMPLETE ✓':'ENDED')+' · AVG REWARD: '+finalAvg.toFixed(4)+'</span>';
  }catch(e){
    const el=document.createElement('div');
    el.style.cssText='color:var(--red);padding:14px;font-size:10px;background:rgba(255,61,61,.08);border:1px solid rgba(255,61,61,.2);border-radius:3px;margin:8px;line-height:1.7;font-family:monospace';
    el.textContent='Error: '+e.message;feed.appendChild(el);
  }
  running=false;btn.textContent='▶ RUN AGAIN';btn.className='launch-btn';btn.disabled=false;
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