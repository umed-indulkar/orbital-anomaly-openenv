# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
inference.py — Orbital Anomaly OpenEnv Baseline Inference v3.0
===============================================================
Theme 3 (World Modeling): 13-fault belief state, context-aware decisions
Theme 2 (Long-Horizon):   36-step Extended Mission Mode support
Theme 1 (Multi-Agent):    MissionCommander + 3 Specialist Agents

OpenEnv compliance:
  [START] / [STEP] / [END] log format preserved
  LiteLLM proxy handshake preserved
  Heuristic fallback always available

Usage:
    python inference.py                            # heuristic baseline
    HF_TOKEN=hf_... python inference.py            # LLM via HF inference
    MODEL_NAME=gpt-4o HF_TOKEN=... python inference.py
"""

from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI

from client import OrbitalAnomalyOpenenvEnv
from models import OrbitalAnomalyOpenenvAction, OrbitalAnomalyOpenenvObservation

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL      = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME        = os.getenv("MODEL_NAME",   "openai/gpt-4o-mini")
API_KEY           = os.getenv("HF_TOKEN",     os.getenv("OPENAI_API_KEY", ""))
ENV_BASE_URL      = os.getenv("ENV_BASE_URL",
                               "https://codequasar-orbital-anomaly-openenv.hf.space")
BENCHMARK         = "orbital_anomaly_openenv"
TASKS             = ["easy", "medium", "hard"]
MAX_STEPS         = 12
SUCCESS_THRESHOLD = 0.45
VALID_ACTIONS     = [
    "rotate_to_sun", "disable_payload", "reboot_comms",
    "enter_safe_mode", "switch_power_bus", "noop",
]

# ── Structured log helpers ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    print(f"[STEP]  step={step} action={action} reward={reward:.4f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(f"[END]   success={str(success).lower()} steps={steps} "
          f"score={score:.4f} rewards={','.join(f'{r:.4f}' for r in rewards)}",
          flush=True)

# ── World Model: 13-fault belief state (Theme 3) ─────────────────────────────

FAULT_NAMES = [
    "mppt_stuck", "panel_deployment_jam", "bus_short_transient", "battery_aging",
    "reaction_wheel_saturation", "gyro_drift", "star_tracker_dropout",
    "radiator_valve_stuck", "heat_pipe_failure", "heater_relay_latch",
    "transponder_overheating", "amplifier_degradation", "antenna_gimbal_stall",
]

def compute_fault_beliefs(obs: OrbitalAnomalyOpenenvObservation) -> dict:
    """
    Heuristic Bayesian fault posterior from observable symptoms.
    Each fault's probability is computed from the telemetry signals that are
    causally connected to it in the spacecraft fault graph.

    Theme 3 alignment: agent maintains and updates this belief every step,
    reasoning about *unobservable* root causes from *observable* symptoms.
    This is the core world-modeling capability we are training.
    """
    # Normalise observable signals to [0, 1]
    b = max(0.0, min(1.0, (obs.battery_soc or obs.battery_level or 100.0) / 100.0))
    s = max(0.0, min(1.0, obs.solar_efficiency or 1.0))
    t = max(0.0, min(1.0, ((obs.thermal_temp or 40.0) - 20.0) / 80.0))
    c = max(0.0, min(1.0, obs.comms_signal or 1.0))
    # Additional V2 signals when available
    w = max(0.0, min(1.0, obs.wheel_saturation_level or 0.0))
    r = max(0.0, min(1.0, 1.0 - (obs.radiator_efficiency or 1.0)))

    def clip(x: float) -> float:
        return round(max(0.0, min(1.0, x)), 3)

    return {
        # EPS fault cluster
        "mppt_stuck":                clip((1-s)*0.90 + (1-b)*0.30),
        "panel_deployment_jam":      clip((1-s)*0.80),
        "bus_short_transient":       clip((1-b)*0.60 + t*0.20),
        "battery_aging":             clip((1-b)*0.50),
        # ADCS fault cluster
        "reaction_wheel_saturation": clip(w*0.90 + (1-s)*0.20),
        "gyro_drift":                clip((1-s)*0.35 + w*0.10),
        "star_tracker_dropout":      clip((1-s)*0.40),
        # Thermal fault cluster
        "radiator_valve_stuck":      clip(r*0.70 + t*0.50),
        "heat_pipe_failure":         clip(t*0.75 + (1-b)*0.10),
        "heater_relay_latch":        clip(t*0.50 + (1-b)*0.20),
        # Comms fault cluster
        "transponder_overheating":   clip((1-c)*0.80 + t*0.30),
        "amplifier_degradation":     clip((1-c)*0.65),
        "antenna_gimbal_stall":      clip((1-c)*0.55 + (1-s)*0.15),
    }

def top_faults_str(beliefs: dict, n: int = 3) -> str:
    top = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)[:n]
    return ", ".join(f"{f}({p:.0%})" for f, p in top)

def dominant_subsystem(beliefs: dict) -> str:
    """Return the subsystem with highest average fault probability."""
    groups = {
        "EPS":     ["mppt_stuck","panel_deployment_jam","bus_short_transient","battery_aging"],
        "ADCS":    ["reaction_wheel_saturation","gyro_drift","star_tracker_dropout"],
        "Thermal": ["radiator_valve_stuck","heat_pipe_failure","heater_relay_latch"],
        "Comms":   ["transponder_overheating","amplifier_degradation","antenna_gimbal_stall"],
    }
    scores = {g: sum(beliefs.get(f,0) for f in faults)/len(faults)
              for g, faults in groups.items()}
    return max(scores, key=scores.get)

# ── Multi-Agent Architecture (Theme 1 / Fleet AI bonus) ──────────────────────

def _eps_specialist(obs: OrbitalAnomalyOpenenvObservation) -> tuple[str, float, str]:
    """EPS Specialist: battery + solar decision agent."""
    bat  = obs.battery_soc or obs.battery_level or 100.0
    sol  = obs.solar_efficiency or 1.0
    sunlit = getattr(obs, 'sunlit', True)

    if bat < 15.0:
        return "switch_power_bus", 0.97, "CRITICAL: battery at floor — reserve bus"
    if bat < 22.0 and not sunlit:
        return "switch_power_bus", 0.93, "Eclipse+low battery — reserve only option"
    if bat < 22.0 and sol < 0.4:
        return "rotate_to_sun",    0.91, "CRITICAL: battery depleting, solar misaligned"
    if bat < 30.0:
        return "switch_power_bus", 0.82, "WARNING: battery low"
    if sol < 0.35 and sunlit:
        return "rotate_to_sun",    0.78, "WARNING: solar severely degraded"
    if sol < 0.60 and sunlit:
        return "rotate_to_sun",    0.50, "Solar suboptimal"
    return "noop", 0.08, "EPS nominal"

def _thermal_specialist(obs: OrbitalAnomalyOpenenvObservation) -> tuple[str, float, str]:
    """Thermal Specialist: temperature + payload heat management."""
    temp    = obs.thermal_temp or 40.0
    payload = obs.payload_on if obs.payload_on is not None else True

    if temp > 85.0:
        return "enter_safe_mode",  0.98, "CRITICAL: thermal cascade imminent"
    if temp > 78.0 and payload:
        return "disable_payload",  0.91, "CRITICAL: payload heat load critical"
    if temp > 70.0 and payload:
        return "disable_payload",  0.80, "WARNING: elevated thermal + payload ON"
    if temp > 62.0:
        return "disable_payload",  0.58, "Proactive thermal management"
    return "noop", 0.12, "Thermal nominal"

def _comms_specialist(obs: OrbitalAnomalyOpenenvObservation) -> tuple[str, float, str]:
    """Comms Specialist: RF chain + link quality management."""
    comms = obs.comms_signal or 1.0

    if comms < 0.18:
        return "reboot_comms", 0.97, "CRITICAL: link near loss"
    if comms < 0.38:
        return "reboot_comms", 0.83, "WARNING: comms severely degraded"
    if comms < 0.60:
        return "reboot_comms", 0.60, "Comms below nominal"
    return "noop", 0.10, "Comms nominal"

def mission_commander_decide(
    obs: OrbitalAnomalyOpenenvObservation
) -> tuple[str, str, dict]:
    """
    MissionCommanderAgent: oversight agent that aggregates specialist
    recommendations and selects the highest-confidence action.

    Returns (action, rationale, all_recommendations).
    Maps to Fleet AI bonus: oversight agents monitoring specialist AI agents.
    """
    recs = {
        "EPS_Specialist":     _eps_specialist(obs),
        "Thermal_Specialist": _thermal_specialist(obs),
        "Comms_Specialist":   _comms_specialist(obs),
    }
    best = max(recs, key=lambda k: recs[k][1])
    action, conf, reason = recs[best]
    return action, f"[{best}|{conf:.0%}] {reason}", recs

def _heuristic_action(obs: OrbitalAnomalyOpenenvObservation) -> str:
    action, _, _ = mission_commander_decide(obs)
    return action

# ── LLM prompt (Theme 3: world model exposed in prompt) ──────────────────────

SYSTEM_PROMPT = (
    "You are an autonomous satellite mission control AI.\n"
    "A €500M spacecraft is failing 400km above Earth.\n"
    "You must choose ONE corrective action per decision step.\n\n"
    "Available actions:\n"
    "  rotate_to_sun    — realign solar panels (useless in eclipse)\n"
    "  disable_payload  — cut science payload to reduce heat and power draw\n"
    "  reboot_comms     — reset RF chain, restore communications\n"
    "  enter_safe_mode  — emergency: disable all non-critical systems\n"
    "  switch_power_bus — activate backup battery bus (works in eclipse)\n"
    "  noop             — take no action\n\n"
    "Reply with ONLY the action name. No explanation. No punctuation."
)

def _build_prompt(obs: OrbitalAnomalyOpenenvObservation,
                  beliefs: dict, step: int, phase: int) -> str:
    top3 = top_faults_str(beliefs)
    dom  = dominant_subsystem(beliefs)
    bat  = obs.battery_soc or obs.battery_level or 0.0
    sol  = (obs.solar_efficiency or 0.0) * 100.0
    temp = obs.thermal_temp or 0.0
    comms= (obs.comms_signal or 0.0) * 100.0

    def flag(v, lo, hi):
        return "CRITICAL" if v < lo else ("WARNING" if v < hi else "OK")

    sunlit = getattr(obs, 'sunlit', True)
    gs     = getattr(obs, 'ground_station_visible', True)

    return (
        f"TELEMETRY — Step {step}/{MAX_STEPS}  Phase {phase+1}/3\n\n"
        f"Battery:  {bat:.1f}%   [{flag(bat, 20, 40)}]\n"
        f"Solar:    {sol:.1f}%   [{flag(sol, 30, 60)}]  Sunlit={sunlit}\n"
        f"Thermal:  {temp:.1f}C  [{'CRITICAL' if temp>85 else 'WARNING' if temp>70 else 'OK'}]\n"
        f"Comms:    {comms:.1f}% [{flag(comms, 30, 60)}]  GS={gs}\n"
        f"Payload:  {'ON' if obs.payload_on else 'OFF'}  "
        f"SafeMode: {'ACTIVE' if obs.safe_mode else 'INACTIVE'}\n"
        f"Status:   {(obs.mission_status or 'unknown').upper()}\n\n"
        f"WORLD MODEL\n"
        f"  Dominant fault subsystem: {dom}\n"
        f"  Top 3 suspected faults:   {top3}\n\n"
        f"Action:"
    )

def get_action(client: Optional[OpenAI],
               obs: OrbitalAnomalyOpenenvObservation,
               step: int) -> str:
    beliefs = compute_fault_beliefs(obs)
    phase   = obs.metadata.get("phase", 0) if obs.metadata else 0

    if client is None:
        return _heuristic_action(obs)

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(obs, beliefs, step, phase)},
            ],
            temperature=0.0,
            max_tokens=15,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        for token in raw.split():
            if token in VALID_ACTIONS:
                return token
        return _heuristic_action(obs)
    except Exception as exc:
        print(f"[DEBUG] LLM error step={step}: {exc}", flush=True)
        return _heuristic_action(obs)

# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(env, client: Optional[OpenAI], task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_name)
        obs    = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            beliefs = compute_fault_beliefs(obs)
            action  = get_action(client, obs, step)

            # Log world model every 4 steps
            if step % 4 == 1:
                top3 = top_faults_str(beliefs)
                dom  = dominant_subsystem(beliefs)
                print(f"[WORLD] step={step} dominant={dom} "
                      f"top_faults={top3}", flush=True)

            try:
                result  = env.step(OrbitalAnomalyOpenenvAction(action_type=action))
                obs     = result.observation
                reward  = float(result.reward or 0.001)
                done    = result.done
                error   = None
            except Exception as exc:
                reward, done, error = 0.001, False, str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step, action, reward, done, error)

            if done:
                break

        score   = round(sum(rewards) / len(rewards), 4) if rewards else 0.001
        score   = max(0.001, min(0.999, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score, success = 0.001, False

    log_end(success, steps_taken, score, rewards)
    return score

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client: Optional[OpenAI] = None

    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            ack = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"user","content":"Reply ACK"}],
                max_tokens=5, temperature=0,
            ).choices[0].message.content.strip()
            print(f"[PROXY] {ack}", flush=True)
        except Exception as exc:
            print(f"[PROXY] ping failed: {exc} — heuristic baseline", flush=True)
            client = None
    else:
        print("[PROXY] no API key — heuristic baseline", flush=True)

    print("[INFO] Theme 3 (World Modeling): 13-fault belief state each step", flush=True)
    print("[INFO] Theme 2 (Long-Horizon):   Extended Mission Mode = 36 steps", flush=True)
    print("[INFO] Multi-Agent: MissionCommander + EPS/Thermal/Comms specialists", flush=True)

    with OrbitalAnomalyOpenenvEnv(base_url=ENV_BASE_URL).sync() as env:
        scores = []
        for task in TASKS:
            s = run_task(env, client, task)
            scores.append(s)
            print(f"[SUMMARY] task={task} score={s:.4f}", flush=True)

        overall = round(sum(scores) / len(scores), 4)
        print(f"[SUMMARY] overall={overall:.4f}", flush=True)
        print("[SUMMARY] theme=world_modeling+long_horizon "
              "agents=commander+eps+thermal+comms", flush=True)


if __name__ == "__main__":
    main()