"""
Orbital Anomaly OpenEnv V2 — Baseline Inference Script
=======================================================

Mandatory stdout format (unchanged from V1):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
  API_BASE_URL      LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME        Model id      (default: openai/gpt-4o-mini)
  HF_TOKEN          API key
  API_KEY           Fallback API key
  ENV_BASE_URL      Deployed HF Space URL
  LOCAL_IMAGE_NAME  Docker image name (if using from_docker_image)
"""

import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import OrbitalAnomalyOpenenvEnv
from models import OrbitalAnomalyOpenenvAction

# ── Environment variables ─────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
ENV_BASE_URL: str = os.getenv(
    "ENV_BASE_URL",
    "https://codequasar-orbital-anomaly-openenv.hf.space",
)
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "orbital_anomaly_openenv"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 12
SUCCESS_THRESHOLD = 0.4

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous mission-control AI for a spacecraft in distress.

    Each step you receive multi-subsystem telemetry and must choose ONE action:

      rotate_to_sun      — correct ADCS attitude to restore solar charging + comms pointing
      disable_payload    — shut down science payload to reduce thermal + power load
      reboot_comms       — reset RF chain to reduce BER, packet loss, and latency
      enter_safe_mode    — conservative hold: disables payload, cools all zones, reduces wheel stress
      switch_power_bus   — activate redundant bus: injects battery reserve, clears bus short faults
      noop               — take no action

    Sentinel values in telemetry mean the sensor is currently unavailable:
      gyro_bias = -999     → gyro telemetry dropout
      avionics_temp = -999 → avionics sensor not available this step
      uplink_margin = -99  → ground station not visible
      solar_array_current = -1 → current sensor dropout

    Priority guidance:
      1. If battery_soc < 25% and sunlit: rotate_to_sun (restore solar first)
      2. If battery_soc < 25% and eclipse: switch_power_bus (emergency reserve)
      3. If payload_temp > 70°C or avionics_temp > 65°C: disable_payload
      4. If attitude_error_deg > 50°: rotate_to_sun (pointing critical)
      5. If bit_error_rate > 0.03 or packet_loss_ratio > 0.2: reboot_comms
      6. If mission_status = critical: enter_safe_mode
      7. If observation_window_active and spacecraft stable: keep payload on (noop)

    Reply with ONLY the action name. No explanation, no punctuation.
""").strip()

# ── Valid actions ─────────────────────────────────────────────────────────────

VALID_ACTIONS = {
    "rotate_to_sun", "disable_payload", "reboot_comms",
    "enter_safe_mode", "switch_power_bus", "noop",
}

# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ── V2-aware heuristic policy ─────────────────────────────────────────────────

def _heuristic_action(obs) -> str:
    """
    Deterministic V2-aware fallback policy.
    Uses rich V2 telemetry when available, falls back to V1 fields otherwise.
    """
    # Get V2 values with V1 fallbacks
    soc       = getattr(obs, "battery_soc",      obs.battery_level)
    att_err   = getattr(obs, "attitude_error_deg", 90.0 if obs.solar_efficiency < 0.5 else 10.0)
    plr       = getattr(obs, "packet_loss_ratio",  1.0 - obs.comms_signal)
    ber       = getattr(obs, "bit_error_rate",     0.05 if obs.comms_signal < 0.6 else 0.005)
    p_temp    = getattr(obs, "payload_temp",       obs.thermal_temp)
    av_temp   = getattr(obs, "avionics_temp",      obs.thermal_temp)
    sunlit    = getattr(obs, "sunlit",             True)
    obs_win   = getattr(obs, "observation_window_active", False)
    wheel_sat = getattr(obs, "wheel_saturation_level", 0.3)
    bat_temp  = getattr(obs, "battery_temp",       15.0)

    # Sentinel: avionics dropout — use thermal_temp as proxy
    if av_temp == -999.0:
        av_temp = obs.thermal_temp

    # Critical battery — different action in eclipse vs sunlit
    if soc < 20.0:
        return "switch_power_bus" if not sunlit else "rotate_to_sun"

    # Severe ADCS misalignment — limits solar charging
    if att_err > 50.0:
        return "rotate_to_sun"

    # Thermal runaway in payload or avionics
    if p_temp > 70.0 or av_temp > 65.0:
        return "disable_payload"

    # Comms chain degraded
    if ber > 0.03 or plr > 0.20:
        return "reboot_comms"

    # Low battery in sunlit — align solar panels
    if soc < 40.0 and sunlit:
        return "rotate_to_sun"

    # Low battery without solar — emergency bus
    if soc < 40.0 and not sunlit:
        return "switch_power_bus"

    # Cold battery in eclipse — heater needed
    if bat_temp < 2.0 and not sunlit:
        return "enter_safe_mode"

    # Moderate attitude drift
    if att_err > 30.0:
        return "rotate_to_sun"

    # Mission critical
    if obs.mission_status == "critical":
        return "enter_safe_mode"

    # Science opportunity — keep payload running if stable
    if obs_win and obs.payload_on and obs.mission_status == "stable":
        return "noop"

    # Near wheel saturation
    if wheel_sat > 0.7:
        return "rotate_to_sun"

    return "noop"


def _build_user_prompt(obs) -> str:
    soc    = getattr(obs, "battery_soc",             obs.battery_level)
    volt   = getattr(obs, "bus_voltage",             28.0)
    p_h    = getattr(obs, "panel_health",            obs.solar_efficiency)
    sac    = getattr(obs, "solar_array_current",     -1.0)
    cch    = getattr(obs, "charge_controller_health", 1.0)
    att    = getattr(obs, "attitude_error_deg",       0.0)
    sva    = getattr(obs, "sun_vector_alignment",     obs.solar_efficiency)
    rw_m   = getattr(obs, "reaction_wheel_momentum",  0.1)
    gb     = getattr(obs, "gyro_bias",               0.0)
    sta    = getattr(obs, "star_tracker_available",  True)
    wsl    = getattr(obs, "wheel_saturation_level",  0.1)
    bt     = getattr(obs, "battery_temp",            15.0)
    pt     = getattr(obs, "payload_temp",            obs.thermal_temp)
    avt    = getattr(obs, "avionics_temp",           obs.thermal_temp)
    rad_e  = getattr(obs, "radiator_efficiency",     1.0)
    tlh    = getattr(obs, "thermal_loop_health",     1.0)
    hs     = getattr(obs, "heater_state",            False)
    ape    = getattr(obs, "antenna_pointing_error",  3.0)
    tp     = getattr(obs, "transmitter_power",       5.0)
    ber    = getattr(obs, "bit_error_rate",          0.001)
    ulm    = getattr(obs, "uplink_margin",           12.0)
    plr    = getattr(obs, "packet_loss_ratio",       0.02)
    lat    = getattr(obs, "command_latency_ms",      120.0)
    sunlit = getattr(obs, "sunlit",                  True)
    ecl    = getattr(obs, "eclipse_timer",           0)
    gs     = getattr(obs, "ground_station_visible",  True)
    rad_z  = getattr(obs, "radiation_zone",          False)
    obsw   = getattr(obs, "observation_window_active", False)

    return textwrap.dedent(f"""
        ── EPS ─────────────────────────────────────────
        battery_soc              : {soc:.1f} %
        bus_voltage              : {volt:.2f} V  (nominal 28V)
        panel_health             : {p_h:.3f}
        solar_array_current      : {sac:.2f} A  (-1 = dropout)
        charge_controller_health : {cch:.3f}

        ── ADCS ────────────────────────────────────────
        attitude_error_deg       : {att:.1f} °
        sun_vector_alignment     : {sva:.4f}
        reaction_wheel_momentum  : {rw_m:.3f}
        wheel_saturation_level   : {wsl:.3f}
        gyro_bias                : {gb:.2f}  (-999 = dropout)
        star_tracker_available   : {sta}

        ── Thermal ─────────────────────────────────────
        battery_temp             : {bt:.1f} °C  (safe: -5 to 35)
        payload_temp             : {pt:.1f} °C  (safe: <75)
        avionics_temp            : {avt:.1f} °C  (safe: <70; -999 = dropout)
        radiator_efficiency      : {rad_e:.3f}
        thermal_loop_health      : {tlh:.3f}
        heater_state             : {hs}

        ── Communications ──────────────────────────────
        antenna_pointing_error   : {ape:.1f} °
        transmitter_power        : {tp:.2f} W
        bit_error_rate           : {ber:.5f}  (good: <0.01)
        uplink_margin            : {ulm:.1f} dB  (-99 = no GS)
        packet_loss_ratio        : {plr:.4f}  (good: <0.05)
        command_latency_ms       : {lat:.0f} ms  (-1 = dropout)

        ── Orbital Context ─────────────────────────────
        sunlit                   : {sunlit}
        eclipse_timer            : {ecl} steps
        ground_station_visible   : {gs}
        radiation_zone           : {rad_z}
        observation_window_active: {obsw}

        ── Mission ─────────────────────────────────────
        task_id                  : {obs.task_id}
        mission_status           : {obs.mission_status}
        payload_on               : {obs.payload_on}
        safe_mode                : {obs.safe_mode}

        Choose one action: rotate_to_sun | disable_payload | reboot_comms | enter_safe_mode | switch_power_bus | noop
    """).strip()


def get_action(client: Optional[OpenAI], obs, step: int) -> str:
    if client is None:
        return _heuristic_action(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(obs)},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        raw = (completion.choices[0].message.content or "").strip().lower()
        for token in raw.split():
            if token in VALID_ACTIONS:
                return token
        return _heuristic_action(obs)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return _heuristic_action(obs)


# ── Per-task episode runner ───────────────────────────────────────────────────

def run_task(env, client: Optional[OpenAI], task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_name)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_name = get_action(client, obs, step)

            try:
                result = env.step(OrbitalAnomalyOpenenvAction(action_type=action_name))
                obs    = result.observation
                reward = float(result.reward or 0.001)
                done   = result.done
                error  = None
            except Exception as exc:
                reward = 0.001
                done   = False
                error  = str(exc)
                print(f"[DEBUG] step error: {exc}", flush=True)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_name, reward=reward, done=done, error=error)

            if done:
                break

        score = round(sum(rewards) / len(rewards), 4) if rewards else 0.001
        score = max(0.001, min(0.999, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = 0.001
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client: Optional[OpenAI] = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            ping = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Reply with only ACK"}],
                max_tokens=5,
                temperature=0,
            )
            ack = (ping.choices[0].message.content or "ACK").strip()
            print(f"[PROXY] {ack}", flush=True)
        except Exception as exc:
            print(f"[PROXY] ping failed: {exc} — running heuristic baseline", flush=True)
            client = None
    else:
        print("[PROXY] no API key found — running heuristic baseline", flush=True)

    with OrbitalAnomalyOpenenvEnv(base_url=ENV_BASE_URL).sync() as env:
        all_scores: List[float] = []
        for task_name in TASKS:
            score = run_task(env, client, task_name)
            all_scores.append(score)
            print(f"[SUMMARY] task={task_name} score={score:.2f}", flush=True)

        overall = round(sum(all_scores) / len(all_scores), 4)
        print(f"[SUMMARY] overall_score={overall:.2f}", flush=True)


if __name__ == "__main__":
    main()