"""
Orbital Anomaly OpenEnv — Baseline Inference Script
=====================================================

Mandatory stdout format
-----------------------
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules enforced here:
- One [START] per episode.
- One [STEP] per env.step() call, immediately after it returns.
- One [END] after the episode closes (always emitted, even on exception).
- reward / rewards formatted to 2 decimal places.
- done / success are lowercase booleans.
- error is the raw error string or "null".
- score is in [0, 1] and formatted to 2 decimal places.

Environment variables
---------------------
API_BASE_URL      LLM endpoint  (default: https://router.huggingface.co/v1)
MODEL_NAME        Model id      (default: openai/gpt-4o-mini)
HF_TOKEN          API key
API_KEY           Fallback API key
ENV_BASE_URL      Deployed HF Space URL
LOCAL_IMAGE_NAME  Docker image name (if using from_docker_image)
"""

import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import OrbitalAnomalyOpenenvEnv
from models import OrbitalAnomalyOpenenvAction

# ── Environment variables ─────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv(
    "API_BASE_URL",
    "https://router.huggingface.co/v1",
)
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
SUCCESS_THRESHOLD = 0.4  # score in [0,1] required for success=true

# ── System prompt for the LLM agent ──────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous mission-control AI managing a spacecraft in distress.
    Each step you receive spacecraft telemetry and must choose exactly ONE
    recovery action from this list:

      rotate_to_sun      — realign solar panels to restore charging
      disable_payload    — shut down science payload to reduce thermal/power load
      reboot_comms       — restart communication subsystem to restore signal
      enter_safe_mode    — enable safe mode (disables payload, stabilises all systems)
      switch_power_bus   — switch to backup power bus for emergency battery boost
      noop               — take no action this step

    Reply with ONLY the action name — no explanation, no punctuation, nothing else.
    Choose the action that best addresses the most critical anomaly in the telemetry.
""").strip()


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM helpers ───────────────────────────────────────────────────────────────

VALID_ACTIONS = {
    "rotate_to_sun",
    "disable_payload",
    "reboot_comms",
    "enter_safe_mode",
    "switch_power_bus",
    "noop",
}


def _build_user_prompt(obs) -> str:
    return textwrap.dedent(f"""
        Spacecraft telemetry:
          battery_level    : {obs.battery_level:.1f} %
          solar_efficiency : {obs.solar_efficiency:.2f}
          thermal_temp     : {obs.thermal_temp:.1f} °C
          comms_signal     : {obs.comms_signal:.2f}
          payload_on       : {obs.payload_on}
          safe_mode        : {obs.safe_mode}
          mission_status   : {obs.mission_status}
          task_id          : {obs.task_id}

        Choose one action:
        rotate_to_sun | disable_payload | reboot_comms |
        enter_safe_mode | switch_power_bus | noop
    """).strip()


def _heuristic_action(obs) -> str:
    """Deterministic fallback policy used when no API key is available."""
    if obs.solar_efficiency < 0.60:
        return "rotate_to_sun"
    if obs.thermal_temp > 80 and obs.payload_on:
        return "disable_payload"
    if obs.comms_signal < 0.75:
        return "reboot_comms"
    if obs.battery_level < 45:
        return "switch_power_bus"
    if obs.mission_status == "critical":
        return "enter_safe_mode"
    return "noop"


def get_action(client: Optional[OpenAI], obs, step: int) -> str:
    """Ask the LLM for an action, falling back to the heuristic policy."""
    if client is None:
        return _heuristic_action(obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(obs)},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        raw = (completion.choices[0].message.content or "").strip().lower()
        # Accept the first token that matches a valid action
        for token in raw.split():
            if token in VALID_ACTIONS:
                return token
        # If nothing matched, fall back
        return _heuristic_action(obs)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return _heuristic_action(obs)


# ── Per-task episode runner ───────────────────────────────────────────────────

def run_task(
    env,
    client: Optional[OpenAI],
    task_name: str,
) -> float:
    """
    Run one full episode for the given task and emit mandatory log lines.
    Returns the normalised score in [0, 1].
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset into the specific task
        result = env.reset(task_id=task_name)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_name = get_action(client, obs, step)

            try:
                result = env.step(OrbitalAnomalyOpenenvAction(action_type=action_name))
                obs = result.observation
                reward = float(result.reward or 0.001)
                done = result.done
                error = None
            except Exception as exc:
                reward = 0.001
                done = False
                error = str(exc)
                print(f"[DEBUG] step error: {exc}", flush=True)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_name, reward=reward, done=done, error=error)

            if done:
                break

        # Score = mean reward over all steps (already in (0,1))
        score = round(sum(rewards) / len(rewards), 4) if rewards else 0.001
        score = max(0.001, min(0.999, score))  # strict open interval
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = 0.001
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Build OpenAI client (required by spec — use OpenAI client for all LLM calls)
    client: Optional[OpenAI] = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            # Mandatory proxy ping — Phase 2 checks that the provided key was used
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

    # Connect to the deployed environment
    with OrbitalAnomalyOpenenvEnv(base_url=ENV_BASE_URL).sync() as env:
        all_scores: List[float] = []
        for task_name in TASKS:
            score = run_task(env, client, task_name)
            all_scores.append(score)
            print(
                f"[SUMMARY] task={task_name} score={score:.2f}",
                flush=True,
            )

        overall = round(sum(all_scores) / len(all_scores), 4)
        print(f"[SUMMARY] overall_score={overall:.2f}", flush=True)


if __name__ == "__main__":
    main()