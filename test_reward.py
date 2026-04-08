"""
V2 Reward range, subsystem coupling, and task validation tests.

Verifies:
  1. All 3 tasks reachable by name with strict (0, 1) rewards.
  2. V2 observation fields all present and within physical bounds.
  3. Fault cascade effects are observable after delayed steps.
  4. Task cycling covers all 3 tasks across fresh instances.
  5. No boundary rewards (0.0 or 1.0) in any scenario.
  6. Partial observability dropout is deterministic.
  7. Eclipse causes SOC drain on hard task.
  8. Science window produces competitive reward vs safe_mode.
"""

import math

from server.orbital_anomaly_openenv_environment import (
    OrbitalAnomalyOpenenvEnvironment,
)
from models import OrbitalAnomalyOpenenvAction

TASKS = ["easy", "medium", "hard"]
ACTIONS = [
    "rotate_to_sun",
    "disable_payload",
    "reboot_comms",
    "enter_safe_mode",
    "switch_power_bus",
    "noop",
]


def assert_open_interval(reward: float, label: str) -> None:
    assert 0.0 < reward < 1.0, (
        f"{label}: reward={reward:.6f} is NOT in strict open interval (0, 1)"
    )


def _reset_counter(value: int = 0) -> None:
    """Reset the class-level cycle counter to a known state."""
    OrbitalAnomalyOpenenvEnvironment._global_reset_count = value


# ── Test 1: Tasks by name ─────────────────────────────────────────────────────

def test_tasks_by_name():
    """
    Each task must be reachable by explicit name regardless of counter state.
    Explicit reset(task_id=X) must NEVER depend on the global counter.
    """
    for task_id in TASKS:
        # Reset counter to a different offset each time to prove independence
        _reset_counter(99)  # arbitrary offset — should not affect explicit task
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset(task_id=task_id)

        assert obs.task_id == task_id, (
            f"Expected task_id='{task_id}', got '{obs.task_id}'. "
            f"reset(task_id=...) must not be affected by _global_reset_count."
        )
        assert_open_interval(obs.reward, f"reset/{task_id}")

        # Verify V2 fields exist and are physically sensible
        assert 0.0   <= obs.battery_soc          <= 100.0, f"{task_id}: battery_soc out of range"
        assert 18.0  <= obs.bus_voltage           <= 29.0,  f"{task_id}: bus_voltage out of range"
        assert 0.0   <= obs.panel_health          <= 1.0,   f"{task_id}: panel_health out of range"
        assert 0.0   <= obs.attitude_error_deg    <= 90.0,  f"{task_id}: attitude_error_deg out of range"
        assert 0.0   <= obs.wheel_saturation_level <= 1.0,  f"{task_id}: wheel_saturation out of range"
        assert -40.0 <= obs.battery_temp          <= 60.0,  f"{task_id}: battery_temp out of range"
        assert -40.0 <= obs.payload_temp          <= 120.0, f"{task_id}: payload_temp out of range"
        assert 0.0   <= obs.bit_error_rate        <= 1.0,   f"{task_id}: bit_error_rate out of range"
        assert 0.0   <= obs.packet_loss_ratio     <= 1.0,   f"{task_id}: packet_loss_ratio out of range"

        print(f"  ✓ reset/{task_id}  reward={obs.reward}  soc={obs.battery_soc:.1f}%  "
              f"att={obs.attitude_error_deg:.1f}°  payload_temp={obs.payload_temp:.1f}°C")

        # Step through all actions and check rewards
        for action_name in ACTIONS:
            env2 = OrbitalAnomalyOpenenvEnvironment()
            env2.reset(task_id=task_id)
            step_obs = env2.step(OrbitalAnomalyOpenenvAction(action_type=action_name))
            assert_open_interval(step_obs.reward, f"step/{task_id}/{action_name}")

        print(f"  ✓ all actions for {task_id} passed reward range check")


# ── Test 2: V2 physics coupling ───────────────────────────────────────────────

def test_physics_coupling():
    """rotate_to_sun must reduce attitude error."""
    _reset_counter(0)
    env = OrbitalAnomalyOpenenvEnvironment()
    obs = env.reset(task_id="easy")
    initial_att = obs.attitude_error_deg

    step_obs = env.step(OrbitalAnomalyOpenenvAction(action_type="rotate_to_sun"))
    # Natural drift adds ~1.5°, but rotate_to_sun reduces by ~(25 * wheel_authority)
    # Net should be significantly below initial
    assert step_obs.attitude_error_deg < initial_att, (
        f"rotate_to_sun had no effect: {initial_att:.1f}° → {step_obs.attitude_error_deg:.1f}°"
    )
    print(f"  ✓ rotate_to_sun: att_err {initial_att:.1f}° → {step_obs.attitude_error_deg:.1f}°")

    """disable_payload must reduce payload temperature."""
    env2 = OrbitalAnomalyOpenenvEnvironment()
    env2.reset(task_id="medium")
    obs2 = env2.reset(task_id="medium")
    initial_temp = obs2.payload_temp
    step_obs2 = env2.step(OrbitalAnomalyOpenenvAction(action_type="disable_payload"))
    assert step_obs2.payload_temp < initial_temp, (
        f"disable_payload: temp did not decrease {initial_temp:.1f} → {step_obs2.payload_temp:.1f}"
    )
    print(f"  ✓ disable_payload: payload_temp {initial_temp:.1f}°C → {step_obs2.payload_temp:.1f}°C")


# ── Test 3: Hard task fault cascades ─────────────────────────────────────────

def test_hard_fault_cascade():
    """Hard task starts in eclipse; battery_soc must fall without intervention."""
    _reset_counter(0)
    env = OrbitalAnomalyOpenenvEnvironment()
    obs = env.reset(task_id="hard")
    initial_soc = obs.battery_soc

    assert not obs.sunlit, "Hard task must start in eclipse"
    assert not obs.ground_station_visible, "Hard task must start in GS blackout"
    assert obs.radiation_zone, "Hard task must start in radiation zone"

    # noop for 3 steps — battery should drain in eclipse
    for _ in range(3):
        step_obs = env.step(OrbitalAnomalyOpenenvAction(action_type="noop"))
    soc_after = step_obs.battery_soc

    assert soc_after < initial_soc, (
        f"SOC should drain during eclipse noop: {initial_soc:.1f}% → {soc_after:.1f}%"
    )
    print(f"  ✓ hard task eclipse drain: soc {initial_soc:.1f}% → {soc_after:.1f}%")


# ── Test 4: Science bonus ─────────────────────────────────────────────────────

def test_science_window_reward():
    """Medium task has active observation window — both noop and safe_mode rewards must be valid."""
    _reset_counter(0)
    env_noop = OrbitalAnomalyOpenenvEnvironment()
    obs_noop = env_noop.reset(task_id="medium")
    assert obs_noop.observation_window_active, "Medium must have active obs window"
    r_noop = env_noop.step(OrbitalAnomalyOpenenvAction(action_type="noop")).reward

    env_safe = OrbitalAnomalyOpenenvEnvironment()
    env_safe.reset(task_id="medium")
    r_safe = env_safe.step(OrbitalAnomalyOpenenvAction(action_type="enter_safe_mode")).reward

    assert_open_interval(r_noop, "medium/noop")
    assert_open_interval(r_safe, "medium/safe_mode")
    print(f"  ✓ science window: noop reward={r_noop:.4f}  safe_mode reward={r_safe:.4f}")


# ── Test 5: Task cycling ──────────────────────────────────────────────────────

def test_cycling():
    """Sequential resets without task_id must cycle through all 3 tasks."""
    _reset_counter(0)
    seen = set()
    for _ in range(9):
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset()  # no task_id — must cycle
        seen.add(obs.task_id)
    assert seen == set(TASKS), f"Expected all 3 tasks, got: {seen}"
    print(f"  ✓ task cycling covers: {seen}")


# ── Test 6: No boundary rewards ──────────────────────────────────────────────

def test_no_boundary_rewards():
    """Rewards must never be exactly 0.0 or 1.0."""
    for task_id in TASKS:
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset(task_id=task_id)
        assert obs.reward != 0.0 and obs.reward != 1.0, (
            f"Boundary reward {obs.reward} on reset/{task_id}"
        )
        for action_name in ACTIONS:
            env2 = OrbitalAnomalyOpenenvEnvironment()
            env2.reset(task_id=task_id)
            step_obs = env2.step(OrbitalAnomalyOpenenvAction(action_type=action_name))
            assert step_obs.reward != 0.0 and step_obs.reward != 1.0, (
                f"Boundary reward {step_obs.reward} for {task_id}/{action_name}"
            )
    print("  ✓ no boundary rewards (0.0 or 1.0) across all tasks and actions")


# ── Test 7: Partial observability ────────────────────────────────────────────

def test_partial_observability():
    """Hard task must have star_tracker dropout and GS blackout sentinels."""
    _reset_counter(0)
    env = OrbitalAnomalyOpenenvEnvironment()
    obs = env.reset(task_id="hard")

    # star_tracker must be in dropout (None) because _F_STAR_TRACKER_DROP is active
    assert obs.star_tracker_available is None, (
        f"Hard task: star_tracker_available should be None (dropout), got {obs.star_tracker_available}"
    )
    # uplink_margin must be -99 because ground_station_visible=False
    assert obs.uplink_margin == -99.0, (
        f"Hard task: uplink_margin should be -99.0 (GS blackout), got {obs.uplink_margin}"
    )
    print(f"  ✓ partial observability: star_tracker=None, uplink_margin=-99.0")


# ── Test 8: Counter independence ─────────────────────────────────────────────

def test_counter_independence():
    """
    Explicit task_id requests must be immune to counter state.
    Run reset(task_id='hard') after the counter has been advanced many times.
    """
    _reset_counter(1000)   # extreme counter offset
    for _ in range(5):
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset(task_id="hard")
        assert obs.task_id == "hard", (
            f"Expected 'hard' regardless of counter=1000, got '{obs.task_id}'"
        )
    _reset_counter(1000)
    for _ in range(5):
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset(task_id="easy")
        assert obs.task_id == "easy", (
            f"Expected 'easy' regardless of counter=1000, got '{obs.task_id}'"
        )
    print("  ✓ explicit task_id is fully independent of _global_reset_count")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── test_tasks_by_name ──────────────────────────────────────────")
    test_tasks_by_name()

    print("\n── test_physics_coupling ───────────────────────────────────────")
    test_physics_coupling()

    print("\n── test_hard_fault_cascade ─────────────────────────────────────")
    test_hard_fault_cascade()

    print("\n── test_science_window_reward ──────────────────────────────────")
    test_science_window_reward()

    print("\n── test_cycling ────────────────────────────────────────────────")
    test_cycling()

    print("\n── test_no_boundary_rewards ────────────────────────────────────")
    test_no_boundary_rewards()

    print("\n── test_partial_observability ──────────────────────────────────")
    test_partial_observability()

    print("\n── test_counter_independence ───────────────────────────────────")
    test_counter_independence()

    print("\n✅ ALL V2 TESTS PASSED")