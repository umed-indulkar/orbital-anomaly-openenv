"""
V2 Reward range, subsystem coupling, and task validation tests.

Verifies:
  1. All 3 tasks reachable by name with strict (0, 1) rewards.
  2. V2 observation fields present and within physical bounds.
  3. Physics coupling: actions affect the right subsystems.
  4. Hard task: eclipse causes SOC drain without intervention.
  5. Science window: noop and safe_mode both valid in (0,1).
  6. Task cycling covers all 3 tasks across sequential resets.
  7. No boundary rewards (0.0 or 1.0) in any scenario.
  8. Partial observability: hard task dropout sentinels correct.
  9. Explicit task_id independent of global counter state.
"""

from server.orbital_anomaly_openenv_environment import (
    OrbitalAnomalyOpenenvEnvironment,
)
from models import OrbitalAnomalyOpenenvAction

TASKS   = ["easy", "medium", "hard"]
ACTIONS = [
    "rotate_to_sun", "disable_payload", "reboot_comms",
    "enter_safe_mode", "switch_power_bus", "noop",
]


def assert_open_interval(reward: float, label: str) -> None:
    assert 0.0 < reward < 1.0, (
        f"{label}: reward={reward:.6f} is NOT in strict open interval (0, 1)"
    )


def _reset_counter(value: int = 0) -> None:
    OrbitalAnomalyOpenenvEnvironment._global_reset_count = value


# ── Test 1: Tasks by name ─────────────────────────────────────────────────────

def test_tasks_by_name():
    """Explicit reset(task_id=X) must return exactly task X regardless of counter."""
    for task_id in TASKS:
        _reset_counter(99)   # offset should have zero effect
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset(task_id=task_id)

        assert obs.task_id == task_id, (
            f"Expected task_id='{task_id}', got '{obs.task_id}'. "
            "Explicit task_id must be independent of _global_reset_count."
        )
        assert_open_interval(obs.reward, f"reset/{task_id}")

        # V2 physical bounds
        assert  0.0  <= obs.battery_soc           <= 100.0
        assert 18.0  <= obs.bus_voltage            <= 29.0
        assert  0.0  <= obs.panel_health           <= 1.0
        assert  0.0  <= obs.attitude_error_deg     <= 90.0
        assert  0.0  <= obs.wheel_saturation_level <= 1.0
        assert -40.0 <= obs.battery_temp           <= 60.0
        assert -40.0 <= obs.payload_temp           <= 120.0
        assert  0.0  <= obs.bit_error_rate         <= 1.0
        assert  0.0  <= obs.packet_loss_ratio      <= 1.0

        print(f"  ✓ reset/{task_id}  reward={obs.reward}  soc={obs.battery_soc:.1f}%  "
              f"att={obs.attitude_error_deg:.1f}°  payload_temp={obs.payload_temp:.1f}°C")

        # Step through every action from this task start
        for action_name in ACTIONS:
            env2 = OrbitalAnomalyOpenenvEnvironment()
            env2.reset(task_id=task_id)
            step_obs = env2.step(OrbitalAnomalyOpenenvAction(action_type=action_name))
            assert_open_interval(step_obs.reward, f"step/{task_id}/{action_name}")

        print(f"  ✓ all 6 actions for {task_id} passed reward range check")


# ── Test 2: Physics coupling ──────────────────────────────────────────────────

def test_physics_coupling():
    # rotate_to_sun must reduce attitude error
    _reset_counter(0)
    env = OrbitalAnomalyOpenenvEnvironment()
    obs = env.reset(task_id="easy")
    initial_att = obs.attitude_error_deg
    step_obs = env.step(OrbitalAnomalyOpenenvAction(action_type="rotate_to_sun"))
    assert step_obs.attitude_error_deg < initial_att, (
        f"rotate_to_sun had no effect: {initial_att:.1f}° → {step_obs.attitude_error_deg:.1f}°"
    )
    print(f"  ✓ rotate_to_sun: att_err {initial_att:.1f}° → {step_obs.attitude_error_deg:.1f}°")

    # disable_payload must reduce payload temperature
    env2 = OrbitalAnomalyOpenenvEnvironment()
    obs2 = env2.reset(task_id="medium")
    initial_temp = obs2.payload_temp
    step_obs2 = env2.step(OrbitalAnomalyOpenenvAction(action_type="disable_payload"))
    assert step_obs2.payload_temp < initial_temp, (
        f"disable_payload no effect: {initial_temp:.1f}°C → {step_obs2.payload_temp:.1f}°C"
    )
    print(f"  ✓ disable_payload: payload_temp {initial_temp:.1f}°C → {step_obs2.payload_temp:.1f}°C")


# ── Test 3: Hard task eclipse drain ──────────────────────────────────────────

def test_hard_fault_cascade():
    _reset_counter(0)
    env = OrbitalAnomalyOpenenvEnvironment()
    obs = env.reset(task_id="hard")
    assert obs.task_id == "hard"
    assert not obs.sunlit,               "Hard task must start in eclipse"
    assert not obs.ground_station_visible, "Hard task must start in GS blackout"
    assert obs.radiation_zone,           "Hard task must start in radiation zone"

    initial_soc = obs.battery_soc
    for _ in range(3):
        step_obs = env.step(OrbitalAnomalyOpenenvAction(action_type="noop"))

    assert step_obs.battery_soc < initial_soc, (
        f"SOC should drain: {initial_soc:.1f}% → {step_obs.battery_soc:.1f}%"
    )
    print(f"  ✓ hard eclipse drain: soc {initial_soc:.1f}% → {step_obs.battery_soc:.1f}%")


# ── Test 4: Science window reward ────────────────────────────────────────────

def test_science_window_reward():
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
    print(f"  ✓ science window: noop={r_noop:.4f}  safe_mode={r_safe:.4f}")


# ── Test 5: Task cycling ──────────────────────────────────────────────────────

def test_cycling():
    """
    Sequential reset() calls WITHOUT task_id on the SAME env instance
    must cycle through all 3 tasks. We test on a single env to avoid
    any per-instance initialisation issues.
    """
    _reset_counter(0)
    env = OrbitalAnomalyOpenenvEnvironment()
    seen = set()

    # 9 resets → 3 full cycles → must have seen all 3 tasks
    for _ in range(9):
        obs = env.reset()   # no task_id — cycling path
        seen.add(obs.task_id)

    assert seen == set(TASKS), (
        f"Expected all 3 tasks in 9 cycling resets, got: {seen}"
    )
    print(f"  ✓ task cycling covers: {seen}")


# ── Test 6: No boundary rewards ──────────────────────────────────────────────

def test_no_boundary_rewards():
    for task_id in TASKS:
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset(task_id=task_id)
        assert obs.reward not in (0.0, 1.0), (
            f"Boundary reward {obs.reward} on reset/{task_id}"
        )
        for action_name in ACTIONS:
            env2 = OrbitalAnomalyOpenenvEnvironment()
            env2.reset(task_id=task_id)
            s = env2.step(OrbitalAnomalyOpenenvAction(action_type=action_name))
            assert s.reward not in (0.0, 1.0), (
                f"Boundary reward {s.reward} for {task_id}/{action_name}"
            )
    print("  ✓ no boundary rewards (0.0 or 1.0) across all tasks and actions")


# ── Test 7: Partial observability ────────────────────────────────────────────

def test_partial_observability():
    _reset_counter(0)
    env = OrbitalAnomalyOpenenvEnvironment()
    obs = env.reset(task_id="hard")

    assert obs.star_tracker_available is None, (
        f"Hard task: star_tracker_available should be None (dropout), "
        f"got {obs.star_tracker_available}"
    )
    assert obs.uplink_margin == -99.0, (
        f"Hard task: uplink_margin should be -99.0 (GS blackout), "
        f"got {obs.uplink_margin}"
    )
    print(f"  ✓ partial obs: star_tracker=None  uplink_margin=-99.0")


# ── Test 8: Counter independence ─────────────────────────────────────────────

def test_counter_independence():
    """Explicit task_id always wins, regardless of counter value."""
    for offset in [0, 1, 2, 7, 100, 999]:
        for task_id in TASKS:
            _reset_counter(offset)
            env = OrbitalAnomalyOpenenvEnvironment()
            obs = env.reset(task_id=task_id)
            assert obs.task_id == task_id, (
                f"counter={offset}: expected '{task_id}', got '{obs.task_id}'"
            )
    print("  ✓ explicit task_id fully independent of _global_reset_count")


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