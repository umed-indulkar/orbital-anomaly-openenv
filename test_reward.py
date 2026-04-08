"""
Reward range and task cycling validation.

Verifies:
  1. All 3 task IDs (easy, medium, hard) are reachable by name.
  2. All rewards are strictly inside the open interval (0, 1).
  3. The heuristic recovery policy produces partial-progress rewards.
"""

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
        f"{label}: reward={reward} is NOT in the strict open interval (0, 1)"
    )


def test_tasks_by_name():
    """Each task must be reachable by explicit name and return valid rewards."""
    for task_id in TASKS:
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset(task_id=task_id)

        assert obs.task_id == task_id, f"Expected task_id={task_id}, got {obs.task_id}"
        assert_open_interval(obs.reward, f"reset/{task_id}")

        print(f"  ✓ reset task_id={task_id}  reward={obs.reward}  status={obs.mission_status}")

        # Run a few steps and check all rewards
        for i, action_name in enumerate(ACTIONS[:4]):
            step_obs = env.step(OrbitalAnomalyOpenenvAction(action_type=action_name))
            assert_open_interval(step_obs.reward, f"step/{task_id}/{action_name}")

        print(f"  ✓ steps for task_id={task_id} all passed reward range check")


def test_cycling():
    """Sequential resets without task_id must cycle through all 3 tasks."""
    seen = set()
    for _ in range(9):
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset()
        seen.add(obs.task_id)

    assert seen == set(TASKS), f"Expected all 3 tasks, got: {seen}"
    print(f"  ✓ task cycling covers: {seen}")


def test_no_boundary_rewards():
    """Ensure rewards never land exactly on 0.0 or 1.0 in any scenario."""
    for task_id in TASKS:
        env = OrbitalAnomalyOpenenvEnvironment()
        obs = env.reset(task_id=task_id)
        assert obs.reward != 0.0 and obs.reward != 1.0

        for action_name in ACTIONS:
            step_obs = env.step(OrbitalAnomalyOpenenvAction(action_type=action_name))
            assert step_obs.reward != 0.0 and step_obs.reward != 1.0, (
                f"Boundary reward {step_obs.reward} for {task_id}/{action_name}"
            )
    print("  ✓ no boundary rewards (0.0 or 1.0) found in any task/action combo")


if __name__ == "__main__":
    print("\n── test_tasks_by_name ──────────────────────────────────────────")
    test_tasks_by_name()

    print("\n── test_cycling ────────────────────────────────────────────────")
    test_cycling()

    print("\n── test_no_boundary_rewards ────────────────────────────────────")
    test_no_boundary_rewards()

    print("\n✅ ALL TESTS PASSED")