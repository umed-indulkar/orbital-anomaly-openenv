from server.orbital_anomaly_openenv_environment import (
    OrbitalAnomalyOpenenvEnvironment
)
from models import OrbitalAnomalyOpenenvAction

env = OrbitalAnomalyOpenenvEnvironment()

action_cycle = [
    "rotate_to_sun",
    "disable_payload",
    "reboot_comms",
    "switch_power_bus",
]

for i in range(3):
    obs = env.reset()
    print(f"\nTASK {i+1}: {obs.task_id}")
    print("reset reward:", obs.reward)

    assert 0 < obs.reward < 1, "Reset reward out of range"

    for step in range(12):
        action = action_cycle[step % len(action_cycle)]

        obs = env.step(
            OrbitalAnomalyOpenenvAction(action_type=action)
        )

        print(
            f"step={step+1} "
            f"action={action} "
            f"reward={obs.reward}"
        )

        assert 0 < obs.reward < 1, "Step reward out of range"

print("\n✅ ALL TASK REWARDS STRICTLY IN (0,1)")