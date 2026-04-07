from server.orbital_anomaly_openenv_environment import (
    OrbitalAnomalyOpenenvEnvironment,
)
from models import OrbitalAnomalyOpenenvAction


seen = set()

for _ in range(6):
    env = OrbitalAnomalyOpenenvEnvironment()
    obs = env.reset()

    print(obs.task_id, obs.reward)

    seen.add(obs.task_id)

    assert 0 < obs.reward < 1

    obs = env.step(
        OrbitalAnomalyOpenenvAction(
            action_type="rotate_to_sun"
        )
    )

    assert 0 < obs.reward < 1

print("SEEN TASKS:", seen)
assert len(seen) == 3

print("✅ TASK CYCLING + REWARD RANGE PASSED")