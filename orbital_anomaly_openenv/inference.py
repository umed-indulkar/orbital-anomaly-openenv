from .client import OrbitalAnomalyOpenenvEnv
from .models import OrbitalAnomalyOpenenvAction


def choose_action(obs):
    if obs.solar_efficiency < 0.5:
        return "rotate_to_sun"

    if obs.thermal_temp > 80:
        return "disable_payload"

    if obs.comms_signal < 0.7:
        return "reboot_comms"

    if obs.mission_status == "critical":
        return "enter_safe_mode"

    return "noop"


def main():
    with OrbitalAnomalyOpenenvEnv(
        base_url="http://localhost:8000"
    ).sync() as env:
        result = env.reset()
        obs = result.observation

        print("\n🛰️ STARTING ORBITAL ANOMALY RESPONSE")
        print("Initial:", obs)

        for step in range(12):
            action_name = choose_action(obs)

            result = env.step(
                OrbitalAnomalyOpenenvAction(action_type=action_name)
            )
            obs = result.observation

            print(f"\nSTEP {step + 1}")
            print("Action:", action_name)
            print("Battery:", obs.battery_level)
            print("Thermal:", obs.thermal_temp)
            print("Comms:", obs.comms_signal)
            print("Reward:", result.reward)

            if result.done:
                print("\n✅ Mission stabilized")
                break

        print("\n🏁 Final reward:", result.reward)


if __name__ == "__main__":
    main()