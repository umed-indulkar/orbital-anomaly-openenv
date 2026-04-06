import os
from openai import OpenAI

from .client import OrbitalAnomalyOpenenvEnv
from .models import OrbitalAnomalyOpenenvAction


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://codequasar-orbital-anomaly-openenv.hf.space"
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


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
    print("[START] orbital anomaly recovery baseline")

    with OrbitalAnomalyOpenenvEnv(base_url=API_BASE_URL).sync() as env:
        result = env.reset()
        obs = result.observation

        for step in range(12):
            action_name = choose_action(obs)

            result = env.step(
                OrbitalAnomalyOpenenvAction(action_type=action_name)
            )
            obs = result.observation

            print(
                f"[STEP] step={step+1} "
                f"action={action_name} "
                f"reward={result.reward:.3f}"
            )

            if result.done:
                break

        print(f"[END] final_reward={result.reward:.3f}")


if __name__ == "__main__":
    main()