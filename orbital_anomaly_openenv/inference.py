import os
from openai import OpenAI

from .client import OrbitalAnomalyOpenenvEnv
from .models import OrbitalAnomalyOpenenvAction


# Required environment variables
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://codequasar-orbital-anomaly-openenv.hf.space"
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional if using local docker image
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


# Required OpenAI client configuration
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
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
    print("START")

    with OrbitalAnomalyOpenenvEnv(base_url=API_BASE_URL).sync() as env:
        result = env.reset()
        obs = result.observation

        for step in range(12):
            action_name = choose_action(obs)

            result = env.step(
                OrbitalAnomalyOpenenvAction(action_type=action_name)
            )
            obs = result.observation

            print(f"STEP {step + 1}: {action_name}")

            if result.done:
                break

        print(f"END reward={result.reward}")


if __name__ == "__main__":
    main()