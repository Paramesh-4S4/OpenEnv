import os
from openai import OpenAI
from env.environment import OpenEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_action(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def run_task(task_name):
    env = OpenEnv()
    obs = env.reset(task_name)

    print(f"[START] task={task_name} env=openenv model={MODEL_NAME}")

    rewards = []
    success = False

    for step in range(1, 6):
        try:
            action_str = get_action(f"Current state: {obs.state}. Next action?")
            action = Action(command=action_str)

            obs, reward, done, info = env.step(action)

            rewards.append(f"{reward:.2f}")

            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

            if done:
                success = info["score"] >= 1.0
                break

        except Exception as e:
            print(f"[STEP] step={step} action=error reward=0.00 done=true error={str(e)}")
            break

    print(f"[END] success={str(success).lower()} steps={step} rewards={','.join(rewards)}")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)