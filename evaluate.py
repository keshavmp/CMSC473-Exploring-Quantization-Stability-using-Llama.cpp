import json
import requests
import lm_eval
from pathlib import Path

BASE_URL = "http://127.0.0.1:8080"

TASKS = [
    "hellaswag",
    "arc_easy",
]

# Get Current Model
response = requests.get(f"{BASE_URL}/v1/models")
model_path = response.json()["data"][0]["id"]

# Turn the filename into a clean result name
model_name = Path(model_path).stem

print(f"Evaluating: {model_name}")

results = lm_eval.simple_evaluate(
    model="gguf",
    model_args=f"base_url={BASE_URL}, timeout=30",
    tasks=TASKS,
    num_fewshot=0,
    limit=50,
    batch_size=1,
    torch_random_seed=None, # PyTorch was not Working, This is a workaround to make it work.
)

Path("results").mkdir(exist_ok=True)

with open(f"results/{model_name}.json", "w") as f:
    json.dump(results["results"], f, indent=2)

print(f"Saved: results/{model_name}.json")