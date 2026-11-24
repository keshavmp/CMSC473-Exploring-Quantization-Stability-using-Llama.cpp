"""Inspect the *actual* request types before starting a costly llama.cpp sweep.

Run from the repository root:
    python imatrix_task_preflight.py

The script loads the same task names as the imatrix benchmark sweep,
constructs two requests per task, and reports their real harness dispatch type.
It does not start llama-server or evaluate any model.
"""
from collections import Counter
from lm_eval.tasks import TaskManager

TASKS = [
    "gsm8k",
    "ifeval",
    "mmlu_college_computer_science_generative",
]


def main():
    manager = TaskManager()
    tasks = manager.load(TASKS)["tasks"]
    invalid = False
    for name, task in tasks.items():
        config = task.dump_config()
        if hasattr(task, "set_fewshot_seed"):
            task.set_fewshot_seed(1234)
        task.build_all_requests(limit=2, rank=0, world_size=1)
        request_types = Counter(item.request_type for item in task.instances)
        print(f"{name}: configured={config.get('output_type')}; "
              f"actual_requests={dict(request_types)}")
        if any(key != "generate_until" for key in request_types):
            invalid = True
    if invalid:
        raise SystemExit("Some tasks produced non-generative requests. Do not run the full sweep.")
    print("All sampled requests use generate_until. Investigate backend dispatch or unexpected tasks if a sweep still calls loglikelihood.")


if __name__ == "__main__":
    main()
