"""Run the experiments below. Use --limit 10 for a short evaluation."""
import argparse

from src.config import Config
from src.experiment import run_experiments
from src.quantize import PURE_QUANT_TYPES
from src.server import get_model_file


# Edit the experiment settings here.
BASE = Config(
    num_examples=None,
    tasks=["gsm8k", "ifeval", "hellaswag"],
    task_sample_caps={"gsm8k": 1000, "ifeval": None, "hellaswag": 2000},
    num_fewshot=0,
    request_workers=4,
    context_size=2048,
    kl_enabled=True,
    hf_repo="unsloth/Llama-3.2-1B-Instruct-GGUF",
    hf_filename="Llama-3.2-1B-Instruct-BF16.gguf",
)

CONFIGS = [BASE.with_overrides(
    name="BF16-pure-sweep",
    model_file=BASE.source_model,
    quantization_type="BF16",
)]
for label, recipe in PURE_QUANT_TYPES.items():
    CONFIGS.append(BASE.with_overrides(
        name=f"pure-{label}-sweep",
        model_file=BASE.models_dir / f"pure-{label}.gguf",
        quantization_type=recipe,
        pure_quantization=True,
        hf_repo=None,
        hf_filename=None,
    ))


def check_models():
    """Offer to download BF16, then check that the Quantized variants exist."""
    if not BASE.source_model.is_file():
        print(f"Missing BF16 source: {BASE.source_model}")
        answer = input("Download the BF16 source from Hugging Face? [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            raise SystemExit("Place the BF16 source at the path above before running evaluation.")
        get_model_file(BASE.with_overrides(model_file=BASE.source_model))

    for config in CONFIGS:
        model = config.resolve_model_file()
        if not model.is_file():
            raise SystemExit(
                f"Missing model: {model}. Edit the settings at the bottom of "
                "src/quantize.py, then run: python src/quantize.py"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", help="save the entire sweep under results/NAME/")
    parser.add_argument("--limit", type=int, help="Maximum questions per task (skips KL)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive number")
    try:
        BASE.with_overrides(experiment=args.experiment)
    except ValueError as error:
        parser.error(str(error))

    check_models()
    configs = CONFIGS
    if args.limit is not None:
        configs = []
        for config in CONFIGS:
            caps = {}
            for task, cap in config.task_sample_caps.items():
                caps[task] = min(cap, args.limit) if cap is not None else args.limit
            configs.append(config.with_overrides(
                name=f"{config.name}-limit{args.limit}",
                task_sample_caps=caps,
                kl_enabled=False,
            ))

    if args.experiment is not None:
        configs = [config.with_overrides(experiment=args.experiment) for config in configs]
    summaries = run_experiments(configs)
    if any(summary["status"] != "ok" for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
