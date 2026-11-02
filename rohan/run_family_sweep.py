"""Pure quantization sweeps for several model families on a CPU-only Mac.

Run from the repository root:

    python -m rohan.run_family_sweep --family qwen gemma llama --prepare
    python -m rohan.run_family_sweep --family qwen gemma llama --limit 3
    python -m rohan.run_family_sweep --family qwen gemma --resume
"""
import argparse
import shutil
from pathlib import Path

from src.config import Config
from src.evaluate import resolve_tasks
from src.experiment import run_experiments
from src.kl import evaluate_kl, prepare_texts
from src.quantize import PURE_QUANT_TYPES
from src.server import get_model_file


ROHAN_DIR = Path(__file__).resolve().parent


def tool(name: str) -> Path:
    """Homebrew llama.cpp binaries; Config's macOS defaults point to a source build."""
    found = shutil.which(name)
    return Path(found) if found else Path("/usr/local/bin") / name


# One entry per model family. The source GGUF is downloaded from Hugging Face;
# every quantized variant is created locally from it (python -m rohan.quantize_family).
FAMILIES = {
    "qwen": {
        "experiment": "Qwen_Pure_Sweep",
        "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "hf_filename": "qwen2.5-1.5b-instruct-fp16.gguf",
        "source_type": "F16",  # no BF16 GGUF is published for Qwen2.5-1.5B
    },
    "gemma": {
        "experiment": "Gemma_Pure_Sweep",
        "hf_repo": "unsloth/gemma-3-1b-it-GGUF",
        "hf_filename": "gemma-3-1b-it-BF16.gguf",
        "source_type": "BF16",
    },
    "llama": {
        "experiment": "Llama_Pure_Sweep",
        "hf_repo": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "hf_filename": "Llama-3.2-1B-Instruct-BF16.gguf",
        "source_type": "BF16",
    },
}

# Same settings as run_evaluation.py except smaller sample caps and CPU only.
BASE = Config(
    project_root=ROHAN_DIR,
    num_examples=None,
    tasks=["gsm8k", "ifeval", "hellaswag"],
    task_sample_caps={"gsm8k": 100, "ifeval": 50, "hellaswag": 200},
    num_fewshot=0,
    request_workers=4,
    context_size=2048,
    gpu_layers="0",
    request_timeout=900,
    server_start_timeout=300,
    kl_enabled=True,
    llama_server=tool("llama-server"),
    quantize_tool=tool("llama-quantize"),
    llama_perplexity=tool("llama-perplexity"),
)


def family_configs(family: str) -> list[Config]:
    """The source baseline plus one pure variant per recipe, under rohan/."""
    spec = FAMILIES[family]
    models = BASE.models_dir / family
    base = BASE.with_overrides(
        experiment=spec["experiment"], source_model=models / spec["hf_filename"]
    )
    # The baseline keeps the BF16-pure-sweep name so src/report.py detects it.
    configs = [base.with_overrides(
        name="BF16-pure-sweep",
        model_file=base.source_model,
        quantization_type=spec["source_type"],
        hf_repo=spec["hf_repo"],
        hf_filename=spec["hf_filename"],
    )]
    for label, recipe in PURE_QUANT_TYPES.items():
        configs.append(base.with_overrides(
            name=f"pure-{label}-sweep",
            model_file=models / f"pure-{label}.gguf",
            quantization_type=recipe,
            pure_quantization=True,
        ))
    return configs


def check_tools():
    for path in (BASE.llama_server, BASE.quantize_tool, BASE.llama_perplexity):
        if not path.is_file():
            raise SystemExit(f"Missing llama.cpp tool: {path}. Run: brew install llama.cpp")


def check_models(families: list[str], download: bool = False):
    """Offer to download each source, then check that the quantized variants exist."""
    check_tools()
    for family in families:
        source = family_configs(family)[0]
        if not source.model_file.is_file():
            print(f"Missing {family} source: {source.model_file}")
            if not download:
                answer = input(f"Download it from {source.hf_repo}? [y/N]: ")
                if answer.strip().lower() not in ("y", "yes"):
                    raise SystemExit("Place the source at the path above before running evaluation.")
            get_model_file(source)
        if download:
            continue
        for config in family_configs(family)[1:]:
            if not config.model_file.is_file():
                raise SystemExit(
                    f"Missing model: {config.model_file}. "
                    f"Run: python -m rohan.quantize_family --family {family}"
                )


def prepare(families: list[str]):
    """Download and cache everything an unattended run needs."""
    check_models(families, download=True)
    prepare_texts(BASE)
    resolve_tasks(BASE)
    for family in families:
        # Builds the shared BF16 KL reference; the full run recomputes kl.json.
        print(f"\n=== Building KL reference: {family} ===")
        evaluate_kl(family_configs(family)[0])


def is_done(config: Config) -> bool:
    finished = (config.run_dir / "metrics.json").is_file()
    return finished and (not config.kl_enabled or (config.run_dir / "kl.json").is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", nargs="+", choices=FAMILIES, required=True,
                        help="families to run, in this order")
    parser.add_argument("--limit", type=int, help="Maximum questions per task (skips KL)")
    parser.add_argument("--resume", action="store_true", help="skip runs that already finished")
    parser.add_argument("--prepare", action="store_true",
                        help="download models, datasets, and KL texts, then exit")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive number")
    families = list(dict.fromkeys(args.family))

    if args.prepare:
        prepare(families)
        return

    check_models(families)
    configs = [config for family in families for config in family_configs(family)]
    if args.limit is not None:
        limited = []
        for config in configs:
            caps = {}
            for task, cap in config.task_sample_caps.items():
                caps[task] = min(cap, args.limit) if cap is not None else args.limit
            # Separate folder, so smoke tests never mix with full results.
            limited.append(config.with_overrides(
                name=f"{config.name}-limit{args.limit}",
                experiment=f"{config.experiment}-limit{args.limit}",
                task_sample_caps=caps,
                kl_enabled=False,
            ))
        configs = limited

    if args.resume:
        skipped = [config for config in configs if is_done(config)]
        for config in skipped:
            print(f"Skipping finished run: {config.experiment}/{config.name}")
        configs = [config for config in configs if not is_done(config)]

    summaries = run_experiments(configs)
    if any(summary["status"] != "ok" for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
