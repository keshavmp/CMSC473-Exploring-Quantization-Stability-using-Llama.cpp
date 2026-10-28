"""Per-block quantization sensitivity: one block at Q2/Q4/Q6 on an otherwise Q8_0 model, scored with KL.
"""
import argparse
import csv
import gc
import json
import re
from pathlib import Path

from gguf import GGUFReader

from src.config import Config
from src.kl import evaluate_kl
from src.quantize import PURE_QUANT_TYPES, quantize_model

BLOCKS = 16        # 0 to 15
KL_CHUNKS = 4        # pure sweep: 2 (510 scored tokens), test 4
EXPERIMENT = "Block_Sensitivity_Differences"

CFG = Config(experiment=EXPERIMENT, kl_enabled=True, kl_chunks=KL_CHUNKS)


def verify(model: Path, level: str, block: int) -> None:
    reader = GGUFReader(str(model))
    wrong = []
    for t in reader.tensors:
        m = re.match(r"blk\.(\d+)\..*weight$", t.name)
        if not m or len(t.shape) < 2:      # skip norms and non-block tensors
            continue
        expected = level.upper() if int(m[1]) == block else "Q8_0"
        if t.tensor_type.name != expected:
            wrong.append((t.name, t.tensor_type.name, expected))
    del reader
    gc.collect()                           
    if wrong:
        raise RuntimeError(f"{model.name}: {len(wrong)} tensors have the wrong type, e.g. {wrong[:3]}")


def delete(model: Path) -> None:
    try:
        model.unlink()
    except PermissionError:
        print(f"Could not delete {model.name}; remove it by hand.")


def save(rows: list[dict]) -> None:
    if not rows:
        return
    path = CFG.results_dir / "block_sensitivity.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", nargs="+", default=["q2_k", "q4_k", "q6_k"])
    parser.add_argument("--blocks", nargs="+", type=int, default=list(range(BLOCKS)))
    parser.add_argument("--skip-baselines", action="store_true", help="skip the pure-q2..q8 reference runs")
    parser.add_argument("--keep", action="store_true", help="keep the per-block .gguf files (about 1.3 GB each)")
    args = parser.parse_args()

    # (run name, level, block, base quant type, --tensor-type overrides)
    jobs = []
    if not args.skip_baselines:
        for label, recipe in PURE_QUANT_TYPES.items():
            jobs.append((f"pure-{label}", recipe, "all", recipe, []))
    for level in args.levels:
        for block in args.blocks:
            jobs.append((f"blk{block:02d}-{level}", level, block, "Q8_0", [rf"blk\.{block}\.={level}"]))

    CFG.models_dir.mkdir(exist_ok=True)
    rows = []
    try:
        for name, level, block, base_type, overrides in jobs:
            run = CFG.with_overrides(name=name, model_file=CFG.models_dir / f"{name}.gguf")
            model, kl_file = run.resolve_model_file(), run.run_dir / "kl.json"
            if not kl_file.exists():
                if not model.exists():
                    quantize_model(CFG.quantize_tool, CFG.source_model, model, base_type,
                                   pure=True, tensor_types=overrides or None)
                if block != "all":
                    verify(model, level, block)
                evaluate_kl(run)
            for domain, d in json.loads(kl_file.read_text(encoding="utf-8"))["domains"].items():
                rows.append(dict(run=name, level=level, block=block, domain=domain,
                                 kl_mean=d["mean"], kl_uncertainty=d["uncertainty"], tokens=d["tokens"]))
            if block != "all" and not args.keep and model.exists():
                delete(model)
    finally:
        save(rows)


if __name__ == "__main__":
    main()