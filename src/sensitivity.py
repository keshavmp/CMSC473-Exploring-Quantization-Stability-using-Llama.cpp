"""Per-block and per-tensor-type quantization sensitivity, measured with KL against the BF16 source.
 
For each precision combination (background -> target) two separate sweeps run:
    block    16 variants: one whole block (blk.N) is set to the target precision
    tensor    7 variants: one tensor type (attn_q ... ffn_down) is set to the target precision in every block
Everything else stays at the background precision. Every variant is generated directly from the BF16 source,
checked tensor by tensor (only the intended weights may differ), and scored on the same held-out
chat/math/code text. The BF16 source and each pure model are scored too, as the control for their background.
 
Combinations: BF16 -> Q2, Q4, Q6 | Q8 -> Q2, Q4, Q6 | Q6 -> Q2, Q4 | Q4 -> Q2 | Q3 -> Q2
 
    python -m src.sensitivity --backgrounds bf16 --levels q4_k --sweep block --blocks 3   # one variant, to time it
    python -m src.sensitivity --backgrounds q8_0 --sweep tensor                           # one background, tensor sweep
    python -m src.sensitivity                                                             # everything (230 variants)
 
Runs that already have a kl.json are skipped. Results go to results/<experiment>/block_sensitivity.csv,
tensor_sensitivity.csv and pure_baselines.csv; separate batches are merged into them.
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
 
PAIRS = [
    ("bf16", "q2_k"), ("bf16", "q4_k"), ("bf16", "q6_k"),
    ("q8_0", "q2_k"), ("q8_0", "q4_k"), ("q8_0", "q6_k"),
    ("q6_k", "q2_k"), ("q6_k", "q4_k"),
    ("q4_k", "q2_k"),
    ("q3_k", "q2_k"),
]
TENSOR_TYPES = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
 
# label -> llama-quantize recipe, and label -> tensor type name stored in the GGUF
FTYPE = {"bf16": "BF16", "q8_0": "Q8_0", "q6_k": "Q6_K", "q4_k": "Q4_K_M", "q3_k": "Q3_K_M", "q2_k": "Q2_K"}
GGML_NAME = {"bf16": "BF16", "q8_0": "Q8_0", "q6_k": "Q6_K", "q4_k": "Q4_K",
             "q3_k": "Q3_K", "q2_k": "Q2_K"}
LABEL_OF = {recipe: label for label, recipe in FTYPE.items()}
# llama-quantize ignores --tensor-type when the base recipe is BF16, so BF16-background variants use Q8_0
# as the base recipe and pin every weight outside the target back to BF16.
VARIANT_RECIPE = {**FTYPE, "bf16": "Q8_0"}
 
# Controls: the BF16 source itself, plus every pure quantization.
BASELINES = {"bf16": "BF16", **PURE_QUANT_TYPES}
 
TENSOR_RE = re.compile(r"blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$")
FIELDS = ["run", "background", "target", "sweep", "changed", "domain", "kl_mean", "kl_uncertainty", "tokens"]
 
 
def verify(model: Path, background: str, target: str, sweep: str, which) -> None:
    """Fail loudly unless only the intended weights are at the target type and every other weight, embedding included, is at the background type."""
    reader = GGUFReader(str(model))
    bg, tg = GGML_NAME[background], GGML_NAME[target]
    wrong, n_target = [], 0
    for t in reader.tensors:
        if len(t.shape) < 2 or not t.name.endswith(".weight"):
            continue  # norms and rope tables stay F32
        if sweep == "block":
            m = re.match(r"blk\.(\d+)\.", t.name)
            hit = bool(m) and int(m[1]) == which
        else:
            tm = TENSOR_RE.match(t.name)
            hit = bool(tm) and tm[2] == which
        expected = tg if hit else bg
        n_target += hit
        if t.tensor_type.name != expected:
            wrong.append((t.name, t.tensor_type.name, expected))
    del reader
    gc.collect()
    if n_target == 0:
        raise RuntimeError(f"{model.name}: no tensors matched the target {sweep}={which}")
    if wrong:
        raise RuntimeError(f"{model.name}: {len(wrong)} tensors have the wrong type, e.g. {wrong[:3]}")
 
 
def delete(model: Path) -> None:
    try:
        model.unlink()
    except PermissionError:
        print(f"Could not delete {model.name}; remove it by hand.")
 
 
def save(rows: list[dict], filename: str) -> None:
    """Merge into the existing CSV (a rerun replaces its own rows), so separate batches don't overwrite each other."""
    if not rows:
        return
    path = CFG.results_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            old = [row for row in csv.DictReader(f) if "background" in row]  # drop rows from the old CSV layout
        runs = {row["run"] for row in rows}
        rows = [row for row in old if row["run"] not in runs] + rows
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")
 
 
def build_jobs(args):
    """Yield (name, background, target, sweep, which, overrides)."""
    pairs = [p for p in PAIRS
             if (not args.backgrounds or p[0] in args.backgrounds)
             and (not args.levels or p[1] in args.levels)]
    for bg, tg in pairs:
        tn = GGML_NAME[tg]
        pin = [r"\.weight=BF16"] if bg == "bf16" else []  # after the target pattern: the first match wins
        if args.sweep in ("block", "both"):
            for b in args.blocks:
                yield (f"{bg}-to-{tg}_blk{b:02d}", bg, tg, "block", b, [rf"blk\.{b}\.={tn}"] + pin)
        if args.sweep in ("tensor", "both"):
            for tt in args.tensor_types:
                yield (f"{bg}-to-{tg}_{tt}", bg, tg, "tensor", tt, [rf"blk\.\d+\.{tt}\.weight={tn}"] + pin)
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", choices=["block", "tensor", "both"], default="both")
    parser.add_argument("--backgrounds", nargs="*", default=None, help="subset of background types")
    parser.add_argument("--levels", nargs="*", default=None, help="subset of target types")
    parser.add_argument("--blocks", nargs="+", type=int, default=list(range(BLOCKS)))
    parser.add_argument("--tensor-types", nargs="+", default=TENSOR_TYPES)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
 
    assert "bf16" in str(CFG.source_model).lower(), f"source_model must be BF16, got {CFG.source_model}"
 
    CFG.models_dir.mkdir(exist_ok=True)
    block_rows, tensor_rows, baseline_rows = [], [], []
 
    def collect(rows, name, bg, tg, sweep, which, kl_file):
        for domain, d in json.loads(kl_file.read_text(encoding="utf-8"))["domains"].items():
            rows.append(dict(run=name, background=bg, target=tg, sweep=sweep, changed=which,
                             domain=domain, kl_mean=d["mean"],
                             kl_uncertainty=d["uncertainty"], tokens=d["tokens"]))
 
    try:
        if not args.skip_baselines:
            for key, recipe in BASELINES.items():
                name, label = f"pure-{key}", LABEL_OF.get(recipe, recipe.lower())
                model_file = CFG.source_model if recipe == "BF16" else CFG.models_dir / f"{name}.gguf"
                run = CFG.with_overrides(name=name, model_file=model_file)
                model, kl_file = run.resolve_model_file(), run.run_dir / "kl.json"
                if not kl_file.exists():
                    if not model.exists():
                        quantize_model(CFG.quantize_tool, CFG.source_model, model, recipe,
                                       pure=True, tensor_types=None)
                    evaluate_kl(run)
                    if not args.keep and model != CFG.source_model and model.exists():
                        delete(model)
                collect(baseline_rows, name, label, label, "baseline", "all", kl_file)
 
        for name, bg, tg, sweep, which, overrides in build_jobs(args):
            run = CFG.with_overrides(name=name, model_file=CFG.models_dir / f"{name}.gguf")
            model, kl_file = run.resolve_model_file(), run.run_dir / "kl.json"
            if not kl_file.exists():
                model.unlink(missing_ok=True)  # never reuse a half-written file from an interrupted run
                quantize_model(CFG.quantize_tool, CFG.source_model, model, VARIANT_RECIPE[bg],
                               pure=True, tensor_types=overrides)
                verify(model, bg, tg, sweep, which)
                evaluate_kl(run)
            collect(block_rows if sweep == "block" else tensor_rows, name, bg, tg, sweep, which, kl_file)
            if not args.keep and model.exists():
                delete(model)
    finally:
        save(block_rows, "block_sensitivity.csv")
        save(tensor_rows, "tensor_sensitivity.csv")
        save(baseline_rows, "pure_baselines.csv")
 
 
if __name__ == "__main__":
    main()