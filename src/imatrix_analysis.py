"""Export existing imatrix statistics, or collect a controlled model/domain sweep."""

import argparse
import glob
import hashlib
import itertools
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL = Path(r"C:\llama.cpp\build\bin\Release\llama-imatrix.exe")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(command):
    result = subprocess.run([str(arg) for arg in command], capture_output=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result.stdout, result.stderr


def parse_statistics(text):
    """Read pipe-delimited tables; fail rather than guess at changed CLI layouts."""
    records = {"tensor": [], "layer": []}
    kind = None
    columns = None
    for line in text.splitlines():
        if line.startswith("Computing tensor statistics"):
            kind, columns = "tensor", None
        elif line.startswith("Computing layer statistics"):
            kind, columns = "layer", None
        elif kind and "|" in line:
            cells = [cell.strip() for cell in line.split("|")]
            if cells[0] == "Layer":
                legacy = "L2 Dist" not in line
                columns = (["mean", "stddev", "skew", "kurtosis", "entropy_pct"]
                           if kind == "tensor" else [])
                columns += ["energy", "gain"]
                if not legacy:
                    columns += ["l2_distance"]
                if kind == "layer":
                    columns += ["cosine_similarity"]
                columns += ["pearson", "covariance"]
                continue
            if columns is None:
                continue
            values = " ".join(cells[2:] if kind == "tensor" else cells[1:]).split()
            if len(values) != len(columns):
                raise ValueError(f"Unrecognized statistics row: {line}")
            row = {"layer": cells[0]}
            if kind == "tensor":
                row["tensor"] = cells[1]
            row.update({key: float(value.rstrip("%")) for key, value in zip(columns, values)})
            records[kind].append(row)
    if not records["tensor"] or not records["layer"]:
        raise ValueError("Missing statistics tables; inspect the saved raw output.")
    return records


def collect(args, tool):
    if not args.model:
        raise ValueError('--collect requires --model (for example --model "models/*.gguf").')
    models = set()
    for pattern in args.model:
        matches = glob.glob(pattern)
        if not matches:
            raise ValueError(f"No models match {pattern}")
        models.update(Path(path).resolve() for path in matches)
    texts = sorted(args.calibration_dir.glob("*.txt"))
    if args.task:
        texts = [path for path in texts if path.stem in args.task]
        missing = set(args.task) - {path.stem for path in texts}
        if missing:
            raise ValueError(f"Missing calibration tasks: {sorted(missing)}")
    if not texts:
        raise ValueError("No calibration .txt files found")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "collection_plan.json").write_text(json.dumps({
        "models": [model.stem for model in sorted(models)], "tasks": [text.stem for text in texts],
        "expected_matrices": len(models) * len(texts), "chunks": args.chunks,
    }, indent=2), encoding="utf-8")
    version = run([tool, "--version"])
    text_hashes = {path: digest(path) for path in texts}
    for model in sorted(models):
        model_hash = digest(model)
        for text in texts:
            provenance = {
                "model": model.stem, "model_path": str(model), "model_sha256": model_hash,
                "task": text.stem, "calibration_path": str(text.resolve()),
                "calibration_sha256": text_hashes[text], "chunks": args.chunks,
                "context": args.context, "gpu_layers": args.gpu_layers,
                "parse_special": True, "batch_size": args.context,
                "tool_version": version, "tool_sha256": digest(tool),
            }
            key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:12]
            folder = args.matrix_dir / "collected" / model.stem / key
            folder.mkdir(parents=True, exist_ok=True)
            matrix = folder / f"{text.stem}.imatrix.gguf"
            manifest = matrix.with_suffix(".json")
            if matrix.exists() and manifest.exists():
                saved = json.loads(manifest.read_text(encoding="utf-8"))
                if saved.get("matrix_sha256") == digest(matrix):
                    print(f"Reuse {model.stem} / {text.stem}", flush=True)
                    continue
            temporary = matrix.with_suffix(".partial.gguf")
            command = [tool, "-m", model, "-f", text, "-o", temporary,
                       "--chunks", args.chunks, "-c", args.context, "-b", args.context,
                       "-ngl", args.gpu_layers, "--parse-special", "--no-ppl"]
            chunk_label = "all chunks" if args.chunks == -1 else f"{args.chunks} chunks"
            print(f"Collect {model.stem} / {text.stem} ({chunk_label})", flush=True)
            with (folder / "collection.log").open("w", encoding="utf-8") as log:
                subprocess.run([str(arg) for arg in command], stdout=log, stderr=log, check=True)
            temporary.replace(matrix)
            provenance.update(matrix_sha256=digest(matrix), command=list(map(str, command)),
                              created_utc=datetime.now(timezone.utc).isoformat())
            manifest.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def analyze(args, tool):
    matrices = sorted(args.matrix_dir.rglob("*.gguf"))
    matrices = [path for path in matrices if not path.name.endswith(".partial.gguf")]
    if not matrices:
        raise ValueError(f"No matrices found in {args.matrix_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {"tensor": [], "layer": []}
    inventory = []
    for matrix in matrices:
        relative = matrix.relative_to(args.matrix_dir)
        identity = relative.as_posix().removesuffix(".gguf")
        raw_dir = args.output / "raw" / relative.parent
        raw_dir.mkdir(parents=True, exist_ok=True)
        stdout, stderr = run([tool, "-m", args.source_model, "--in-file", matrix, "--show-statistics"])
        (raw_dir / f"{matrix.stem}.txt").write_text(stdout, encoding="utf-8")
        (raw_dir / f"{matrix.stem}.stderr.txt").write_text(stderr, encoding="utf-8")
        manifest = matrix.with_suffix(".json")
        provenance = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        matrix_hash = digest(matrix)
        if provenance and provenance.get("matrix_sha256") != matrix_hash:
            raise ValueError(f"Matrix does not match provenance: {matrix}")
        metadata = {"matrix": identity, "model": provenance.get("model", "unrecorded"),
                    "task": provenance.get("task", matrix.stem.removesuffix(".imatrix"))}
        for kind, table in parse_statistics(stdout).items():
            rows[kind].extend({**metadata, **row} for row in table)
        inventory.append({**metadata, "path": str(matrix.resolve()), "sha256": matrix_hash,
                          "collection": provenance or None})
    tables = {kind: pd.DataFrame(table) for kind, table in rows.items()}
    for kind, table in tables.items():
        table.to_csv(args.output / f"{kind}_statistics.csv", index=False)
    layers = tables["layer"]
    gain = layers.pivot(index="layer", columns="matrix", values="gain")
    gain = gain.reindex(sorted(gain.index, key=lambda x: (not str(x).isdigit(), int(x) if str(x).isdigit() else str(x))))
    gain.to_csv(args.output / "layer_gain.csv")
    comparisons = []
    for left, right in itertools.combinations(gain.columns, 2):
        paired = gain[[left, right]].replace([float("inf"), -float("inf")], float("nan")).dropna()
        correlation = float("nan")
        if len(paired) >= 3 and all(paired[col].nunique() > 1 for col in paired):
            correlation = paired[left].rank().corr(paired[right].rank())
        comparisons.append({"matrix_a": left, "matrix_b": right, "layers_compared": len(paired),
                            "gain_rank_correlation": correlation,
                            "mean_absolute_gain_difference": (paired[left] - paired[right]).abs().mean()})
    pd.DataFrame(comparisons, columns=["matrix_a", "matrix_b", "layers_compared",
                                     "gain_rank_correlation", "mean_absolute_gain_difference"]).to_csv(
                                         args.output / "pairwise_gain.csv", index=False)
    findings = ["Descriptive imatrix analysis", "",
                "Gain compares activation RMS with the previous layer; it is not accuracy or quantization error.",
                "The layer table aggregates collected tensor inputs, not a single residual-stream activation.",
                "Calibration domains (chat/math/code) are distinct from evaluation benchmarks.",
                "Existing matrices without manifests have unrecorded collection settings/model identity.",
                "Compare controlled runs with identical text, context, chunks and tool build.",
                "CLI values are rounded; CSVs preserve displayed precision only.", ""]
    for identity, group in layers.groupby("matrix", sort=True):
        valid = group[group["gain"].map(math.isfinite)]
        if not valid.empty:
            peak = valid.loc[valid["gain"].idxmax()]
            findings.append(f"{identity}: highest layer Gain = {peak['gain']:.2f} at layer {peak['layer']}.")
    for comparison in comparisons:
        value = comparison["gain_rank_correlation"]
        if math.isfinite(value):
            findings.append(f"{comparison['matrix_a']} vs {comparison['matrix_b']}: layer Gain rank correlation "
                            f"= {value:.3f} ({comparison['layers_compared']} layers; descriptive, no significance test).")
    findings += ["", "Next experiment: repeat controlled collection across independent calibration samples,",
                 "then test whether Gain changes track held-out task accuracy or KL divergence.",
                 "High Gain alone does not establish which tensors need higher precision."]
    (args.output / "findings.txt").write_text("\n".join(findings) + "\n", encoding="utf-8")
    (args.output / "inventory.json").write_text(json.dumps({
        "exported_utc": datetime.now(timezone.utc).isoformat(), "tool": str(tool),
        "tool_sha256": digest(tool), "tool_version": run([tool, "--version"]), "matrices": inventory,
    }, indent=2), encoding="utf-8")
    try:
        from src.imatrix_report import generate_report
    except ModuleNotFoundError:
        from imatrix_report import generate_report
    generate_report(args.output, args.source_model.stem)
    print("\n".join(findings))
    print(f"\nSaved tables, raw output, provenance and plot to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL if DEFAULT_TOOL.exists() else shutil.which("llama-imatrix"))
    parser.add_argument("--matrix-dir", type=Path, default=ROOT / "models" / "imatrix")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "imatrix_analysis")
    parser.add_argument("--source-model", type=Path, default=ROOT / "models" / "Llama-3.2-1B-Instruct-BF16.gguf",
                        help="Required CLI model argument for display; does not determine matrix provenance")
    parser.add_argument("--collect", action="store_true", help="Run calibration before exporting (can be slow)")
    parser.add_argument("--model", action="append", help="Model path or quoted glob; repeat for more models")
    parser.add_argument("--task", action="append", help="Calibration filename stem to collect; default all")
    parser.add_argument("--calibration-dir", type=Path, default=ROOT / "data" / "calibration")
    parser.add_argument("--chunks", type=int, default=-1,
                        help="Chunks per run; default -1 uses all complete chunks; 2 is a quick pilot")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--gpu-layers", default="99")
    args = parser.parse_args()
    if not args.tool or not Path(args.tool).is_file():
        parser.error("llama-imatrix not found; supply --tool PATH")
    if args.chunks != -1 and args.chunks < 1:
        parser.error("--chunks must be -1 (all) or a positive integer")
    if args.context < 1:
        parser.error("--context must be positive")
    if not args.collect and (args.model or args.task):
        parser.error("--model and --task select collection inputs; use with --collect")
    if args.collect:
        collect(args, Path(args.tool))
    analyze(args, Path(args.tool))


if __name__ == "__main__":
    main()
