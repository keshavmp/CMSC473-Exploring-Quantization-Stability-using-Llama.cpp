"""Compare every run in results/ on one table, CSV, and bar chart.

Each results/<name>/metrics.json becomes one row; every task found in the files
becomes one column, so new models and new tasks are picked up automatically.

    python src/report.py                     # compare every run
    python src/report.py chat-q4 uniform-q4  # only these runs, in this order
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.config import Config
except ModuleNotFoundError:  # running directly: python src/report.py
    from config import Config

# Preferred metric per task; tasks not listed use the first numeric,
# non-stderr metric found in their results.
PRIMARY_METRICS = {
    "gsm8k": "exact_match,strict-match",
    "ifeval": "prompt_level_strict_acc,none",
    "mmlu_college_computer_science": "acc,none",
}


def find_result_files(results_dir: Path) -> list[Path]:
    """Find task metrics only; accept legacy flat results during migration."""
    nested = sorted(results_dir.glob("*/metrics.json"))
    names = {path.parent.name for path in nested}
    legacy = [path for path in results_dir.glob("*.json")
              if not path.name.endswith("_samples.json") and path.stem not in names]
    return sorted(nested + legacy, key=run_name)


def run_name(path: Path) -> str:
    return path.parent.name if path.name == "metrics.json" else path.stem


def score_for(task: str, task_results: dict) -> float | None:
    """Pick the primary metric for this task, skipping stderr and "N/A" values."""
    primary = PRIMARY_METRICS.get(task)
    if primary is not None and isinstance(task_results.get(primary), (int, float)):
        return task_results[primary]

    for key, value in task_results.items():
        if "stderr" in key or not isinstance(value, (int, float)):
            continue
        return value
    return None


def model_size_mib(path: Path) -> float | None:
    """Prefer recorded run size; older runs use the local GGUF when available."""
    nested = path.name == "metrics.json"
    config_file = path.parent / "config.json" if nested else path.with_name("missing-config")
    settings = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
    size = settings.get("model_size_bytes")
    if size is None:
        root = path.parent.parent.parent if nested else path.parent.parent
        model = Path(settings.get("model_file") or root / "models" / f"{run_name(path)}.gguf")
        if not model.is_file():
            return None
        size = model.stat().st_size
    return round(size / 1024**2, 2)


def task_columns(table: pd.DataFrame) -> list[str]:
    return [col for col in table if col != "size_mib" and not col.startswith("kl_")]


def build_table(files: list[Path]) -> pd.DataFrame:
    """Task scores are percentages; domain KL statistics stay in nats."""
    runs: dict[str, dict[str, float | None]] = {}
    tasks: list[str] = []

    for path in files:
        results = json.loads(path.read_text(encoding="utf-8"))
        name = run_name(path)
        results = results.get("quality", results)  # standalone KV-cache experiments
        runs[name] = {}
        size = model_size_mib(path)
        if size is not None:
            runs[name]["size_mib"] = size
            if "size_mib" not in tasks:
                tasks.append("size_mib")
        for task, task_results in results.items():
            if task not in tasks:
                tasks.append(task)
            score = score_for(task, task_results)
            runs[name][task] = None if score is None else round(100 * score, 1)

        kl_file = path.parent / "kl.json" if path.name == "metrics.json" else None
        if kl_file is not None and kl_file.exists():
            domains = json.loads(kl_file.read_text(encoding="utf-8"))["domains"]
            for domain, metrics in domains.items():
                for metric in ("mean", "uncertainty", "median", "p95"):
                    column = f"kl_{domain}_{metric}"
                    if column not in tasks:
                        tasks.append(column)
                    runs[name][column] = metrics[metric]

    return pd.DataFrame.from_dict(runs, orient="index").reindex(columns=tasks)


def save_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, na_rep="")


def save_plot(table: pd.DataFrame, path: Path) -> None:
    """One group of bars per task, one bar per run."""
    scores = table[task_columns(table)]
    ax = scores.plot(kind="bar", figsize=(max(10, len(table) * 1.5), 5), ylim=(0, 100))
    ax.set_ylabel("score (%)")
    ax.set_xlabel("")
    ax.legend(title="task", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(ax.figure)


def save_kl_plot(table: pd.DataFrame, path: Path) -> None:
    """KL has its own axis: nats, lower is better."""
    columns = [col for col in table if col.startswith("kl_") and col.endswith("_mean")]
    if not columns:
        return
    ax = table[columns].plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("KL (nats, lower is better)")
    ax.set_xlabel("")
    ax.legend(title="domain", labels=[col[3:-5] for col in columns])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(ax.figure)


def save_size_plot(table: pd.DataFrame, path: Path) -> None:
    """One panel per task or KL domain, with model size on every x-axis."""
    if "size_mib" not in table:
        return
    columns = task_columns(table) + [col for col in table if col.startswith("kl_") and col.endswith("_mean")]
    if not columns:
        return
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4), squeeze=False)
    colors = plt.get_cmap("tab10")
    for ax, column in zip(axes[0], columns):
        for index, (name, row) in enumerate(table.iterrows()):
            if pd.notna(row["size_mib"]) and pd.notna(row[column]):
                ax.scatter(row["size_mib"], row[column], color=colors(index % 10), label=name)
        ax.set_title(column)
        ax.set_xlabel("GGUF size (MiB)")
        ax.set_ylabel("KL (nats, lower is better)" if column.startswith("kl_") else "score (%)")
        ax.grid(alpha=0.2)
    handles, labels = [], []
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_samples(path: Path | None) -> dict:
    if path is None:
        return {}
    samples = path.parent / "samples.json" if path.name == "metrics.json" else path.with_name(f"{path.stem}_samples.json")
    return json.loads(samples.read_text(encoding="utf-8")) if samples.exists() else {}


def sample_scores(rows: list[dict], task: str) -> dict:
    """One binary correctness value per identical question, prompt, and target."""
    if task not in PRIMARY_METRICS:
        return {}
    metric, filter_name = PRIMARY_METRICS[task].split(",")
    scores = {}
    for row in rows:
        hashes = tuple(row.get(key) for key in ("doc_hash", "prompt_hash", "target_hash"))
        if row.get("filter", "none") != filter_name or not all(hashes):
            continue
        value = row.get(metric)
        if isinstance(value, (bool, int, float)) and value in (0, 1):
            scores[hashes] = bool(value)
    return scores


def build_stability_table(files: list[Path], baseline: Path | None) -> pd.DataFrame:
    """Compare binary correctness only on samples shared with the baseline."""
    columns = ["run", "task", "compared", "correct_to_incorrect", "incorrect_to_correct", "unchanged"]
    reference = load_samples(baseline)
    records = []
    for path in files:
        if path == baseline:
            continue
        for task, rows in load_samples(path).items():
            base = sample_scores(reference.get(task, []), task)
            run = sample_scores(rows, task)
            shared = base.keys() & run.keys()
            if not shared:
                continue
            lost = sum(base[key] and not run[key] for key in shared)
            gained = sum(not base[key] and run[key] for key in shared)
            records.append([run_name(path), task, len(shared), lost, gained, len(shared) - lost - gained])
    return pd.DataFrame(records, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        help="run names to include (default: every results/<name>/metrics.json)",
    )
    parser.add_argument("--baseline", default=Config().source_model.stem,
                        help="reference run for question-level correctness comparisons")
    args = parser.parse_args()

    cfg = Config()
    files = find_result_files(cfg.results_dir)
    baseline = next((path for path in files if run_name(path) == args.baseline), None)
    if args.names:
        by_name = {run_name(path): path for path in files}
        missing = [name for name in args.names if name not in by_name]
        if missing:
            parser.error(f"no results for: {', '.join(missing)}")
        files = [by_name[name] for name in args.names]

    if not files:
        parser.error(f"no result files in {cfg.results_dir}")

    table = build_table(files)

    print("\n" + table.to_string(na_rep="-") + "\n")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = cfg.results_dir / "summary.csv"
    plot_file = cfg.results_dir / "summary.png"
    save_csv(table, csv_file)
    save_plot(table, plot_file)

    kl_plot = cfg.results_dir / "kl_summary.png"
    if any(col.startswith("kl_") for col in table):
        save_kl_plot(table, kl_plot)
        print(f"Saved: {kl_plot}")

    if "size_mib" in table:
        size_plot = cfg.results_dir / "size_quality.png"
        save_size_plot(table, size_plot)
        print(f"Saved: {size_plot}")

    stability = build_stability_table(files, baseline)
    stability_file = cfg.results_dir / "stability.csv"
    stability.to_csv(stability_file, index=False)
    if stability.empty:
        print(f"No matching samples for baseline {args.baseline}; rerun it with sample logging.")
    else:
        print(f"\nCorrectness changes versus {args.baseline}:\n{stability.to_string(index=False)}\n")
    print(f"Saved: {stability_file}")
    print(f"Saved: {csv_file}")
    print(f"Saved: {plot_file}")


if __name__ == "__main__":
    main()
