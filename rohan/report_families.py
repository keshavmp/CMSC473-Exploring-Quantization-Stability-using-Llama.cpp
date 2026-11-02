"""Report each family sweep under rohan/results/ and compare families.

    python -m rohan.report_families

Each rohan/results/<Family>_Pure_Sweep/ gets the same summary.csv, plots, and
stability.csv as results/Pure_Sweep/. The cross-family comparison is saved to
rohan/results/family_comparison.csv and family_comparison.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import pandas as pd

from rohan.run_family_sweep import BASE, FAMILIES
from src.config import PROJECT_ROOT
from src.report import (build_stability_table, build_table, find_result_files,
                        save_csv, save_kl_plot, save_plot, save_size_plot,
                        select_baseline)


ORIGINAL_LLAMA = PROJECT_ROOT / "results" / "Pure_Sweep" / "summary.csv"
ORIGINAL_LABEL = "llama (original GPU run, n=1000/541/2000)"
LEVELS = ["Q2", "Q3", "Q4", "Q5", "Q6", "Q8", "BF16"]
TASKS = {"gsm8k": "GSM8K (%)", "ifeval": "IFEval (%)", "hellaswag": "HellaSwag (%)"}

# Reference palette: the first three categorical slots pass all-pairs CVD checks.
COLORS = {"qwen": "#2a78d6", "gemma": "#eb6834", "llama": "#1baf7a"}
MUTED = "#8a8985"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def report_family(results_dir: Path) -> pd.DataFrame | None:
    """Write the standard src/report.py outputs for one experiment folder."""
    files = find_result_files(results_dir)
    if not files:
        return None
    baseline = select_baseline(files, None, "BF16-pure-sweep")
    table = build_table(files)
    save_csv(table, results_dir / "summary.csv")
    save_plot(table, results_dir / "summary.png")
    if any(col.startswith("kl_") for col in table):
        save_kl_plot(table, results_dir / "kl_summary.png")
    if "size_mib" in table:
        save_size_plot(table, results_dir / "size_quality.png")
    build_stability_table(files, baseline).to_csv(results_dir / "stability.csv", index=False)
    print(f"\n{results_dir.name}\n{table.to_string(na_rep='-')}")
    return table


def level(run: str) -> str | None:
    """BF16-pure-sweep -> BF16, pure-q4-sweep -> Q4; anything else is skipped."""
    if run == "BF16-pure-sweep":
        return "BF16"
    if run.startswith("pure-") and run.endswith("-sweep"):
        return run[len("pure-"):-len("-sweep")].upper()
    return None


def family_rows(family: str, table: pd.DataFrame) -> pd.DataFrame:
    """One row per quantization level: task scores, size, and mean KL over domains."""
    rows = []
    kl_means = [col for col in table if col.startswith("kl_") and col.endswith("_mean")]
    for run, values in table.iterrows():
        quant = level(str(run))
        if quant is None:
            continue
        row = {"family": family, "level": quant, "size_mib": values.get("size_mib")}
        row.update({task: values.get(task) for task in TASKS})
        row["kl_mean"] = values[kl_means].mean() if kl_means else None
        rows.append(row)
    return pd.DataFrame(rows)


def save_comparison_plot(comparison: pd.DataFrame, path: Path) -> None:
    """Small multiples: one panel per task plus mean KL, one line per family."""
    panels = list(TASKS.items()) + [("kl_mean", "Mean KL vs source (nats, log)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.2))
    for ax, (column, title) in zip(axes, panels):
        levels = LEVELS[:-1] if column == "kl_mean" else LEVELS  # source KL is ~0
        for family, rows in comparison.groupby("family", sort=False):
            rows = rows.set_index("level").reindex(levels)
            values = rows[column].astype(float)
            if values.isna().all():
                continue
            original = family == ORIGINAL_LABEL
            ax.plot(levels, values,
                    color=MUTED if original else COLORS.get(family, MUTED),
                    linestyle="--" if original else "-", linewidth=2,
                    marker="o", markersize=5, label=family, zorder=2 if original else 3)
        if column == "kl_mean":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.suptitle("Pure quantization sweep by model family (CPU, n=100/50/200)",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    frames = []
    for family, spec in FAMILIES.items():
        table = report_family(BASE.project_root / "results" / spec["experiment"])
        if table is None:
            print(f"No results yet: {spec['experiment']}")
            continue
        frames.append(family_rows(family, table))
    if not frames:
        raise SystemExit("No family results found under rohan/results/")
    if ORIGINAL_LLAMA.is_file():
        frames.append(family_rows(ORIGINAL_LABEL, pd.read_csv(ORIGINAL_LLAMA, index_col=0)))

    comparison = pd.concat(frames, ignore_index=True)
    output = BASE.project_root / "results"
    comparison.to_csv(output / "family_comparison.csv", index=False)
    save_comparison_plot(comparison, output / "family_comparison.png")
    print(f"\n{comparison.to_string(index=False)}")
    print(f"\nSaved: {output / 'family_comparison.csv'}")
    print(f"Saved: {output / 'family_comparison.png'}")


if __name__ == "__main__":
    main()
