"""Compare model and calibration-domain effects using exported imatrix tables."""

import argparse
import hashlib
import itertools
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROTOCOL_KEYS = ("chunks", "context", "batch_size", "parse_special", "gpu_layers",
                 "tool_sha256", "tool_version")
IDENTIFIERS = ["matrix", "model", "task", "layer", "tensor", "protocol", "calibration_sha256"]


def load_statistics(folder):
    inventory = json.loads((folder / "inventory.json").read_text(encoding="utf-8"))
    metadata = {}
    for item in inventory["matrices"]:
        collection = item.get("collection")
        if not collection:
            continue
        if any(key not in collection for key in PROTOCOL_KEYS + ("calibration_sha256",)):
            raise ValueError(f"Incomplete collection settings for {item['matrix']}")
        protocol = {key: collection[key] for key in PROTOCOL_KEYS}
        key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
        metadata[item["matrix"]] = {"protocol": key,
                                    "calibration_sha256": collection["calibration_sha256"]}
    frames = []
    for kind in ("layer", "tensor"):
        frame = pd.read_csv(folder / f"{kind}_statistics.csv", dtype={"layer": str})
        frame = frame[frame["matrix"].isin(metadata)].copy()
        if frame.empty:
            continue
        if kind == "layer":
            frame["tensor"] = ""
        for column in ("protocol", "calibration_sha256"):
            frame[column] = frame["matrix"].map(lambda matrix: metadata[matrix][column])
        frame = frame.melt(id_vars=IDENTIFIERS, var_name="statistic", value_name="value")
        frame["kind"] = kind
        frame["value"] = pd.to_numeric(frame["value"], errors="raise")
        frame["value"] = frame["value"].replace([np.inf, -np.inf], np.nan)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(long, baseline):
    unit_keys = ["protocol", "model", "task", "kind", "statistic"]
    if long.duplicated(unit_keys + ["layer", "tensor"]).any():
        raise ValueError("Multiple runs for the same model/domain/protocol; export each sweep separately.")
    run_summary = long.groupby(unit_keys, as_index=False).agg(
        mean_value=("value", "mean"), std_across_units=("value", "std"),
        n_units=("value", "count"))
    model_averages = run_summary.groupby(["protocol", "model", "kind", "statistic"], as_index=False).agg(
        mean_value=("mean_value", "mean"), std_across_tasks=("mean_value", "std"),
        n_tasks=("mean_value", "count"))
    unit_averages = long.groupby(
        ["protocol", "task", "kind", "statistic", "layer", "tensor"], as_index=False).agg(
            mean_value=("value", "mean"), std_across_models=("value", "std"), n_models=("value", "count"))
    task_averages = []
    overall_averages = []
    for scope, subset in (("all_models", run_summary),
                          ("quantized_models", run_summary[run_summary["model"] != baseline])):
        averaged = subset.groupby(["protocol", "task", "kind", "statistic"], as_index=False).agg(
            mean_value=("mean_value", "mean"), std_across_models=("mean_value", "std"),
            n_models=("mean_value", "count"))
        averaged["scope"] = scope
        task_averages.append(averaged)
        model_subset = model_averages if scope == "all_models" else model_averages[model_averages["model"] != baseline]
        overall = model_subset.groupby(["protocol", "kind", "statistic"], as_index=False).agg(
            mean_value=("mean_value", "mean"), std_across_models=("mean_value", "std"),
            n_models=("mean_value", "count"))
        overall["scope"] = scope
        overall_averages.append(overall)
    match_keys = ["protocol", "task", "calibration_sha256", "kind", "statistic", "layer", "tensor"]
    reference = long[long["model"] == baseline][match_keys + ["value"]].rename(columns={"value": "baseline_value"})
    if reference.duplicated(match_keys).any():
        raise ValueError("Ambiguous BF16 reference rows")
    changes = long.merge(reference, on=match_keys, how="inner", validate="many_to_one")
    changes["delta"] = changes["value"] - changes["baseline_value"]
    changes["absolute_delta"] = changes["delta"].abs()
    quantized = changes[changes["model"] != baseline]
    unit_change_averages = quantized.groupby(
        ["protocol", "task", "kind", "statistic", "layer", "tensor"], as_index=False).agg(
            mean_signed_delta=("delta", "mean"), mean_absolute_delta=("absolute_delta", "mean"),
            n_models=("delta", "count"))
    change_summary = quantized.groupby(unit_keys, as_index=False).agg(
        mean_signed_delta=("delta", "mean"), mean_absolute_delta=("absolute_delta", "mean"),
        max_absolute_delta=("absolute_delta", "max"), n_matched_units=("delta", "count"))
    model_changes = change_summary.groupby(["protocol", "model", "kind", "statistic"], as_index=False).agg(
        mean_signed_delta=("mean_signed_delta", "mean"), mean_absolute_delta=("mean_absolute_delta", "mean"),
        n_tasks=("mean_signed_delta", "count"))
    task_changes = change_summary.groupby(["protocol", "task", "kind", "statistic"], as_index=False).agg(
        mean_signed_delta=("mean_signed_delta", "mean"), mean_absolute_delta=("mean_absolute_delta", "mean"),
        std_absolute_delta_across_models=("mean_absolute_delta", "std"),
        n_models=("mean_signed_delta", "count"))
    grand_averages = model_changes.groupby(["protocol", "kind", "statistic"], as_index=False).agg(
        mean_signed_delta=("mean_signed_delta", "mean"), mean_absolute_delta=("mean_absolute_delta", "mean"),
        n_models=("mean_signed_delta", "count"))
    contrasts = []
    for (protocol, model, kind, statistic), group in long.groupby(["protocol", "model", "kind", "statistic"]):
        for task_a, task_b in itertools.combinations(sorted(group["task"].unique()), 2):
            left = group[group["task"] == task_a][["layer", "tensor", "value"]]
            right = group[group["task"] == task_b][["layer", "tensor", "value"]]
            paired = left.merge(right, on=["layer", "tensor"], suffixes=("_a", "_b"), validate="one_to_one")
            paired = paired.dropna(subset=["value_a", "value_b"])
            raw_delta = paired["value_b"] - paired["value_a"]
            effect = changes[(changes["protocol"] == protocol) & (changes["model"] == model)
                             & (changes["kind"] == kind) & (changes["statistic"] == statistic)]
            effect_pair = effect[effect["task"] == task_a][["layer", "tensor", "delta"]].merge(
                effect[effect["task"] == task_b][["layer", "tensor", "delta"]],
                on=["layer", "tensor"], suffixes=("_a", "_b"), validate="one_to_one").dropna()
            interaction = effect_pair["delta_b"] - effect_pair["delta_a"]
            contrasts.append({"protocol": protocol, "model": model, "kind": kind, "statistic": statistic,
                              "task_a": task_a, "task_b": task_b, "n_units": len(paired),
                              "mean_task_b_minus_a": raw_delta.mean(),
                              "mean_absolute_task_difference": raw_delta.abs().mean(),
                              "n_baseline_matched_units": len(effect_pair),
                              "mean_change_b_minus_a": interaction.mean(),
                              "mean_absolute_change_difference": interaction.abs().mean()})
    contrasts = pd.DataFrame(contrasts, columns=["protocol", "model", "kind", "statistic", "task_a", "task_b",
        "n_units", "mean_task_b_minus_a", "mean_absolute_task_difference", "n_baseline_matched_units",
        "mean_change_b_minus_a", "mean_absolute_change_difference"])
    task_contrast_averages = contrasts[contrasts["model"] != baseline].groupby(
        ["protocol", "kind", "statistic", "task_a", "task_b"], as_index=False).agg(
            mean_absolute_task_difference=("mean_absolute_task_difference", "mean"),
            mean_signed_change_difference=("mean_change_b_minus_a", "mean"),
            mean_absolute_change_difference=("mean_absolute_change_difference", "mean"),
            n_models=("mean_change_b_minus_a", "count"))
    return {"model_task_summary": run_summary, "model_averages": model_averages,
            "unit_task_averages": unit_averages, "unit_change_averages": unit_change_averages,
            "task_averages": pd.concat(task_averages, ignore_index=True),
            "overall_averages": pd.concat(overall_averages, ignore_index=True),
            "paired_changes": changes, "model_task_changes": change_summary,
            "model_change_averages": model_changes, "task_change_averages": task_changes,
            "overall_change_averages": grand_averages,
            "task_contrasts": contrasts, "task_contrast_averages": task_contrast_averages}


def model_label(name, baseline):
    if name == baseline:
        return "BF16"
    return name.removeprefix("pure-").upper()


def layer_order(values):
    return sorted(values, key=lambda value: (not str(value).isdigit(),
                  int(value) if str(value).isdigit() else str(value)))


def complete_domain_average(frame, tasks, value="value"):
    """Exclude unfinished models, then give each calibration domain equal weight."""
    complete = frame.groupby("model")["task"].agg(lambda found: set(tasks).issubset(set(found)))
    selected = frame[frame["model"].isin(complete[complete].index) & frame["task"].isin(tasks)]
    return selected.groupby(["model", "statistic", "layer"], as_index=False)[value].mean()


def line_comparison(frame, series, value, path, title, ylabel, baseline, zero=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, profile in frame.groupby(series, sort=True):
        profile = profile.copy()
        profile["layer_number"] = pd.to_numeric(profile["layer"], errors="coerce")
        profile = profile.dropna(subset=["layer_number"]).sort_values("layer_number")
        if not profile[value].notna().any():
            continue
        label = model_label(name, baseline) if series == "model" else name
        style = {"color": "black", "linestyle": "--", "linewidth": 2} if name == baseline else {}
        ax.plot(profile["layer_number"], profile[value], marker="o", markersize=4, label=label, **style)
    if zero:
        ax.axhline(0, color="gray", linewidth=1)
    ax.set(xlabel="Layer", ylabel=ylabel, title=title)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title="Precision" if series == "model" else "Calibration domain")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gain_heatmap(frame, models, path, title, baseline, subtitle, difference=False):
    grid = frame.pivot(index="layer", columns="model", values="value")
    grid = grid.reindex(index=layer_order(grid.index), columns=models)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(path.with_suffix(".csv"))
    values = grid.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    cmap = plt.get_cmap("RdBu_r" if difference else "viridis").with_extremes(bad="#e6e6e6")
    if difference:
        limit = max(float(np.abs(finite).max()), 0.01) if finite.size else 0.01
        limits = {"vmin": -limit, "vmax": limit}
    else:
        limits = {"vmin": 0, "vmax": max(float(finite.max()), 1) if finite.size else 1}
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.85), 8))
    picture = ax.imshow(values, aspect="auto", cmap=cmap, **limits)
    ax.set_xticks(range(len(models)), [model_label(model, baseline) for model in models])
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set(xlabel="Model precision", ylabel="Layer", title=title)
    for row, column in itertools.product(range(len(grid.index)), range(len(models))):
        value = grid.iloc[row, column]
        if pd.notna(value):
            norm = picture.norm(value)
            color = ("black" if 0.25 < norm < 0.75 else "white") if difference else ("black" if norm > 0.55 else "white")
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(picture, ax=ax, label="Gain change from BF16" if difference else "Gain")
    fig.text(0.5, 0.015, subtitle + "\nGray cells: undefined or unfinished.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_changes(tables, folder, long, baseline):
    legacy = list(folder.glob("model_task_changes_*.png")) + list(folder.glob("layer_gain.png"))
    for path in legacy:
        archive = folder / "previous_plots" / path.name
        archive.parent.mkdir(exist_ok=True)
        path.replace(archive)
    plan_path = folder / "collection_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig")) if plan_path.exists() else {}
    layers = long[long["kind"] == "layer"]
    labels = {"gain": "Gain", "energy": "Activation energy", "l2_distance": "Adjacent-layer L2 distance",
              "pearson": "Adjacent-layer Pearson correlation", "covariance": "Adjacent-layer covariance",
              "cosine_similarity": "Adjacent-layer cosine similarity"}
    highlights = []
    for protocol, group in layers.groupby("protocol"):
        tasks = plan.get("tasks", sorted(group["task"].unique()))
        models = plan.get("models", sorted(group["model"].unique()))
        average = complete_domain_average(group, tasks)
        average.to_csv(folder / f"model_layer_averages_{protocol}.csv", index=False)
        for statistic, profiles in average.groupby("statistic"):
            label = labels.get(statistic, statistic)
            line_comparison(profiles, "model", "value", folder / f"{statistic}_by_precision_{protocol}.png",
                            f"{label} by precision\nEqual average of {', '.join(tasks)}", label, baseline)
        gains = average[average["statistic"] == "gain"]
        if not gains.empty:
            gain_heatmap(gains, models, folder / f"gain_heatmap_{protocol}.png", "Gain by layer and precision",
                         baseline, "Equal average of " + ", ".join(tasks) + "; complete models only.")
        changes = tables["paired_changes"]
        changes = changes[(changes["protocol"] == protocol) & (changes["kind"] == "layer")]
        average_changes = complete_domain_average(changes, tasks, "delta")
        average_changes.to_csv(folder / f"model_layer_changes_{protocol}.csv", index=False)
        for statistic, profiles in average_changes.groupby("statistic"):
            label = labels.get(statistic, statistic)
            line_comparison(profiles, "model", "delta", folder / f"{statistic}_change_by_precision_{protocol}.png",
                            f"{label} change from BF16\nEqual average of {', '.join(tasks)}",
                            f"{label} minus BF16", baseline, zero=True)
        gain_changes = average_changes[average_changes["statistic"] == "gain"].rename(columns={"delta": "value"})
        if not gain_changes.empty:
            gain_heatmap(gain_changes, models, folder / f"gain_change_heatmap_{protocol}.png",
                         "Gain change from BF16 by layer and precision", baseline,
                         "Equal average of " + ", ".join(tasks) + "; complete models only.", difference=True)
        for model, profile in gains.groupby("model"):
            finite = profile.dropna(subset=["value"])
            if finite.empty:
                continue
            peak = finite.loc[finite["value"].idxmax()]
            row = {"protocol": protocol, "model": model, "n_tasks": len(tasks),
                   "peak_gain_layer": peak["layer"], "peak_gain": peak["value"]}
            delta = gain_changes[gain_changes["model"] == model].dropna(subset=["value"])
            if model != baseline and not delta.empty:
                increase = delta.loc[delta["value"].idxmax()]
                decrease = delta.loc[delta["value"].idxmin()]
                row.update(highest_delta_layer=increase["layer"], highest_delta=increase["value"],
                           lowest_delta_layer=decrease["layer"], lowest_delta=decrease["value"])
            highlights.append(row)
        comparisons = folder / "task_comparisons"
        for (model, statistic), profiles in group.groupby(["model", "statistic"]):
            label = labels.get(statistic, statistic)
            safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
            line_comparison(profiles, "task", "value", comparisons / f"{statistic}_tasks_{safe_model}_{protocol}.png",
                            f"{label} across tasks: {model_label(model, baseline)}", label, baseline)
        complete_models = set(average["model"])
        task_profiles = group[group["model"].isin(complete_models)].groupby(
            ["task", "statistic", "layer"], as_index=False)["value"].mean()
        for statistic, profiles in task_profiles.groupby("statistic"):
            label = labels.get(statistic, statistic)
            path = comparisons / f"average_{statistic}_by_task_{protocol}.png"
            line_comparison(profiles, "task", "value", path,
                            f"{label} across tasks\nMean across the same {len(complete_models)} complete models",
                            f"Mean {label}", baseline)
            if statistic == "gain":
                shutil.copyfile(path, folder / f"average_gain_by_task_{protocol}.png")
    pd.DataFrame(highlights, columns=["protocol", "model", "n_tasks", "peak_gain_layer", "peak_gain",
        "highest_delta_layer", "highest_delta", "lowest_delta_layer", "lowest_delta"]).to_csv(
            folder / "gain_layer_highlights.csv", index=False)


def generate_report(folder, baseline):
    folder = Path(folder)
    long = load_statistics(folder)
    if long.empty:
        (folder / "analysis.txt").write_text("No collection manifests: model comparisons require recorded provenance.\n", encoding="utf-8")
        return
    tables = summarize(long, baseline)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    plot_changes(tables, folder, long, baseline)
    lines = ["Model and calibration-domain analysis", "", f"Reference: {baseline}",
             "Changes = model statistic minus BF16 statistic, paired on domain, layer/tensor and collection settings.",
             "L2/PCC/CosSim columns in the raw tables compare adjacent layers, not the quantized model against BF16.",
             "Paired changes compare those statistics with BF16; they are not direct activation-vector distances.",
             "Means weight layers/tensors equally within a run, then domains equally within a model.",
             "Ensemble means weight models equally. Quantized-change averages exclude the BF16 zero-change reference.",
             "Sample counts are given; undefined values stay missing and standard deviations are descriptive, not confidence intervals.",
             "Raw statistics depend on domain content and token count; full-file domains have unequal lengths.",
             "Calibration domains chat/code/math are not evaluation benchmarks GSM8K/IFEval/HellaSwag.",
             "Changes smaller than the CLI display precision cannot be resolved here.", ""]
    plan_file = folder / "collection_plan.json"
    if plan_file.exists():
        plan = json.loads(plan_file.read_text(encoding="utf-8-sig"))
        observed = long["matrix"].nunique()
        expected = plan["expected_matrices"]
        status = "COMPLETE" if observed == expected else "PARTIAL"
        lines[2:2] = [f"{status}: {observed}/{expected} planned model/domain combinations exported.",
                      "Planned models: " + ", ".join(plan["models"]), ""]
    for protocol, subset in long.groupby("protocol"):
        coverage = subset[["model", "task"]].drop_duplicates()
        model_count, task_count = coverage["model"].nunique(), coverage["task"].nunique()
        lines += [f"Protocol {protocol}: {len(coverage)}/{model_count * task_count} model/domain combinations "
                  f"among {model_count} observed models and {task_count} observed domains."]
        if len(coverage) != model_count * task_count:
            lines.append("Coverage is incomplete; averages may use different domain/model sets (see n_tasks/n_models).")
        if baseline not in set(coverage["model"]):
            lines.append("No matching BF16 reference; baseline changes are unavailable.")
        values = tables["model_change_averages"]
        values = values[(values["protocol"] == protocol) & (values["kind"] == "layer") & (values["statistic"] == "gain")]
        lines.append("Per-model Gain changes (equal-domain averages of layer changes):")
        for row in values.itertuples():
            lines.append(f"  {row.model}: mean signed = {row.mean_signed_delta:.5f}; mean absolute = "
                         f"{row.mean_absolute_delta:.5f}; domains = {row.n_tasks}.")
        tasks = tables["task_change_averages"]
        tasks = tasks[(tasks["protocol"] == protocol) & (tasks["kind"] == "layer") & (tasks["statistic"] == "gain")]
        lines.append("Per-domain Gain changes averaged across quantized models:")
        for row in tasks.itertuples():
            lines.append(f"  {row.task}: mean signed = {row.mean_signed_delta:.5f}; mean absolute = "
                         f"{row.mean_absolute_delta:.5f}; models = {row.n_models}.")
        overall = tables["overall_change_averages"]
        overall = overall[(overall["protocol"] == protocol) & (overall["kind"] == "layer") & (overall["statistic"] == "gain")]
        for row in overall.itertuples():
            lines.append(f"Overall Gain change: signed {row.mean_signed_delta:.5f}; absolute {row.mean_absolute_delta:.5f} "
                         f"(equal-domain then equal-model averages across {row.n_models} quantized models).")
        contrast = tables["task_contrast_averages"]
        contrast = contrast[(contrast["protocol"] == protocol) & (contrast["kind"] == "layer") & (contrast["statistic"] == "gain")]
        for row in contrast.itertuples():
            lines.append(f"  {row.task_a} vs {row.task_b}: mean absolute difference in BF16-relative Gain change "
                         f"= {row.mean_absolute_change_difference:.5f} across {row.n_models} quantized models.")
        if not values.empty and values["mean_absolute_delta"].notna().any():
            largest = values.loc[values["mean_absolute_delta"].idxmax()]
            lines.append(f"Largest average Gain deviation: {largest['model']} ({largest['mean_absolute_delta']:.5f}).")
        if not tasks.empty and tasks["mean_absolute_delta"].notna().any():
            largest = tasks.loc[tasks["mean_absolute_delta"].idxmax()]
            lines.append(f"Largest domain-average Gain deviation: {largest['task']} ({largest['mean_absolute_delta']:.5f}).")
        layer_highlights = pd.read_csv(folder / "gain_layer_highlights.csv", dtype={
            "peak_gain_layer": str, "highest_delta_layer": str, "lowest_delta_layer": str})
        lines.append("Gain by layer and precision (equal-domain averages; complete models only):")
        for row in layer_highlights[layer_highlights["protocol"] == protocol].itertuples():
            detail = f"  {model_label(row.model, baseline)}: peak Gain {row.peak_gain:.3f} at layer {row.peak_gain_layer}."
            if pd.notna(row.highest_delta):
                detail += (f" Highest change from BF16 {row.highest_delta:+.3f} at layer {row.highest_delta_layer};"
                           f" lowest change {row.lowest_delta:+.3f} at layer {row.lowest_delta_layer}.")
            lines.append(detail)
        lines.append("Largest mean absolute BF16-relative changes for each reported statistic:")
        all_models = tables["model_change_averages"]
        all_models = all_models[all_models["protocol"] == protocol]
        all_tasks = tables["task_change_averages"]
        all_tasks = all_tasks[all_tasks["protocol"] == protocol]
        for (kind, statistic), group in all_models.groupby(["kind", "statistic"]):
            finite = group.dropna(subset=["mean_absolute_delta"])
            domain_group = all_tasks[(all_tasks["kind"] == kind) & (all_tasks["statistic"] == statistic)]
            domain_group = domain_group.dropna(subset=["mean_absolute_delta"])
            if finite.empty or domain_group.empty:
                continue
            model_peak = finite.loc[finite["mean_absolute_delta"].idxmax()]
            domain_peak = domain_group.loc[domain_group["mean_absolute_delta"].idxmax()]
            lines.append(f"  {kind}/{statistic}: model {model_peak['model']} = {model_peak['mean_absolute_delta']:.6g}; "
                         f"domain {domain_peak['task']} = {domain_peak['mean_absolute_delta']:.6g}.")
        lines.append("")
    lines += ["Domain contrasts are descriptive. Without repeated independent samples, they do not establish statistical significance.",
              "Domain-specific quantization effects are measured by (model_B - BF16_B) - (model_A - BF16_A),",
              "rather than attributing all raw domain differences to quantization.",
              "Confirm practical impact against held-out accuracy/KL results; activation Gain alone does not establish quality loss."]
    (folder / "analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved per-model changes, ensemble averages and domain contrasts to {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", default="Llama-3.2-1B-Instruct-BF16")
    args = parser.parse_args()
    generate_report(args.input, args.baseline)
