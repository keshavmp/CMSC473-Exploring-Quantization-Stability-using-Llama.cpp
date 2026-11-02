"""Three figures connecting calibration importance to local quantization error."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("code", "math", "chat")
TYPES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
LABELS = ("Attention Q", "Attention K", "Attention V", "Attention output", "FFN gate", "FFN up", "FFN down")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def relative_energy(frame):
    result = frame.copy()
    if result.duplicated(["task", "tensor"]).any():
        raise ValueError("Repeated reference tensor/domain rows")
    counts = result.groupby("tensor")["task"].nunique()
    if not counts.eq(len(TASKS)).all():
        raise ValueError("Every tensor must have all three calibration domains")
    denominator = result.groupby("tensor")["energy"].transform("mean")
    result["relative_energy"] = result["energy"].div(denominator.where(denominator > 0))
    return result


def input_second_moment(sums, counts):
    sums = np.asarray(sums, dtype=np.float64).reshape(-1)
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if len(counts) != 1 or not np.isfinite(counts[0]) or counts[0] <= 0:
        raise ValueError("This analysis requires a dense tensor with one positive input count")
    moments = sums / counts[0]
    if not np.isfinite(moments).all() or (moments < 0).any():
        raise ValueError("Input second moments must be finite and nonnegative")
    return moments


def weighted_error(second_moments, column_squared_error):
    second_moments = np.asarray(second_moments, dtype=np.float64)
    column_squared_error = np.asarray(column_squared_error, dtype=np.float64)
    if second_moments.shape != column_squared_error.shape or second_moments.ndim != 1:
        raise ValueError("Weight input columns and imatrix dimensions must match")
    return float(np.dot(second_moments, column_squared_error))


def column_squared_error(original, quantized, dequantize, rows_per_batch=256):
    if not np.array_equal(original.shape, quantized.shape) or len(original.shape) != 2:
        raise ValueError(f"Weight shapes do not match for {original.name}")
    n_inputs, n_outputs = map(int, original.shape)
    result = np.zeros(n_inputs, dtype=np.float64)
    for start in range(0, n_outputs, rows_per_batch):
        end = min(start + rows_per_batch, n_outputs)
        source = dequantize(original.data[start:end], original.tensor_type).reshape(end - start, n_inputs)
        quant = dequantize(quantized.data[start:end], quantized.tensor_type).reshape(end - start, n_inputs)
        difference = np.subtract(quant, source, dtype=np.float64)
        if not np.isfinite(difference).all():
            raise ValueError(f"Nonfinite weights in {original.name}")
        result += np.einsum("ij,ij->j", difference, difference, dtype=np.float64)
    return result


def tensor_key(name):
    match = re.fullmatch(r"blk\.(\d+)\.([a-z_]+)\.weight", name)
    if not match or match[2] not in TYPES:
        raise ValueError(f"Unexpected tensor: {name}")
    return int(match[1]), TYPES.index(match[2])


def load_inputs(args, gguf):
    inventory = json.loads((args.input / "inventory.json").read_text(encoding="utf-8"))
    references = [item for item in inventory["matrices"] if item["model"] == args.reference.stem]
    if len(references) != 3 or {item["task"] for item in references} != set(TASKS):
        raise ValueError("Exactly one provenanced BF16 matrix per code/math/chat domain is required")
    source_hash = sha256(args.reference)
    vectors, provenance = {}, {}
    stats = pd.read_csv(args.input / "tensor_statistics.csv")
    stats = stats[stats["matrix"].isin({item["matrix"] for item in references})].copy()
    stats["tensor_type"] = stats["tensor"].map(lambda name: TYPES[tensor_key(name)[1]])
    controls = []
    for item in references:
        collection = item["collection"]
        if collection["model_sha256"] != source_hash or sha256(item["path"]) != item["sha256"]:
            raise ValueError("Source model or imatrix no longer matches recorded hashes")
        controls.append({key: collection[key] for key in (
            "context", "chunks", "batch_size", "parse_special", "tool_sha256", "tool_version")})
        reader = gguf.GGUFReader(item["path"])
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        task_vectors = {}
        task_stats = stats[stats["task"] == item["task"]]
        for row in task_stats.itertuples():
            moments = input_second_moment(tensors[row.tensor + ".in_sum2"].data,
                                          tensors[row.tensor + ".counts"].data)
            if not np.isclose(moments.sum(), row.energy, rtol=1e-5, atol=0.001):
                raise ValueError(f"Raw imatrix energy disagrees with CSV: {item['task']} / {row.tensor}")
            task_vectors[row.tensor] = moments
        vectors[item["task"]] = task_vectors
        provenance[item["task"]] = {"path": item["path"], "sha256": item["sha256"],
            "chunks": reader.fields["imatrix.chunk_count"].contents(),
            "chunk_size": reader.fields["imatrix.chunk_size"].contents(),
            "calibration_sha256": collection["calibration_sha256"]}
    if any(control != controls[0] for control in controls[1:]):
        raise ValueError("Reference collection settings differ between domains")
    if any(set(vectors[task]) != set(vectors[TASKS[0]]) for task in TASKS):
        raise ValueError("Reference domains have different tensor coverage")
    models = sorted(args.models_dir.glob("pure-q*.gguf"), key=lambda path: int(re.search(r"q(\d+)", path.stem)[1]))
    if not models:
        raise ValueError("No pure quantized model files found")
    provenance = {"reference": {"path": str(args.reference.resolve()), "sha256": source_hash},
                  "imatrices": provenance, "controls": controls[0],
                  "models": [{"name": model.stem, "path": str(model.resolve()), "sha256": sha256(model)} for model in models],
                  "gguf_reader_sha256": sha256(Path(gguf.__file__).parent / "gguf_reader.py"),
                  "dequantizer_sha256": sha256(Path(gguf.__file__).parent / "quants.py"),
                  "formula_version": 1}
    return stats, vectors, models, provenance


def save_figure(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=200)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)


def figure_energy(frame, folder):
    frame = relative_energy(frame)
    frame.to_csv(folder / "relative_activation_energy.csv", index=False)
    values = frame["relative_energy"].dropna()
    spread = max(0.05, float((values - 1).abs().max()))
    norm = TwoSlopeNorm(vmin=max(0, 1 - spread), vcenter=1, vmax=1 + spread)
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), sharey=True, layout="constrained")
    for ax, task in zip(axes, TASKS):
        grid = frame[frame["task"] == task].pivot(index="layer", columns="tensor_type", values="relative_energy").reindex(columns=TYPES).sort_index()
        image = ax.imshow(grid, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(TYPES)), LABELS, rotation=50, ha="right")
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.set_title(task.capitalize())
        for row, column in np.ndindex(grid.shape):
            value = grid.iloc[row, column]
            if pd.notna(value):
                color = "white" if abs(norm(value) - 0.5) > 0.3 else "black"
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)
    axes[0].set_ylabel("Transformer layer")
    fig.colorbar(image, ax=axes, label="Tensor energy / same tensor's mean energy across code, math, chat", shrink=0.8)
    fig.suptitle("1. Where do calibration datasets emphasize different dimensions?\nBF16 reference; 1 = cross-dataset average for that exact tensor")
    save_figure(fig, folder, "01_relative_activation_energy")


def figure_concentration(frame, folder):
    fig, axes = plt.subplots(2, len(TYPES), figsize=(23, 8), sharex=True, sharey="row", layout="constrained")
    colors = dict(zip(TASKS, ("#0072B2", "#D55E00", "#009E73")))
    for column, (tensor_type, label) in enumerate(zip(TYPES, LABELS)):
        subset = frame[frame["tensor_type"] == tensor_type]
        for task in TASKS:
            profile = subset[subset["task"] == task].sort_values("layer")
            for row, metric in enumerate(("entropy_pct", "kurtosis")):
                axes[row, column].plot(profile["layer"], profile[metric], marker="o", markersize=3,
                                       label=task, color=colors[task], linewidth=1.4)
        axes[0, column].set_title(label)
        axes[0, column].set_ylim(0, 100)
        axes[1, column].set_yscale("symlog", linthresh=1)
        axes[1, column].set_xlabel("Layer")
        for row in range(2):
            axes[row, column].grid(alpha=0.2)
            axes[row, column].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=5))
    axes[0, 0].set_ylabel("Normalized energy entropy (%)")
    axes[1, 0].set_ylabel("Mean-activation excess kurtosis\n(symmetric log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("2. Importance concentration and activation outliers\nEntropy describes energy distribution; existing kurtosis describes mean activations across dimensions")
    frame[["task", "layer", "tensor", "tensor_type", "entropy_pct", "kurtosis"]].to_csv(
        folder / "importance_concentration.csv", index=False)
    save_figure(fig, folder, "02_importance_concentration_and_outliers")


def calculate_errors(args, gguf, vectors, models, provenance):
    output = args.output / "activation_weighted_errors.csv"
    manifest = args.output / "weight_error_provenance.json"
    if output.exists() and manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("inputs") == provenance and saved.get("csv_sha256") == sha256(output):
            print("Reuse verified activation-weighted errors", flush=True)
            return pd.read_csv(output)
    source_reader = gguf.GGUFReader(args.reference)
    sources = {tensor.name: tensor for tensor in source_reader.tensors}
    readers = {path.stem: gguf.GGUFReader(path) for path in models}
    quantized = {model: {tensor.name: tensor for tensor in reader.tensors} for model, reader in readers.items()}
    names = sorted(vectors[TASKS[0]], key=tensor_key)
    rows = []
    for number, name in enumerate(names, 1):
        original = sources[name]
        for model, tensors in quantized.items():
            quant = tensors[name]
            error = column_squared_error(original, quant, gguf.dequantize, args.row_batch)
            for task in TASKS:
                rows.append({"task": task, "model": model, "precision": model.removeprefix("pure-").upper(),
                             "tensor": name, "layer": tensor_key(name)[0], "tensor_type": TYPES[tensor_key(name)[1]],
                             "n_inputs": int(original.shape[0]), "n_outputs": int(original.shape[1]),
                             "weight_type": quant.tensor_type.name,
                             "weighted_error": weighted_error(vectors[task][name], error)})
        if number % 7 == 0:
            print(f"Weighted errors: {number}/{len(names)} tensors ({name})", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    manifest.write_text(json.dumps({"inputs": provenance, "csv_sha256": sha256(output)}, indent=2), encoding="utf-8")
    return frame


def figure_errors(frame, folder):
    names = sorted(frame["tensor"].unique(), key=tensor_key)
    precisions = sorted(frame["precision"].unique(), key=lambda value: int(value[1:]))
    positive = frame.loc[frame["weighted_error"] > 0, "weighted_error"]
    if positive.empty:
        raise ValueError("All weighted errors are zero; a logarithmic heatmap is undefined")
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    fig, axes = plt.subplots(1, 3, figsize=(18, max(14, len(names) * 0.19)), sharey=True, layout="constrained")
    for ax, task in zip(axes, TASKS):
        grid = frame[frame["task"] == task].pivot(index="tensor", columns="precision", values="weighted_error").reindex(index=names, columns=precisions)
        image = ax.imshow(np.ma.masked_less_equal(grid.to_numpy(), 0), aspect="auto", cmap="magma", norm=norm)
        ax.set_xticks(range(len(precisions)), precisions)
        ax.set_yticks(range(len(names)), [f"L{tensor_key(name)[0]:02d} {TYPES[tensor_key(name)[1]]}" for name in names], fontsize=6)
        ax.set_title(task.capitalize() + " imatrix")
        ax.set_xlabel("Weight precision")
        for boundary in range(len(TYPES) - 1, len(names) - 1, len(TYPES)):
            ax.axhline(boundary + 0.5, color="white", alpha=0.45, linewidth=0.6)
    axes[0].set_ylabel("Tensor (layer and type; every measured tensor retained)")
    fig.colorbar(image, ax=axes, label="Activation-weighted squared weight error (logarithmic color scale)", shrink=0.6)
    fig.suptitle("3. Where does low precision damage important input dimensions?\nApproximate local error measure; BF16 activation weights; not a direct measure of final performance\nQ7 is unavailable; columns show the six existing pure quantizations")
    save_figure(fig, folder, "03_activation_weighted_quantization_error")


def block_averages(stats, errors):
    """Equal tensor weights within each block; retain domain and precision."""
    for frame, keys in ((stats, ["task", "layer"]),
                        (errors, ["task", "model", "precision", "layer"])):
        coverage = frame.groupby(keys)["tensor_type"].agg(list)
        if not coverage.map(lambda types: len(types) == len(TYPES) and set(types) == set(TYPES)).all():
            raise ValueError("Block averages require exactly one of each of the seven measured tensor types")
    blocks = stats.groupby(["task", "layer"], as_index=False).agg(
        mean_energy=("energy", "mean"), mean_entropy_pct=("entropy_pct", "mean"),
        mean_kurtosis=("kurtosis", "mean"), n_tensors=("tensor_type", "size"))
    counts = blocks.groupby("layer")["task"].agg(set)
    if not counts.map(lambda tasks: tasks == set(TASKS)).all():
        raise ValueError("Every block must have code, math and chat statistics")
    denominator = blocks.groupby("layer")["mean_energy"].transform("mean")
    blocks["relative_energy"] = blocks["mean_energy"].div(denominator.where(denominator > 0))
    block_errors = errors.groupby(["task", "model", "precision", "layer"], as_index=False).agg(
        mean_weighted_error=("weighted_error", "mean"), n_tensors=("tensor_type", "size"))
    return blocks, block_errors


def figure_blocks(stats, errors, folder):
    folder = folder / "by block"
    folder.mkdir(parents=True, exist_ok=True)
    blocks, block_errors = block_averages(stats, errors)
    blocks.to_csv(folder / "block_statistics.csv", index=False)
    block_errors.to_csv(folder / "block_weighted_errors.csv", index=False)

    grid = blocks.pivot(index="layer", columns="task", values="relative_energy").reindex(columns=TASKS).sort_index()
    spread = max(0.05, float((grid - 1).abs().max().max()))
    norm = TwoSlopeNorm(vmin=max(0, 1 - spread), vcenter=1, vmax=1 + spread)
    fig, ax = plt.subplots(figsize=(7, 8), layout="constrained")
    image = ax.imshow(grid, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(TASKS)), [task.capitalize() for task in TASKS])
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set(xlabel="Calibration dataset", ylabel="Transformer block",
           title="1. Relative activation energy by block\nMean across seven tensor types, then normalized across datasets")
    for row, column in np.ndindex(grid.shape):
        value = grid.iloc[row, column]
        if pd.notna(value):
            color = "white" if abs(norm(value) - 0.5) > 0.3 else "black"
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, label="Block mean energy / cross-dataset average for the same block")
    save_figure(fig, folder, "01_relative_activation_energy_by_block")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")
    colors = dict(zip(TASKS, ("#0072B2", "#D55E00", "#009E73")))
    for task in TASKS:
        profile = blocks[blocks["task"] == task].sort_values("layer")
        for ax, metric in zip(axes, ("mean_entropy_pct", "mean_kurtosis")):
            ax.plot(profile["layer"], profile[metric], marker="o", label=task, color=colors[task])
    axes[0].set_ylabel("Mean tensor normalized entropy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_ylabel("Mean tensor excess kurtosis\n(symmetric log scale)")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_xlabel("Transformer block")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].legend(title="Calibration dataset")
    fig.suptitle("2. Importance concentration and outliers by block\nArithmetic means of seven tensor statistics; not pooled entropy or kurtosis")
    save_figure(fig, folder, "02_importance_concentration_and_outliers_by_block")

    positive = block_errors.loc[block_errors["mean_weighted_error"] > 0, "mean_weighted_error"]
    if positive.empty:
        raise ValueError("Block weighted errors must have positive values for a logarithmic heatmap")
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    precisions = sorted(block_errors["precision"].unique(), key=lambda value: int(value[1:]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharey=True, layout="constrained")
    for ax, task in zip(axes, TASKS):
        grid = block_errors[block_errors["task"] == task].pivot(
            index="layer", columns="precision", values="mean_weighted_error").reindex(columns=precisions).sort_index()
        image = ax.imshow(np.ma.masked_less_equal(grid.to_numpy(), 0), aspect="auto", cmap="magma", norm=norm)
        ax.set_xticks(range(len(precisions)), precisions)
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.set_title(task.capitalize() + " imatrix")
        ax.set_xlabel("Weight precision")
        for row, column in np.ndindex(grid.shape):
            value = grid.iloc[row, column]
            if pd.notna(value) and value > 0:
                color = "black" if norm(value) > 0.65 else "white"
                ax.text(column, row, f"{value:.2g}", ha="center", va="center", color=color, fontsize=7)
    axes[0].set_ylabel("Transformer block")
    fig.colorbar(image, ax=axes, label="Mean activation-weighted error per tensor (logarithmic color scale)", shrink=0.8)
    fig.suptitle("3. Activation-weighted quantization error by block\nEqual mean of seven tensor errors; approximate local measure, not final performance")
    save_figure(fig, folder, "03_activation_weighted_quantization_error_by_block")
    (folder / "README.txt").write_text(
        "Block-level versions of the three importance figures. Original tensor plots remain in the parent folder.\n\n"
        "Every block average gives equal weight to its seven measured tensor types: " + ", ".join(TYPES) + ".\n"
        "Code, math and chat are kept separate for direct comparison.\n"
        "Figure 1 averages raw tensor energies within a block, then divides by that same block's mean across datasets.\n"
        "It is not an average of the previously normalized per-tensor ratios.\n"
        "Figure 2 averages each tensor's entropy and kurtosis statistics. These are NOT entropy or kurtosis\n"
        "computed from a pooled block activation distribution. Kurtosis concerns dimension-wise mean activations.\n"
        "Figure 3 averages the seven already-calculated tensor weighted errors; it is not a sum or a per-weight mean.\n"
        "This remains an approximate local error measure, not a direct measure of final performance.\n"
        "Shared input activations can appear in multiple tensor types. Block aggregates are not residual-output energy.\n"
        "Only the measured projection tensors are included; embeddings, norms and uncollected output tensors are excluded.\n"
        "CSVs include n_tensors so the seven-tensor averaging denominator can be checked.\n",
        encoding="utf-8")


def write_analysis(stats, errors, provenance, folder):
    relative = relative_energy(stats)
    disagreement = relative.groupby("tensor")["relative_energy"].agg(["min", "max"])
    disagreement["spread"] = disagreement["max"] - disagreement["min"]
    lines = ["Where datasets disagree -> how importance is distributed -> where quantization damages important dimensions", "",
        "All figures use the BF16 source model's code, math and chat imatrices. Layers are retained.",
        "Figure 1: energy divided by the arithmetic mean across the three domains for the SAME exact tensor.",
        "Figure 2: normalized entropy is computed from the energy distribution across input dimensions.",
        "The existing CLI excess kurtosis is computed from mean input activations across dimensions; it is not",
        "kurtosis of per-token activations or of the imatrix importance values. Large values indicate outlying dimension means.",
        "Figure 3: weighted_error = sum_j E[x_j^2] * sum_i (Wq[i,j] - W[i,j])^2.",
        "Weights W are the original BF16 weights converted to float32; Wq are dequantized stored quantized weights.",
        "Raw imatrix input squared sums are divided by their input counts, not by chunk count.",
        "This is an approximate local error measure, not a direct measure of final performance.",
        "It uses a diagonal input second-moment approximation and omits cross-dimension terms and downstream effects.",
        "Absolute error sums depend on tensor size and output scale, so high scores identify candidates for testing, not proven precision needs.",
        "Energy itself does not establish sensitivity. Compare weighted error across precisions for the same tensor.",
        "Full-file domains have unequal calibration lengths; per-token normalization removes count scaling but not sampling uncertainty.",
        "Only tensors measured by the imatrices are included; embeddings, norms and uncollected output tensors are excluded.", "",
        "Calibration coverage:"]
    for task in TASKS:
        item = provenance["imatrices"][task]
        lines.append(f"  {task}: {item['chunks']} chunks of {item['chunk_size']} tokens.")
    lines += ["", "Largest relative energy disagreements (maximum minus minimum dataset ratio):"]
    for name, row in disagreement.nlargest(8, "spread").iterrows():
        ratios = relative[relative["tensor"] == name].set_index("task")["relative_energy"]
        lines.append(f"  {name}: " + ", ".join(f"{task}={ratios[task]:.3f}" for task in TASKS))
    lines += ["", "Largest activation-weighted errors at the lowest available precision:"]
    lowest = min(errors["precision"].unique(), key=lambda value: int(value[1:]))
    for task in TASKS:
        top = errors[(errors["task"] == task) & (errors["precision"] == lowest)].nlargest(5, "weighted_error")
        lines.append(f"  {task} / {lowest}:")
        lines.extend(f"    {row.tensor}: {row.weighted_error:.6g}" for row in top.itertuples())
    lines += ["", "Three figures are provided as PNG and PDF. CSV files preserve the plotted data.",
              "No independent calibration repeats were run, so these comparisons do not establish statistical significance."]
    (folder / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "imatrix_full_pure_sweep")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "important plots")
    parser.add_argument("--reference", type=Path, default=ROOT / "models" / "Llama-3.2-1B-Instruct-BF16.gguf")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--gguf-python", type=Path, default=Path(r"C:\llama.cpp\gguf-py"))
    parser.add_argument("--row-batch", type=int, default=256)
    parser.add_argument("--precision-only", action="store_true", help="Regenerate only task-averaged precision comparisons")
    args = parser.parse_args()
    if args.row_batch < 1:
        parser.error("--row-batch must be positive")
    if args.gguf_python.is_dir():
        sys.path.insert(0, str(args.gguf_python))
    gguf = importlib.import_module("gguf")
    args.output.mkdir(parents=True, exist_ok=True)
    stats, vectors, models, provenance = load_inputs(args, gguf)
    if args.precision_only:
        try:
            from src.important_precision_plots import figure_precision_comparisons
        except ModuleNotFoundError:
            from important_precision_plots import figure_precision_comparisons
        calculate_errors(args, gguf, vectors, models, provenance)
        figure_precision_comparisons(args.input, args.output, args.reference, vectors, gguf, provenance)
        return
    figure_energy(stats, args.output)
    figure_concentration(stats, args.output)
    print("Figures 1 and 2 saved; computing real weight errors for Figure 3", flush=True)
    errors = calculate_errors(args, gguf, vectors, models, provenance)
    figure_errors(errors, args.output)
    figure_blocks(stats, errors, args.output)
    try:
        from src.important_precision_plots import figure_precision_comparisons
    except ModuleNotFoundError:
        from important_precision_plots import figure_precision_comparisons
    figure_precision_comparisons(args.input, args.output, args.reference, vectors, gguf, provenance)
    write_analysis(stats, errors, provenance, args.output)
    print(f"Saved all three figures, plotted data and methods to {args.output}", flush=True)


if __name__ == "__main__":
    main()
