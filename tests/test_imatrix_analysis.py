import argparse
import json
import math

import pytest

from src import imatrix_analysis as analysis


CURRENT = """
Computing tensor statistics for test.gguf (1 tensors)
 Layer | Tensor | Mean StdDev Skew Kurt H Norm | sum Gain | L2 Dist PCC Cov
 0 | blk.0.attn_q.weight | -0.1 0.2 0.3 4.0 86.85% | 100.7400 nan | nan nan nan
Computing layer statistics for test.gguf (1 layers)
 Layer | sum Gain | L2 Dist CosSim PCC Cov
 0 | 430.6851 nan | nan nan nan nan
"""


def test_parse_current_tables_preserves_undefined_values():
    records = analysis.parse_statistics(CURRENT)
    assert records["tensor"][0]["entropy_pct"] == 86.85
    assert records["tensor"][0]["tensor"] == "blk.0.attn_q.weight"
    assert records["layer"][0]["energy"] == 430.6851
    assert math.isnan(records["layer"][0]["gain"])


def test_parse_legacy_tables_without_l2():
    text = """
Computing tensor statistics for old.dat (1 tensors)
Layer | Tensor | Mean StdDev Skew Kurt H Norm | sum Gain | PCC Cov
1 | blk.1.attn_q.weight | 0.1 0.2 0.3 4.0 50% | 10 2 | 0.9 0.1
Computing layer statistics for old.dat (1 layers)
Layer | sum Gain | CosSim PCC Cov
1 | 10 2 | 0.8 0.9 0.1
"""
    records = analysis.parse_statistics(text)
    assert records["layer"][0]["cosine_similarity"] == 0.8
    assert "l2_distance" not in records["tensor"][0]


def test_reject_truncated_output():
    with pytest.raises(ValueError, match="Missing statistics"):
        analysis.parse_statistics(CURRENT.split("Computing layer")[0])
    with pytest.raises(ValueError, match="Unrecognized"):
        analysis.parse_statistics(CURRENT.replace("100.7400 nan", "100.7400"))


def test_collection_reuses_only_intact_matrix_and_separates_settings(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    tool = tmp_path / "tool.exe"
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "math.txt").write_text("calibration")
    model.write_bytes(b"model")
    tool.write_bytes(b"tool")
    args = argparse.Namespace(model=[str(model)], calibration_dir=text_dir, task=None,
                              chunks=2, context=512, gpu_layers="0", matrix_dir=tmp_path / "matrices",
                              output=tmp_path / "reports")
    calls = []

    def fake_subprocess(command, **kwargs):
        calls.append(command)
        output = command[command.index("-o") + 1]
        analysis.Path(output).write_bytes(b"matrix")

    monkeypatch.setattr(analysis, "run", lambda command: ("version", ""))
    monkeypatch.setattr(analysis.subprocess, "run", fake_subprocess)
    analysis.collect(args, tool)
    analysis.collect(args, tool)
    assert len(calls) == 1
    matrix = next(args.matrix_dir.rglob("*.gguf"))
    metadata = json.loads(matrix.with_suffix(".json").read_text())
    assert metadata["calibration_sha256"] == analysis.digest(text_dir / "math.txt")
    matrix.write_bytes(b"corrupted")
    analysis.collect(args, tool)
    assert len(calls) == 2
    args.context = 1024
    analysis.collect(args, tool)
    assert len(list(args.matrix_dir.rglob("*.gguf"))) == 2
    args.chunks = -1
    analysis.collect(args, tool)
    assert calls[-1][calls[-1].index("--chunks") + 1] == "-1"
    assert len(list(args.matrix_dir.rglob("*.gguf"))) == 3
