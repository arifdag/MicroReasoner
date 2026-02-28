from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_colab_smoke_notebook_contains_smoke_pipeline_steps() -> None:
    notebook = _repo_root() / "notebooks" / "colab_smoke_microreasoner.ipynb"
    assert notebook.exists()

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert int(payload.get("nbformat", 0)) >= 4

    cells = payload.get("cells", [])
    assert isinstance(cells, list)
    assert len(cells) >= 5

    full_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict)
    ).lower()
    assert "data build-sft" in full_text
    assert "data build-rl" in full_text
    assert "train sft" in full_text
    assert "train grpo" in full_text
    assert "validate-run" in full_text
    assert "scripts/run_final_evaluation.py" in full_text
    assert "scripts/run_demo_compare.py" in full_text
