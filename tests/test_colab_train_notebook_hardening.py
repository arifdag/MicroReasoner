from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_colab_train_notebook_requires_pinned_ref_and_versions() -> None:
    notebook = _repo_root() / "notebooks" / "colab_train_microreasoner.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))

    full_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if isinstance(cell, dict)
    )
    assert 'MICROREASONER_REF' in full_text
    assert 'Set MICROREASONER_REF to an exact git commit before running paid training.' in full_text
    assert '"transformers==4.46.0"' in full_text
    assert '"trl==0.9.6"' in full_text
    assert '"datasets==2.20.0"' in full_text
    assert '"numpy==1.26.4"' in full_text
    assert '"pandas==2.2.2"' in full_text
    assert '"pyarrow==16.1.0"' in full_text
    assert '"bitsandbytes":' not in full_text
    assert '"bitsandbytes==' not in full_text
    assert 'pip", "uninstall", "-y", "bitsandbytes"' in full_text
    assert 'raise SystemExit(' in full_text
    assert 'Restart the Colab runtime once, then rerun the notebook from the top.' in full_text
    assert '!subprocess.check_call(cmd)' not in full_text
    assert '!run([sys.executable' not in full_text


def test_colab_train_notebook_does_not_relax_full_run_quality_gates() -> None:
    notebook = _repo_root() / "notebooks" / "colab_train_microreasoner.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))

    full_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if isinstance(cell, dict)
    )
    assert "train_sft.gates.schema_min=0.0" not in full_text
    assert "train_grpo.gates.min_schema_compliance_rate=0.0" not in full_text
    assert "train_grpo.gates.max_parser_failure_rate=1.0" not in full_text
    assert "train_grpo.gates.min_reward_std=0.0" not in full_text
