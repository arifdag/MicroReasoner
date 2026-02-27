from __future__ import annotations

from pathlib import Path

from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import repo_root
from microreasoner.train.sft_model import select_sft_mode


def test_train_sft_config_parses_from_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    assert config.train_sft.mode == "auto"
    assert config.train_sft.lora.r > 0
    assert config.train_sft.batch.max_seq_len >= 512


def test_mode_selection_respects_explicit_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train_sft:\n  mode: lora\n", encoding="utf-8")
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    selected = select_sft_mode(config)
    assert selected.selected_mode == "lora"
    assert selected.reason == "explicit_mode"


def test_mode_selection_auto_falls_back_to_qlora_when_no_cuda(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train_sft:\n  mode: auto\n", encoding="utf-8")
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)

    monkeypatch.setattr("microreasoner.train.sft_model._cuda_total_vram_gb", lambda: None)
    selected = select_sft_mode(config)
    assert selected.selected_mode == "qlora"

