from __future__ import annotations

from pathlib import Path

import pytest

from microreasoner.runtime.configuration import RuntimeConfigError, resolve_config
from microreasoner.runtime.context import repo_root


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resolve_config_includes_train_grpo_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_text(config_path, "{}\n")
    resolved = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    assert resolved.train_grpo.algo.loss_type == "dr_grpo"
    assert resolved.train_grpo.algo.group_size > 0


def test_resolve_config_rejects_invalid_grpo_scale_rewards(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_text(
        config_path,
        "\n".join(
            [
                "train_grpo:",
                "  algo:",
                "    scale_rewards: bad_mode",
                "",
            ]
        ),
    )
    with pytest.raises(RuntimeConfigError):
        resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
