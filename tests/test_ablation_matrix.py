from __future__ import annotations

from microreasoner.ablation.runner import build_experiment_specs


def test_ablation_matrix_contains_required_families() -> None:
    specs = build_experiment_specs()
    ids = {item.experiment_id for item in specs}
    families = {item.family for item in specs}
    assert "exp_sft_only" in ids
    assert "exp_loss_dr" in ids
    assert "exp_loss_grpo" in ids
    assert "exp_scale_group" in ids
    assert "exp_schema_050" in ids
    assert "exp_rs_sft" in ids
    assert {"baseline", "loss", "scale_rewards", "schema_weight", "alternative"} <= families

