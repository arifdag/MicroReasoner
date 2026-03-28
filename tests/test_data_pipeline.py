from __future__ import annotations

import json
from pathlib import Path

from microreasoner.cli.main import main
from microreasoner.data.adapters import load_sources
from microreasoner.data.build_rl_dataset import build_rl_dataset
from microreasoner.data.build_sft_dataset import build_sft_dataset
from microreasoner.data.normalize import normalize_examples
from microreasoner.data.split import split_examples
from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import repo_root


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_config_file(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "raw"
    _write_jsonl(
        source_dir / "s1.jsonl",
        [
            {
                "id": "ex1",
                "source_name": "s1",
                "benchmark": "gsm8k",
                "question": "What is 1+1?",
                "think": "Add the two numbers",
                "answer_boxed": "2",
                "metadata": {"difficulty": "easy"},
            },
            {
                "id": "ex2",
                "source_name": "s1",
                "benchmark": "math",
                "question": "Compute 2+2",
                "think": "Add two and two",
                "answer_boxed": "4",
            },
            {
                "id": "ex3",
                "source_name": "s1",
                "benchmark": "math",
                "question": "Compute 2+2",
                "think": "Duplicate should be removed",
                "answer_boxed": "4",
            },
            {
                "id": "ex4",
                "source_name": "s1",
                "benchmark": "gsm8k",
                "question": "Bad example",
                "think": "Missing boxed answer",
                "answer_boxed": "",
            },
        ],
    )

    config_path = tmp_path / "data_config.yaml"
    _write_text(
        config_path,
        "\n".join(
            [
                "data_pipeline:",
                "  input_sources:",
                "    - name: source_one",
                "      adapter: canonical_jsonl",
                "      path: s1.jsonl",
                "  split:",
                "    strategy: hash",
                "    train_ratio: 0.5",
                "    val_ratio: 0.5",
                "    seed: 123",
                "  filters:",
                "    min_think_tokens: 1",
                "    max_think_tokens: 128",
                "    require_single_boxed_answer: true",
                "    drop_duplicates: true",
                "  outputs:",
                "    root_dir: artifacts/datasets",
                "    write_rejects: true",
                "    compression: none",
                "  rl:",
                "    curriculum_rules:",
                "      - name: gsm8k_heavy",
                "        benchmarks: [gsm8k]",
                "      - name: mixed",
                "        benchmarks: [gsm8k, math]",
                "    benchmark_mix_targets:",
                "      gsm8k: 0.7",
                "      math: 0.3",
            ]
        ),
    )
    return config_path, source_dir


def test_load_sources_and_split_are_deterministic(tmp_path: Path) -> None:
    config_path, source_dir = _build_config_file(tmp_path)
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    examples = load_sources(config.data_pipeline.input_sources, source_dir=source_dir)
    normalized = normalize_examples(examples)
    first_train, first_val = split_examples(normalized, config.data_pipeline.split)
    second_train, second_val = split_examples(normalized, config.data_pipeline.split)
    assert [item.example_id for item in first_train] == [item.example_id for item in second_train]
    assert [item.example_id for item in first_val] == [item.example_id for item in second_val]


def test_build_sft_dataset_is_deterministic_and_manifested(tmp_path: Path) -> None:
    config_path, source_dir = _build_config_file(tmp_path)
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    first = build_sft_dataset(config=config, output_root=out_a, source_dir=source_dir)
    second = build_sft_dataset(config=config, output_root=out_b, source_dir=source_dir)
    assert first.dataset_id == second.dataset_id
    assert first.train_count + first.val_count == 2
    assert first.reject_count == 2

    manifest = _read_json(Path(first.manifest_path))
    assert manifest["dataset_type"] == "sft"
    assert manifest["split_counts"]["train"] + manifest["split_counts"]["val"] == 2
    assert "duplicate" in manifest["reject_stats"]
    assert "missing_boxed_answer" in manifest["reject_stats"]


def test_build_rl_dataset_outputs_prompt_records(tmp_path: Path) -> None:
    config_path, source_dir = _build_config_file(tmp_path)
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    out_root = tmp_path / "out_rl"

    result = build_rl_dataset(config=config, output_root=out_root, source_dir=source_dir)
    manifest = _read_json(Path(result.manifest_path))
    assert manifest["dataset_type"] == "rl"
    assert result.train_count + result.val_count == 2

    train_path = Path(result.output_dir) / "train_prompts.jsonl"
    val_path = Path(result.output_dir) / "val_prompts.jsonl"
    records: list[dict] = []
    for path in (train_path, val_path):
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    assert records
    sample = records[0]
    assert "prompt" in sample
    assert "gold_answer" in sample
    assert "curriculum_stage" in sample
    assert "difficulty_tag" in sample
    assert "Respond using the exact format" in sample["prompt"]
    assert "<answer>\\boxed{...}</answer>" in sample["prompt"]


def test_data_cli_build_and_inspect(tmp_path: Path, monkeypatch) -> None:
    config_path, source_dir = _build_config_file(tmp_path)
    dataset_root = tmp_path / "datasets_out"

    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "data",
            "build-sft",
            "--config",
            str(config_path),
            "--run-id",
            "data-build-run",
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(dataset_root),
        ]
    )
    assert code == 0

    summary_path = tmp_path / "artifacts" / "runs" / "data-build-run" / "summary.json"
    summary = _read_json(summary_path)
    manifest_path = Path(summary["artifacts"]["dataset_manifest_path"])
    assert manifest_path.exists()

    inspect_code = main(["data", "inspect", "--dataset-manifest", str(manifest_path)])
    assert inspect_code == 0
