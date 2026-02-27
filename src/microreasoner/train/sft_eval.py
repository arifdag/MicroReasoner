from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from microreasoner.eval.parser import parse_response
from microreasoner.eval.verifier import build_verifier
from microreasoner.train.sft_data import SFTRecordItem


@dataclass(frozen=True)
class SFTMetrics:
    schema_compliance: float
    parser_failure_rate: float
    greedy_pass_at_1: float
    sampled_pass_at_1: float
    think_tokens_mean: float
    think_tokens_p95: float
    eval_size: int

    def to_metrics_json(self, benchmark_name: str = "sft_val") -> dict[str, Any]:
        return {
            "accuracy": {
                benchmark_name: {
                    "greedy_pass_at_1": self.greedy_pass_at_1,
                    "sampled_pass_at_1": self.sampled_pass_at_1,
                }
            },
            "schema": {"compliance_rate": self.schema_compliance},
            "parser": {"extraction_failure_rate": self.parser_failure_rate},
            "length": {
                "think_tokens": {
                    "mean": self.think_tokens_mean,
                    "p95": self.think_tokens_p95,
                }
            },
        }


def _p95(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, int(0.95 * len(ordered)) - 1)
    idx = min(idx, len(ordered) - 1)
    return float(ordered[idx])


def _count_tokens(text: str | None) -> int:
    if text is None or text.strip() == "":
        return 0
    return len(text.split())


def evaluate_fixture(records: list[SFTRecordItem]) -> SFTMetrics:
    verifier = build_verifier("simple")
    schema_ok = 0
    parse_fail = 0
    greedy_correct = 0
    sampled_correct = 0
    think_counts: list[int] = []

    for item in records:
        predicted = item.target_response
        parsed = parse_response(predicted, strict_boxed_only=True)
        if parsed.schema_ok:
            schema_ok += 1
        if not parsed.parse_ok:
            parse_fail += 1
        think_counts.append(_count_tokens(parsed.think_text))
        if parsed.boxed_answer is not None and item.gold_answer is not None:
            is_correct = verifier.verify(parsed.boxed_answer, item.gold_answer).correct
        else:
            is_correct = False
        if is_correct:
            greedy_correct += 1
            sampled_correct += 1

    total = max(1, len(records))
    return SFTMetrics(
        schema_compliance=schema_ok / total,
        parser_failure_rate=parse_fail / total,
        greedy_pass_at_1=greedy_correct / total,
        sampled_pass_at_1=sampled_correct / total,
        think_tokens_mean=float(mean(think_counts)) if think_counts else 0.0,
        think_tokens_p95=_p95(think_counts),
        eval_size=len(records),
    )


def evaluate_transformers(
    *,
    records: list[SFTRecordItem],
    model_bundle: Any,
    max_new_tokens: int,
    sampled_temperature: float,
    sampled_top_p: float,
    sampled_n: int,
) -> SFTMetrics:
    # Fall back to fixture semantics if record count is empty.
    if not records:
        return evaluate_fixture(records)

    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    torch = model_bundle.stack["torch"]
    verifier = build_verifier("simple")

    schema_ok = 0
    parse_fail = 0
    greedy_correct = 0
    sampled_correct = 0
    think_counts: list[int] = []

    for item in records:
        inputs = tokenizer(item.prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            greedy_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        greedy_text = tokenizer.decode(greedy_ids[0][prompt_len:], skip_special_tokens=True)
        parsed = parse_response(greedy_text, strict_boxed_only=True)
        if parsed.schema_ok:
            schema_ok += 1
        if not parsed.parse_ok:
            parse_fail += 1
        think_counts.append(_count_tokens(parsed.think_text))
        greedy_is_correct = (
            parsed.boxed_answer is not None
            and item.gold_answer is not None
            and verifier.verify(parsed.boxed_answer, item.gold_answer).correct
        )
        if greedy_is_correct:
            greedy_correct += 1

        sample_correct = False
        if sampled_n <= 1:
            sample_correct = greedy_is_correct
        else:
            with torch.no_grad():
                sample_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=sampled_temperature,
                    top_p=sampled_top_p,
                    num_return_sequences=sampled_n,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            if sample_ids.dim() == 1:
                sample_ids = sample_ids.unsqueeze(0)
            for row in sample_ids:
                sample_text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
                sample_parsed = parse_response(sample_text, strict_boxed_only=True)
                if (
                    sample_parsed.boxed_answer is not None
                    and item.gold_answer is not None
                    and verifier.verify(sample_parsed.boxed_answer, item.gold_answer).correct
                ):
                    sample_correct = True
                    break
        if sample_correct:
            sampled_correct += 1

    total = max(1, len(records))
    return SFTMetrics(
        schema_compliance=schema_ok / total,
        parser_failure_rate=parse_fail / total,
        greedy_pass_at_1=greedy_correct / total,
        sampled_pass_at_1=sampled_correct / total,
        think_tokens_mean=float(mean(think_counts)) if think_counts else 0.0,
        think_tokens_p95=_p95(think_counts),
        eval_size=len(records),
    )

