from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microreasoner.prompting import render_generation_prompt
from microreasoner.eval.types import EvalExample


class InferenceError(RuntimeError):
    """Raised when generation fails."""


@dataclass(frozen=True)
class InferenceSettings:
    backend: str
    max_new_tokens: int
    device: str
    dtype: str
    greedy_temperature: float
    sampled_temperature: float
    sampled_top_p: float
    sampled_n: int
    seed: int


class InferenceEngine:
    def generate_greedy(self, prompt: str, example: EvalExample) -> str:
        raise NotImplementedError

    def generate_sampled(self, prompt: str, example: EvalExample) -> list[str]:
        raise NotImplementedError


def _safe_generate(model: Any, kwargs: dict[str, Any]) -> Any:
    try:
        return model.generate(**kwargs)
    except ValueError as exc:
        text = str(exc)
        if (
            "model_kwargs" in text
            and "generator" in text
            and "generator" in kwargs
        ):
            retry = dict(kwargs)
            retry.pop("generator", None)
            return model.generate(**retry)
        raise


class FixtureInferenceEngine(InferenceEngine):
    def __init__(self, settings: InferenceSettings) -> None:
        self._settings = settings
        self._rng = random.Random(settings.seed)

    def generate_greedy(self, prompt: str, example: EvalExample) -> str:
        del prompt
        if example.mock_greedy_response is not None:
            return example.mock_greedy_response
        if example.mock_sampled_responses:
            return example.mock_sampled_responses[0]
        return "<think>no fixture</think><answer>\\boxed{0}</answer>"

    def generate_sampled(self, prompt: str, example: EvalExample) -> list[str]:
        del prompt
        if example.mock_sampled_responses:
            pool = list(example.mock_sampled_responses)
        elif example.mock_greedy_response is not None:
            pool = [example.mock_greedy_response]
        else:
            pool = ["<think>no fixture</think><answer>\\boxed{0}</answer>"]

        if len(pool) >= self._settings.sampled_n:
            return pool[: self._settings.sampled_n]

        out: list[str] = []
        for _ in range(self._settings.sampled_n):
            out.append(self._rng.choice(pool))
        return out


class TransformersInferenceEngine(InferenceEngine):
    def __init__(self, checkpoint: Path, settings: InferenceSettings) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise InferenceError(
                "transformers backend requires 'torch' and 'transformers' packages"
            ) from exc

        self._torch = torch
        device = "cuda" if settings.device == "auto" and torch.cuda.is_available() else settings.device
        self._device = device

        if settings.dtype == "auto":
            dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            if settings.dtype not in dtype_map:
                raise InferenceError(f"Unsupported dtype setting: {settings.dtype}")
            dtype = dtype_map[settings.dtype]

        self._tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        self._model = AutoModelForCausalLM.from_pretrained(str(checkpoint), torch_dtype=dtype)
        self._model.to(device)
        self._model.eval()
        self._settings = settings

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _generate(self, prompt: str, *, do_sample: bool, temperature: float, top_p: float, n: int) -> list[str]:
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        rendered_prompt = render_generation_prompt(tokenizer, prompt)
        inputs = tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])

        generator = None
        if do_sample:
            generator = torch.Generator(device=self._device)
            generator.manual_seed(self._settings.seed)

        generate_kwargs: dict[str, Any] = {
            **inputs,
            "do_sample": do_sample,
            "temperature": (temperature if do_sample else None),
            "top_p": (top_p if do_sample else None),
            "num_return_sequences": n,
            "max_new_tokens": self._settings.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if generator is not None:
            generate_kwargs["generator"] = generator

        with torch.no_grad():
            output_ids = _safe_generate(model, generate_kwargs)

        if output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)

        generations: list[str] = []
        for row in output_ids:
            completion_ids = row[prompt_len:]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            generations.append(text)
        return generations

    def generate_greedy(self, prompt: str, example: EvalExample) -> str:
        del example
        return self._generate(
            prompt,
            do_sample=False,
            temperature=self._settings.greedy_temperature,
            top_p=self._settings.sampled_top_p,
            n=1,
        )[0]

    def generate_sampled(self, prompt: str, example: EvalExample) -> list[str]:
        del example
        return self._generate(
            prompt,
            do_sample=True,
            temperature=self._settings.sampled_temperature,
            top_p=self._settings.sampled_top_p,
            n=self._settings.sampled_n,
        )


def build_inference_engine(
    *,
    checkpoint: Path,
    settings: InferenceSettings,
) -> InferenceEngine:
    backend = settings.backend.lower()
    if backend == "fixture":
        return FixtureInferenceEngine(settings)
    if backend == "transformers":
        return TransformersInferenceEngine(checkpoint=checkpoint, settings=settings)
    raise InferenceError(f"Unsupported inference backend: {settings.backend}")

