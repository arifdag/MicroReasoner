from __future__ import annotations

from pathlib import Path
from typing import Any


class HFCheckpointLoadError(RuntimeError):
    """Raised when a causal LM checkpoint cannot be loaded."""


def is_adapter_checkpoint(checkpoint: Path) -> bool:
    return (checkpoint / "adapter_config.json").exists()


def load_causal_lm_checkpoint(
    checkpoint: Path,
    *,
    use_fast: bool = True,
    torch_dtype: Any | None = None,
    device_map: Any | None = None,
    trainable_adapter: bool = False,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise HFCheckpointLoadError(
            "Loading a transformers checkpoint requires 'transformers'"
        ) from exc

    model_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=use_fast)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter_checkpoint(checkpoint):
        try:
            from peft import AutoPeftModelForCausalLM  # type: ignore
        except ImportError as exc:
            raise HFCheckpointLoadError(
                "Loading a PEFT adapter checkpoint requires 'peft'"
            ) from exc

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(checkpoint),
            is_trainable=trainable_adapter,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **model_kwargs)

    return tokenizer, model


def prepare_causal_lm_for_training(model: Any) -> Any:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    config_obj = getattr(model, "config", None)
    if config_obj is not None and hasattr(config_obj, "use_cache"):
        config_obj.use_cache = False
    return model
