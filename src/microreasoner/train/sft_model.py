from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microreasoner.runtime.models import ResolvedConfig


class SFTModelSetupError(RuntimeError):
    """Raised when SFT model stack cannot be initialized."""


@dataclass(frozen=True)
class SFTModeSelection:
    requested_mode: str
    selected_mode: str
    reason: str


def _cuda_total_vram_gb() -> float | None:
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(0)
    total_bytes = float(getattr(properties, "total_memory", 0))
    if total_bytes <= 0:
        return None
    return total_bytes / float(1024**3)


def select_sft_mode(config: ResolvedConfig) -> SFTModeSelection:
    requested = config.train_sft.mode.lower()
    if requested in {"lora", "qlora"}:
        return SFTModeSelection(
            requested_mode=requested,
            selected_mode=requested,
            reason="explicit_mode",
        )
    if requested != "auto":
        raise SFTModelSetupError(f"Unsupported train_sft.mode: {config.train_sft.mode}")

    vram_gb = _cuda_total_vram_gb()
    if vram_gb is None:
        return SFTModeSelection(
            requested_mode=requested,
            selected_mode="qlora",
            reason="no_cuda_or_torch_detected",
        )
    if vram_gb >= 48.0:
        return SFTModeSelection(
            requested_mode=requested,
            selected_mode="lora",
            reason=f"cuda_vram_gb={vram_gb:.2f}",
        )
    return SFTModeSelection(
        requested_mode=requested,
        selected_mode="qlora",
        reason=f"cuda_vram_gb={vram_gb:.2f}",
    )


def resolve_sft_backend(config: ResolvedConfig) -> str:
    backend = config.train_sft.backend.trainer.lower()
    if backend not in {"transformers", "fixture"}:
        raise SFTModelSetupError(f"Unsupported train_sft.backend.trainer: {backend}")
    return backend


def require_transformers_stack(*, selected_mode: str) -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")  # type: ignore
        peft = importlib.import_module("peft")  # type: ignore
        transformers = importlib.import_module("transformers")  # type: ignore
    except ImportError as exc:
        raise SFTModelSetupError(
            "Transformers SFT backend requires torch, transformers, and peft"
        ) from exc

    stack = {
        "torch": torch,
        "AutoModelForCausalLM": getattr(transformers, "AutoModelForCausalLM"),
        "AutoTokenizer": getattr(transformers, "AutoTokenizer"),
        "Trainer": getattr(transformers, "Trainer"),
        "TrainingArguments": getattr(transformers, "TrainingArguments"),
        "LoraConfig": getattr(peft, "LoraConfig"),
        "get_peft_model": getattr(peft, "get_peft_model"),
    }
    if selected_mode == "qlora":
        try:
            importlib.import_module("bitsandbytes")  # type: ignore
        except ImportError as exc:
            raise SFTModelSetupError(
                "QLoRA SFT backend requires bitsandbytes in addition to torch, transformers, and peft"
            ) from exc
        stack["BitsAndBytesConfig"] = getattr(transformers, "BitsAndBytesConfig")
    return stack


@dataclass
class HFModelBundle:
    model: Any
    tokenizer: Any
    stack: dict[str, Any]


def build_transformers_model(
    *,
    config: ResolvedConfig,
    selected_mode: str,
    checkpoint_or_model: Path | str,
) -> HFModelBundle:
    stack = require_transformers_stack(selected_mode=selected_mode)
    torch = stack["torch"]

    AutoTokenizer = stack["AutoTokenizer"]
    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    LoraConfig = stack["LoraConfig"]
    get_peft_model = stack["get_peft_model"]

    model_path = str(checkpoint_or_model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if selected_mode == "qlora":
        BitsAndBytesConfig = stack["BitsAndBytesConfig"]
        quant_conf = config.train_sft.quantization
        if not quant_conf.enabled:
            raise SFTModelSetupError("QLoRA selected but quantization.enabled=false")
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        compute_dtype = dtype_map.get(quant_conf.bnb_4bit_compute_dtype.lower())
        if compute_dtype is None:
            raise SFTModelSetupError(
                "Unsupported bnb_4bit_compute_dtype: "
                f"{quant_conf.bnb_4bit_compute_dtype}"
            )
        bnb_conf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_conf.double_quant,
            bnb_4bit_quant_type=quant_conf.quant_type,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_conf,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    lora_conf = config.train_sft.lora
    peft_config = LoraConfig(
        r=lora_conf.r,
        lora_alpha=lora_conf.alpha,
        lora_dropout=lora_conf.dropout,
        target_modules=list(lora_conf.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return HFModelBundle(model=model, tokenizer=tokenizer, stack=stack)

