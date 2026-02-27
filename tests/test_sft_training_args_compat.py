from __future__ import annotations

from microreasoner.train.sft_trainer import _build_training_arguments


class _OldStyleArgs:
    def __init__(self, output_dir: str, learning_rate: float) -> None:
        self.output_dir = output_dir
        self.learning_rate = learning_rate


class _VarKwArgs:
    def __init__(self, output_dir: str, **kwargs) -> None:
        self.output_dir = output_dir
        self.kwargs = kwargs


def test_build_training_arguments_filters_unknown_keys() -> None:
    kwargs = {
        "output_dir": "out",
        "learning_rate": 1e-4,
        "overwrite_output_dir": False,
    }
    args = _build_training_arguments(_OldStyleArgs, kwargs)
    assert isinstance(args, _OldStyleArgs)
    assert args.output_dir == "out"
    assert args.learning_rate == 1e-4


def test_build_training_arguments_keeps_kwargs_for_var_kw_signature() -> None:
    kwargs = {
        "output_dir": "out",
        "learning_rate": 1e-4,
        "overwrite_output_dir": False,
    }
    args = _build_training_arguments(_VarKwArgs, kwargs)
    assert isinstance(args, _VarKwArgs)
    assert args.output_dir == "out"
    assert args.kwargs["learning_rate"] == 1e-4
    assert args.kwargs["overwrite_output_dir"] is False
