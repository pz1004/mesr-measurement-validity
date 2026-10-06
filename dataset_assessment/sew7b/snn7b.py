"""Modern SpikingJelly implementation of the published DVS Gesture 7B-Net."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Mapping

import torch
from torch import nn

from spikingjelly.activation_based import functional, layer, neuron, surrogate


# torchvision DatasetFolder sorted the historical directories lexicographically
# as 0, 1, 10, 2, ..., 9.  These indices map semantic CSV class IDs to the
# corresponding released-classifier logit rows.
DVS_GESTURE_SEMANTIC_TO_RELEASED = (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 2)


class SEWBlock(nn.Module):
    """Two-convolution spike-element-wise residual block."""

    def __init__(self, channels: int = 32, connection: str = "ADD"):
        super().__init__()
        self.connection = connection
        node = dict(
            init_tau=2.0,
            detach_reset=True,
            # The April 2021 cext neuron used by the released checkpoint
            # defaulted to ATan(alpha=2), before the later Sigmoid default.
            surrogate_function=surrogate.ATan(alpha=2.0),
        )
        self.branch = nn.Sequential(
            layer.Conv2d(channels, channels, 3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.ParametricLIFNode(**deepcopy(node)),
            layer.Conv2d(channels, channels, 3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.ParametricLIFNode(**deepcopy(node)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        branch = self.branch(inputs)
        if self.connection == "ADD":
            return inputs + branch
        if self.connection == "AND":
            return inputs * branch
        if self.connection == "IAND":
            return inputs * (1.0 - branch)
        raise ValueError(f"Unsupported SEW connection: {self.connection}")


class SEW7BNet(nn.Module):
    """Seven-block, 32-channel network used in the SEW-ResNet paper.

    Input shape is ``[N, T, 2, 128, 128]`` and output shape is ``[N, 11]``.
    """

    def __init__(self, classes: int = 11, connection: str = "ADD"):
        super().__init__()
        self.stem = nn.Sequential(
            # The released DVS Gesture model's first stage uses up_kernel_size
            # 1 when projecting the two polarities to 32 channels.
            layer.Conv2d(2, 32, 1, padding=0, bias=False),
            layer.BatchNorm2d(32),
            neuron.ParametricLIFNode(
                init_tau=2.0,
                detach_reset=True,
                surrogate_function=surrogate.ATan(alpha=2.0),
            ),
        )
        stages = []
        for _ in range(7):
            stages.extend((SEWBlock(32, connection), layer.MaxPool2d(2, 2)))
        self.stages = nn.Sequential(*stages)
        self.classifier = layer.Linear(32, classes)
        functional.set_step_mode(self, "m")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError("inputs must have shape [N, T, 2, 128, 128]")
        values = inputs.transpose(0, 1)
        values = self.stem(values)
        values = self.stages(values)
        values = torch.flatten(values, 2)
        logits = self.classifier(values)
        return logits.mean(0)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def released_sew7b_key_map() -> "OrderedDict[str, str]":
    """Map the April 2021 checkpoint names to the modern module names."""

    mapping = OrderedDict(
        (
            ("conv.0.0.module.0.weight", "stem.0.weight"),
            ("conv.0.0.module.1.0.weight", "stem.1.weight"),
            ("conv.0.0.module.1.0.bias", "stem.1.bias"),
            ("conv.0.0.module.1.0.running_mean", "stem.1.running_mean"),
            ("conv.0.0.module.1.0.running_var", "stem.1.running_var"),
            ("conv.0.0.module.1.0.num_batches_tracked", "stem.1.num_batches_tracked"),
            ("conv.0.1.w", "stem.2.w"),
        )
    )
    for block in range(7):
        old_stage = 1 + 2 * block
        new_stage = 2 * block
        for convolution, offsets in enumerate(((0, 1, 2), (3, 4, 5))):
            conv_offset, bn_offset, neuron_offset = offsets
            old_prefix = f"conv.{old_stage}.conv.{convolution}"
            new_prefix = f"stages.{new_stage}.branch"
            mapping[f"{old_prefix}.0.module.0.weight"] = (
                f"{new_prefix}.{conv_offset}.weight"
            )
            for suffix in (
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ):
                mapping[f"{old_prefix}.0.module.1.0.{suffix}"] = (
                    f"{new_prefix}.{bn_offset}.{suffix}"
                )
            mapping[f"{old_prefix}.1.w"] = f"{new_prefix}.{neuron_offset}.w"
    mapping["out.weight"] = "classifier.weight"
    mapping["out.bias"] = "classifier.bias"
    return mapping


def convert_released_sew7b_state_dict(
    released: Mapping[str, torch.Tensor], model: nn.Module
) -> "OrderedDict[str, torch.Tensor]":
    """Convert and strictly validate the official legacy state dictionary."""

    mapping = released_sew7b_key_map()
    expected = model.state_dict()
    if set(released) != set(mapping):
        missing = sorted(set(mapping) - set(released))
        extra = sorted(set(released) - set(mapping))
        raise ValueError(
            f"Unexpected released checkpoint keys; missing={missing}, extra={extra}"
        )
    if set(mapping.values()) != set(expected):
        raise RuntimeError("Legacy key map does not cover the modern model exactly")
    converted = OrderedDict()
    for old_key, new_key in mapping.items():
        value = released[old_key]
        target = expected[new_key]
        if value.numel() != target.numel():
            raise ValueError(
                f"Shape mismatch for {old_key} -> {new_key}: "
                f"{tuple(value.shape)} versus {tuple(target.shape)}"
            )
        value = value.reshape(target.shape)
        if new_key in ("classifier.weight", "classifier.bias"):
            value = value[list(DVS_GESTURE_SEMANTIC_TO_RELEASED)]
        converted[new_key] = value
    return converted


def load_sew7b_state_dict(
    model: nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> str:
    """Load either a native reproduction or the official legacy checkpoint."""

    if "conv.0.0.module.0.weight" in state_dict:
        state_dict = convert_released_sew7b_state_dict(state_dict, model)
        checkpoint_format = "official SpikingJelly cext (April 2021)"
    else:
        checkpoint_format = "modern reproduction"
    model.load_state_dict(state_dict, strict=True)
    return checkpoint_format
