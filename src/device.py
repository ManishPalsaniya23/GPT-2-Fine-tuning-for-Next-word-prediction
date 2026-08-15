"""GPU detection and hardware-aware training defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    device: torch.device
    name: str
    total_vram_gb: float
    supports_bf16: bool
    supports_fp16: bool
    cpu_count: int

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


def detect_device() -> DeviceInfo:
    cpu_count = os.cpu_count() or 1

    if not torch.cuda.is_available():
        return DeviceInfo(
            device=torch.device("cpu"),
            name="CPU",
            total_vram_gb=0.0,
            supports_bf16=False,
            supports_fp16=False,
            cpu_count=cpu_count,
        )

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return DeviceInfo(
        device=torch.device("cuda", index),
        name=props.name,
        total_vram_gb=props.total_memory / (1024**3),
        supports_bf16=torch.cuda.is_bf16_supported(),
        supports_fp16=props.major >= 6,
        cpu_count=cpu_count,
    )


def describe(info: DeviceInfo) -> str:
    lines = [
        "=" * 62,
        " Hardware check",
        "=" * 62,
        f"  torch version     : {torch.__version__}",
        f"  CUDA available    : {torch.cuda.is_available()}",
        f"  CUDA runtime      : {torch.version.cuda or 'n/a'}",
        f"  Device            : {info.name}",
        f"  VRAM              : {info.total_vram_gb:.2f} GB" if info.is_cuda else "  VRAM              : n/a",
        f"  bf16 supported    : {info.supports_bf16}",
        f"  fp16 supported    : {info.supports_fp16}",
        f"  CPU cores         : {info.cpu_count}",
        "=" * 62,
    ]
    return "\n".join(lines)


def free_vram_gb() -> float:
    """VRAM currently unallocated on the active CUDA device, in GB."""
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
