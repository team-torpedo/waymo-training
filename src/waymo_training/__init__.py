"""waymo-training: configurable camera training pipeline for the Waymo Open Dataset."""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path

# Optimize TensorFlow GPU memory management:
# 1. Silence benign cuDNN algorithm benchmark memory probe warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# 2. Disable expensive BFC garbage collection sweeps during algorithm probes
os.environ.setdefault("TF_ENABLE_GPU_GARBAGE_COLLECTION", "false")

__version__ = "0.1.0"


def _preload_cuda_libraries() -> None:
    """Preload NVIDIA CUDA/cuDNN shared libraries bundled in python site-packages.

    TensorFlow dlopens CUDA libraries at runtime using the dynamic linker. When
    installed in virtualenv site-packages (e.g. nvidia-cudnn, nvidia-cublas),
    dlopen fails unless either LD_LIBRARY_PATH is set or the .so files are
    already loaded into the global symbol table (RTLD_GLOBAL).
    """
    try:
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidate_dirs = [
            Path(sys.executable).parent.parent / "lib" / py_ver / "site-packages" / "nvidia",
            Path(__file__).resolve().parents[2] / "myenv" / "lib" / py_ver / "site-packages" / "nvidia",
        ]
        for nvidia_dir in candidate_dirs:
            if nvidia_dir.is_dir():
                for lib in sorted(glob.glob(str(nvidia_dir / "*" / "lib" / "*.so*"))):
                    if "nvblas" in lib:
                        continue
                    try:
                        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
                break
    except Exception:
        pass


_preload_cuda_libraries()

