# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PyTorch implementation of [Soft-NMS](https://arxiv.org/abs/1704.04503) (soft non-maximum suppression) for object detection. The core algorithm runs as a PyTorch C++ extension (`soft_nms.so`/`soft_nms.pyd`) registered via `TORCH_LIBRARY`, while the Python package (`pt_soft_nms`) provides the public API and `torch.compile`/`torch.export` compatibility shims.

## Commands

**Install (builds the C++ extension):**
```bash
pip install --no-build-isolation -e .
```

**Run Python tests:**
```bash
pytest tests/
# single test:
pytest tests/test_soft_nms.py::TestSoftNMS::test_correct_keep
```

**Lint / format (Python):**
```bash
bash scripts/ci_checks.sh      # ruff check + ruff format --check + mypy
ruff check . --fix             # auto-fix lint
ruff format .                  # auto-format
```

**Format C++ (requires clang-format):**
```bash
bash scripts/run_format.sh     # format in-place
bash scripts/run_format.sh -c  # check only
```

**Run C++ tests (requires libtorch unpacked locally):**
```bash
bash scripts/run_cpp_tests.sh  # builds via cmake then runs ./build/tests/cpp/run_tests
```

## Architecture

```
pt_soft_nms/
  csrc/          — C++ source: soft_nms.cpp (algorithm), op.cpp (TORCH_LIBRARY registration)
  soft_nms.py    — Python wrappers: soft_nms() and batched_soft_nms()
  fake_soft_nms.py — Abstract/fake impl for torch.compile / torch.export (registered via torch.library.register_fake)
  __init__.py    — Loads soft_nms.so and imports the fake impl at package import time
tests/
  test_soft_nms.py      — Core correctness tests
  test_torch_compile.py — torch.compile compatibility
  test_jit_exports.py   — torch.export compatibility
```

**Key design points:**

- The C++ op is registered under the `soft_nms` namespace (`torch.ops.soft_nms.soft_nms`). Both CPU and CUDA dispatch keys are registered in `op.cpp`, but the actual implementation in `soft_nms.cpp` runs on CPU; CUDA tensors are handled by falling back through PyTorch's dispatcher.
- `batched_soft_nms` is pure Python: it offsets boxes by class index to make classes non-overlapping in coordinate space, then calls `soft_nms` once.
- `fake_soft_nms.py` is required for `torch.compile` and `torch.export` — it tells the tracing infrastructure the output shapes without executing the real op.
- The C++ extension is built with `torch.utils.cpp_extension.CppExtension` (no CUDA sources); CMake is only used for the standalone C++ unit tests.
- Requires PyTorch >= 2.4, Python >= 3.10.
