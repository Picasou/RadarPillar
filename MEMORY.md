# MEMORY

## WSL2 + numba+CUDA SIGSEGV

WSL2 下 `import numba.cuda` 会调用 CUDA driver，与已存在的 PyTorch CUDA context 冲突，触发 SIGSEGV（`exit 139`），Python 抓不到任何 stack trace。

- **Why**：WSL2 通过 PTX/nvidia 透传层暴露 CUDA，driver state machine 在并发 context 时崩。
- **How to apply**：训练跑在 WSL2 时，不要用 numba 任何 CUDA 调用。已用 PyTorch 实现替换 `rotate_iou.py`（慢但精确）。真 Ubuntu 原生跑用回原版。
