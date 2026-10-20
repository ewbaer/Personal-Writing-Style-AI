"""Run a small CUDA calculation, not just a GPU availability check."""
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check your NVIDIA driver and CUDA PyTorch install.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print("GPU calculation passed. You can launch the app.")
