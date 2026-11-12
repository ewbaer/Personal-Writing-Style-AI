# Deployment

The existing Space remains on Gradio 6.30.0, Python 3.12, and ZeroGPU. Upload the package following [UPLOAD.md](UPLOAD.md). Dependency versions are retained from the current Space. No account-specific quota or hardware entitlement is assumed.

Startup loads Qwen once. `runtime.py` keeps GPU generation behind `spaces.GPU`. Detector checks and UI updates run on CPU. A local launch uses your own hardware and requires a compatible PyTorch installation. The tiny CPU model used by tests is not the production model.
