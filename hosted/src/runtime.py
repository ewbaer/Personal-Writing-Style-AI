"""Load configured engines once and route each request by its model selection."""
import spaces  # Must precede torch imports.
from .hosted_engine import HostedEngine, gpu_seconds
from .config import model_profiles
from .ai_detector import AIDetector
from .exports import JsonExportStore
from .workflow import RewriteWorkflow
from .controller import DeskController


def production_controller(profiles=None):
    profiles = tuple(model_profiles() if profiles is None else profiles)
    engines = {}
    for profile in profiles:
        engine = HostedEngine(profile)
        engine.load()  # ZeroGPU places weights on CUDA at Space startup.
        if profile.key == "qwen" or engine.model is not None:
            engines[profile.key] = engine

    def duration(text, method, model_key):
        return gpu_seconds(text, method)

    @spaces.GPU(duration=duration)
    def generate(text, method, model_key):
        engine = engines[model_key]
        clean, tokens, count, ids = engine.prepare(text, method)
        stream = engine.stream(tokens)
        try:
            for item in stream:
                if isinstance(item, dict):
                    item.update(original=clean, method=method, example_ids=ids, prompt_tokens=count)
                yield item
        finally:
            stream.close()

    class ZeroGPURewriter:
        def __init__(self, engine):
            self.engine = engine

        def validate(self, text, method):
            self.engine.prepare(text, method)

        def stream(self, text, method):
            yield from generate(text, method, self.engine.profile.key)

    detector, exports = AIDetector(), JsonExportStore()
    controllers = {
        key: DeskController(RewriteWorkflow(ZeroGPURewriter(engine), detector, exports, engine.profile))
        for key, engine in engines.items()
    }
    default = controllers["qwen"]
    default.model_routes = controllers
    default.model_choices = [(engine.profile.label, key) for key, engine in engines.items()]
    default.experiment_prompt = engines["qwen"].profile.experiment_prompt
    return default
