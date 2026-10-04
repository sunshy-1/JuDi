"""JuDi with the repository's vLLM 0.8.3 runtime.

Import this package before importing vllm. Preparation commands intentionally
work without importing torch, triton or vllm.
"""

__all__ = ["create_judi_llm", "SamplingParams", "activate"]


def __getattr__(name):
    if name == "activate":
        from .runtime import activate
        return activate
    if name == "create_judi_llm":
        from .runtime import activate
        activate()
        from .patch import create_judi_llm
        return create_judi_llm
    if name == "SamplingParams":
        from .runtime import activate
        activate()
        from vllm import SamplingParams
        return SamplingParams
    raise AttributeError(name)
