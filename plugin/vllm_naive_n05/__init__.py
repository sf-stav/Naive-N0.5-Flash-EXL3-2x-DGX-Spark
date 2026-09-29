"""Out-of-tree vLLM plugin for Naive-N0.5-Flash (NaiveN05FlashForCausalLM).

Registers the config class and the model with vLLM's registries. Loaded through the
`vllm.general_plugins` entry point in every vLLM process (API, engine core, workers,
model-inspection subprocess), so registration must be idempotent and CUDA-free at import.

Companion to the vllm-exl3 quantization plugin (EXL3 pack loading) — see
the repository README.
"""


def register() -> None:
    # --- config: model_type -> PretrainedConfig subclass -------------------------------
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from .configuration import NaiveN05FlashConfig

    if "naive_n05_flash" not in _CONFIG_REGISTRY:
        _CONFIG_REGISTRY["naive_n05_flash"] = NaiveN05FlashConfig

    # --- model: architectures[0] -> model class (lazy import to keep CUDA out of import) --
    from vllm.model_executor.models.registry import ModelRegistry

    try:
        ModelRegistry.register_model(
            "NaiveN05FlashForCausalLM", "vllm_naive_n05.model:NaiveN05FlashForCausalLM"
        )
    except ValueError:
        # already registered (plugin loads in multiple processes)
        pass

    # --- draft model: DFlash spec head ---------------------------------------------------
    # SpeculativeConfig wraps the draft Qwen3Config in an EAGLEConfig for method=dflash,
    # rewriting architectures DSparkDraftModel -> DFlashDSparkDraftModel. Register the
    # rewritten name to the Naive DSpark draft (mask embedding + SWA full-KV spec).
    try:
        ModelRegistry.register_model(
            "DFlashDSparkDraftModel",
            "vllm_naive_n05.dspark_draft:NaiveDSparkDraftForCausalLM",
        )
    except ValueError:
        pass

    # --- draft model: DSpark spec head (anchor sampling + Markov) ------------------------
    # method=dspark rewrites the architecture to Qwen3DSparkModel (in-tree map:
    # qwen3_dspark.Qwen3DSparkForCausalLM). Override it with the Naive subclass so the
    # learned `mask_embedding` tensor loads (the in-tree loader drops it) and every
    # sliding-window layer advertises a full KV spec for the pre-inserted context.
    try:
        ModelRegistry.register_model(
            "Qwen3DSparkModel",
            "vllm_naive_n05.dspark_draft:NaiveDSparkDraftForCausalLM",
        )
    except ValueError:
        pass

    # --- vllm-exl3 compat: allow fractional-K (3.5bpw) dense trellises -------------------
    from .exl3_compat import apply as _apply_exl3_compat

    _apply_exl3_compat()


__all__ = ["register"]
