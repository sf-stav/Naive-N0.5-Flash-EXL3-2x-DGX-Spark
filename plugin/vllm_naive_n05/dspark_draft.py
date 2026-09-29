"""Naive-N0.5-Flash DSpark/DFlash draft head for vLLM speculation.

Out-of-tree draft model for the Naive drafter (Naive-N0.5-Flash-FP8-Draft).
The checkpoint is a DSpark draft: a DFlash Qwen3 backbone with Markov /
confidence heads whose native drafting semantics is anchor sampling -- each
query position's hidden state predicts the NEXT token (the anchor predicts the
first draft token, hidden@p predicts token@p+1). `--speculative-config
method=dspark` selects vLLM's DSpark speculator, which implements exactly that
layout plus the sequential Markov head. `method=dflash` instead samples each
hidden state at its own position; that off-by-one layout makes every draft token
land one position late and collapses acceptance to ~3% on this checkpoint.

Two deviations from upstream `Qwen3DSparkForCausalLM` are required:

1. The learned mask embedding ships as a `mask_embedding` tensor inside
   model.safetensors (dflash_config.use_mask_embedding=true). Upstream
   Qwen3DSparkForCausalLM deliberately maps `mask_embedding` to None (its
   checkpoints mask via the frozen target vocab row); we load it and enable
   `has_separate_mask_embedding` so `embed_input_ids` substitutes it.

2. Every draft layer is sliding-window (window 1024). DFlash pre-inserts the
   verifier context K/V at absolute cache slots, so a SlidingWindowSpec would
   let the KV manager free/reuse those blocks underneath the precompute write.
   Nulling the inner Attention layer's `sliding_window` after construction
   makes it advertise a full-attention KV spec; the FlashAttention impl keeps
   the window it was built with, so SWA still applies at compute time. This
   mirrors the in-tree DFlashLagunaModel fix for SWA drafters.

`__init__.register` maps the `DSparkDraftModel` (method=dspark rewrites it to
`Qwen3DSparkModel`) and `DFlashDSparkDraftModel` (method=dflash EAGLEConfig)
architectures to this class so the checkpoint loads in either method.
"""

from collections.abc import Iterable
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

logger = init_logger(__name__)

# Env-gated pipeline debugging (inert unless set). NAIVE_DRAFT_DUMP_DIR dumps
# the first few draft-model calls (context-KV inputs, query forward I/O,
# sampled logits) so the runtime inputs can be replayed against the HF
# reference drafter.
_DRAFT_DUMP_DIR = os.environ.get("NAIVE_DRAFT_DUMP_DIR")
_DRAFT_DUMP_MAX = int(os.environ.get("NAIVE_DRAFT_DUMP_MAX", "6"))


class NaiveDSparkDraftForCausalLM(Qwen3DSparkForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._draft_dump_counts: dict[str, int] = {}

        # Keep full KV allocation for the pre-inserted verifier context; the
        # compute-time window lives in the already-built Attention impl.
        for layer in self.model.layers:
            inner = getattr(layer.self_attn, "attn", None)
            if inner is not None and getattr(inner, "sliding_window", None) is not None:
                inner.sliding_window = None

        # The inner model embeds input ids itself (DFlashQwen3Model.forward);
        # wrap it so debug runs capture the exact embedding the draft consumes.
        if _DRAFT_DUMP_DIR is not None:
            inner_model = self.model
            orig_embed = inner_model.embed_input_ids

            def _embed_and_dump(ids, _orig=orig_embed):
                emb = _orig(ids)
                self._debug_dump("embed", input_ids=ids, embeds=emb)
                return emb

            inner_model.embed_input_ids = _embed_and_dump

    def _debug_dump(self, kind: str, **fields) -> None:
        if _DRAFT_DUMP_DIR is None:
            return
        idx = self._draft_dump_counts.get(kind, 0)
        self._draft_dump_counts[kind] = idx + 1
        if idx >= _DRAFT_DUMP_MAX:
            return
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            tp_rank = 0
        os.makedirs(_DRAFT_DUMP_DIR, exist_ok=True)
        payload = {
            "mask_token_id": getattr(self.model, "mask_token_id", None),
            "has_separate_mask_embedding": getattr(
                self.model, "has_separate_mask_embedding", None
            ),
            "mask_embedding": (
                None
                if getattr(self.model, "mask_embedding", None) is None
                else self.model.mask_embedding.detach().float().cpu()
            ),
        }
        for name, value in fields.items():
            if isinstance(value, torch.Tensor):
                payload[name] = value.detach().float().cpu()
            else:
                payload[name] = value
        path = os.path.join(_DRAFT_DUMP_DIR, f"draft_{kind}.{tp_rank}.{idx:03d}.pt")
        torch.save(payload, path)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        embeds = super().embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal
        )
        self._debug_dump("embed", input_ids=input_ids, embeds=embeds)
        return embeds

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out = super().combine_hidden_states(hidden_states)
        self._debug_dump("combine", hidden_states=hidden_states, output=out)
        return out

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping=None,
    ) -> None:
        self._debug_dump(
            "ctxkv",
            context_states=context_states,
            context_positions=context_positions,
            context_slot_mapping=context_slot_mapping,
        )
        return super().precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = super().forward(input_ids, positions, inputs_embeds)
        self._debug_dump(
            "fwd", input_ids=input_ids, positions=positions, output=out
        )
        return out

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None:
            top = logits.float().topk(5, dim=-1)
            self._debug_dump(
                "logits",
                hidden_states=hidden_states,
                logits_shape=tuple(logits.shape),
                top_ids=top.indices,
                top_vals=top.values,
                argmax=logits.argmax(dim=-1),
            )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Intercept the learned mask embedding before the upstream loader,
        # which intentionally drops `mask_embedding`.
        mask_embedding = None
        filtered: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name == "mask_embedding" or name.endswith(".mask_embedding"):
                mask_embedding = weight
                continue
            filtered.append((name, weight))

        super().load_weights(filtered)

        if mask_embedding is not None:
            self.model.mask_embedding.data.copy_(
                mask_embedding.reshape(-1).to(self.model.mask_embedding.dtype)
            )
            self.model.has_separate_mask_embedding = True
            logger.info(
                "NaiveDSparkDraft: loaded learned mask embedding %s norm=%.4f",
                tuple(mask_embedding.shape),
                mask_embedding.float().norm().item(),
            )
        else:
            logger.warning(
                "NaiveDSparkDraft: no mask_embedding tensor seen during weight load; "
                "masked slots will use the target vocab row"
            )
