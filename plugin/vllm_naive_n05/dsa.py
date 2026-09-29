# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-A DSA indexer module for Naive-N0.5-Flash.

Reference semantics: ``models/tiny-naive/modeling_naive_n05_flash.py:61-83``
(``NaiveN05FlashIndexer``).

Stage A constructs and loads this module (its tensors are part of the
checkpoint / EXL3 pack and must be consumed by the weight loader) but never
calls it.  Stage B/C implements the fused score -> top-k path documented at
the bottom of this file.  ``forward`` is intentionally absent.
"""

from torch import nn

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig


class NaiveN05FlashIndexer(nn.Module):
    """Replicated linear planes + LayerNorm for DSA key scoring.

    All projections are ``ReplicatedLinear`` because every TP rank scores the
    full visible history (reference: one indexer KV head, ``index_n_kv_heads``
    == 1).  ``disable_tp=True`` pins the layer's TP geometry to ``(0, 1)`` from
    the moment ``create_weights()`` runs; the explicit ``tp_rank``/``tp_size``
    attributes document the contract for the EXL3 plugin's
    ``_resolve_tp_geometry`` (exl3.py:1033-1052).  This matters because EXL3
    pads ``weights_proj`` (out=16) to 128 and refuses padded geometry when
    TP > 1 (exl3.py:3863-3876), and because the setter must happen before
    ``create_weights`` (which is called from ``ReplicatedLinear.__init__``).
    """

    def __init__(
        self,
        config,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.top_k = config.index_top_k
        # fp8_e4m3 rounding of q/k in the reference (round_indexer_fp8).
        self.fp8 = config.indexer_activation_dtype == "fp8_e4m3"

        self.wq = ReplicatedLinear(
            h,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq",
            disable_tp=True,
        )
        self.wk = ReplicatedLinear(
            h,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
            disable_tp=True,
        )
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        self.weights_proj = ReplicatedLinear(
            h,
            self.n_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.weights_proj",
            disable_tp=True,
        )

        # Pin replicated geometry on the module and, when
        # the params are vLLM parameters, on the params as well.
        for lin in (self.wq, self.wk, self.weights_proj):
            lin.tp_rank, lin.tp_size = 0, 1
            update_tp = getattr(lin, "update_param_tp_status", None)
            if callable(update_tp):
                update_tp()

        # ------------------------------------------------------------------
        # Stage A: intentionally no forward().
        #
        # Stage B/C hook -- fused equivalent of the reference math, to be
        # implemented against models/tiny-naive/modeling_naive_n05_flash.py:
        #
        #   q = self.wq(x).view(..., self.n_heads, self.head_dim)
        #   k = self.k_norm(self.wk(x))
        #   q, k = apply_rope(q, k)              # NEOX, partial rotary
        #   if self.fp8:
        #       # round_indexer_fp8: per-token amax(dim=-1) clamped >= 1e-4,
        #       # scale = amax / 448; round (x / scale).clamp(-448, 448) to
        #       # torch.float8_e4m3fn, dequantize * scale.
        #       q = round_indexer_fp8(q)
        #       k = round_indexer_fp8(k)
        #   # k is appended to a per-layer indexer KV cache in the reference
        #   # (past_key_values.update_indexer); the Stage B/C backend owns it.
        #   scores = (q.float() @ k.float().transpose(-1, -2)).relu()
        #   w = self.weights_proj(x) * self.n_heads**-0.5          # (B, T, H)
        #   scores = (scores * w.transpose(1, 2).unsqueeze(-1).float()).sum(1)
        #   scores = scores.masked_fill(~allowed, -torch.inf)
        #   # Stable ties keep padding from changing which equal keys win:
        #   selected = scores.argsort(dim=-1, descending=True, stable=True)
        #   selected = selected[..., : self.top_k]
        # ------------------------------------------------------------------


__all__ = ["NaiveN05FlashIndexer"]
