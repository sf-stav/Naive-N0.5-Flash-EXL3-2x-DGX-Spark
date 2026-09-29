# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-A vLLM port of the Naive-N0.5-Flash architecture.

Ported from vLLM's ``vllm/model_executor/models/mimo_v2.py``.  Stage A serves dense causal attention on DSA layers (exact for
context <= ``index_top_k`` because top-k then selects the entire visible
history) and never invokes the DSA indexer.  Stage B/C hook points are marked
with comments.  Everything here is imported without initializing CUDA.
"""

import os
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
    str_dtype_to_torch_dtype,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Env-gated aux-capture debugging (inert unless set). NAIVE_AUX_DUMP_DIR dumps
# each forward's captured aux hidden states; NAIVE_AUX_FORCE_LAYERS forces the
# captured layer list when no speculative config is present (tiny-model tests).
_AUX_DUMP_DIR = os.environ.get("NAIVE_AUX_DUMP_DIR")
_AUX_FORCE_LAYERS = os.environ.get("NAIVE_AUX_FORCE_LAYERS")
_AUX_DUMP_ONLY = os.environ.get("NAIVE_AUX_DUMP_ONLY")
_AUX_DUMP_MAX = int(os.environ.get("NAIVE_AUX_DUMP_MAX", "4"))

from .dsa import NaiveN05FlashIndexer

logger = init_logger(__name__)


class NaiveN05FlashMLP(nn.Module):
    # Copied from MiMoV2MLP. Dense MLP for layer 0; same fused gate_up layout.
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class NaiveN05FlashMoE(nn.Module):
    # Copied from MiMoV2MoE. Config field names are identical; n_group ==
    # topk_group == 1 keeps grouped top-k equivalent to plain top-k.
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.n_routed_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        dtype = getattr(config, "moe_router_dtype", "float32")
        self.gate_dtype = str_dtype_to_torch_dtype(dtype)
        self.gate = nn.Linear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            dtype=self.gate_dtype,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=self.gate_dtype)
        )

        # Expert path decision (R4): the factory's `experts` child is a
        # MoERunner wrapping a RoutedExperts. The EXL3 plugin's
        # `isinstance(layer, RoutedExperts)` hook (exl3.py:2351-2378) fires
        # during create_weights and replaces RoutedExperts.load_weights with
        # its `_exl3_routed_experts_loader`; AutoWeightsLoader recursion
        # reaches it. Do NOT add a custom expert loop here. `prefix` must end
        # in `.experts` so layer_name and the `experts.{id}.gate_proj.`
        # mapping line up (factory layer_name, fused_moe/layer.py:221).
        self.experts = FusedMoEFactory(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            router_logits_dtype=self.gate_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, "NaiveN05FlashMoE only supports 1D or 2D inputs"
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.gate_dtype is not None:
            gate_input = hidden_states.to(self.gate_dtype)
        else:
            gate_input = hidden_states
        router_logits = self.gate(gate_input)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


def _make_sink_bias_weight_loader(tp_rank: int, tp_size: int):
    """Head-shard the checkpoint's full sink bias (mimo_v2.py:752-759)."""

    def _load(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        heads_per_rank = loaded_weight.shape[0] // tp_size
        head_start = tp_rank * heads_per_rank
        param.data.copy_(loaded_weight.narrow(0, head_start, heads_per_rank))

    return _load


def _is_exl3_quant_method(quant_method) -> bool:
    if quant_method is None:
        return False
    cls = type(quant_method)
    return cls.__name__ == "Exl3LinearMethod" and (cls.__module__ or "").split(
        "."
    )[0] == "vllm_exl3"


def _linear_is_exl3(quant_config, linear: nn.Module | None, prefix: str) -> bool:
    """Detect a linear served by the EXL3 plugin (R1/R3 dedicated layouts).

    Primary check is the quant method vLLM actually resolved for ``linear``;
    if that is unavailable (e.g. before construction), fall back to the
    plugin's Exl3Config prefix match.  Never raises when the EXL3 plugin is
    not installed.
    """
    if _is_exl3_quant_method(getattr(linear, "quant_method", None)):
        return True
    try:
        from vllm_exl3.exl3 import Exl3Config
    except Exception:
        return False
    if not isinstance(quant_config, Exl3Config):
        return False
    matches = getattr(quant_config, "_matches_non_routed_exl3", None)
    if callable(matches):
        try:
            return bool(matches(prefix))
        except Exception:
            return False
    return False


class NaiveN05FlashAttention(nn.Module):
    """Both attention branches, selected by ``config.hybrid_layer_pattern``.

    0 = DSA/full attention (no sinks, indexer constructed but never called in
    Stage A); 1 = SWA (sinks, window ``config.sliding_window``).

    Deviations from mimo_v2.MiMoV2Attention:
      * split ``q_proj``/``k_proj``/``v_proj`` ColumnParallelLinear instead of
        one QKVParallelLinear (checkpoint and EXL3 pack are split; EXL3's
        loader only fuses the ``qkv_proj`` layout, exl3.py:4071-4077);
      * DSA layers construct a NaiveN05FlashIndexer;
      * ``v_scale`` is skipped when ``o_proj`` is EXL3-packed (risk R1).
    """

    def __init__(
        self,
        config,
        layer_id: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        # Deviation vs mimo's is_compressed_softmax_layer (inverted polarity):
        # 0 = DSA, 1 = SWA.
        self.is_swa = bool(config.hybrid_layer_pattern[layer_id])
        self.is_dsa = not self.is_swa
        p = "swa_" if self.is_swa else ""

        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()

        self.total_num_heads = getattr(config, p + "num_attention_heads")
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = getattr(config, p + "num_key_value_heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(config, p + "head_dim")
        self.v_head_dim = getattr(config, p + "v_head_dim", None)
        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim

        self.v_scale = getattr(config, "attention_value_scale", None)
        self.scaling = self.head_dim**-0.5

        # exllamav3 packs store v_proj with each head's v_head_dim rows first
        # and zero padding up to head_dim (naive_vproj.py:40-42), so the stored
        # output width is num_kv_heads * head_dim, not * v_head_dim. Native /
        # BF16 checkpoints store the true compact layout. Detect from the
        # quant config before building the module (the plugin's Exl3Config
        # declares v_proj in non_routed_exl3).
        self.v_proj_is_exl3 = _linear_is_exl3(
            quant_config, None, f"{prefix}.v_proj"
        )
        self.v_proj_layout_padded = (
            self.v_proj_is_exl3 and self.v_head_dim != self.head_dim
        )
        v_proj_out = (
            self.total_num_kv_heads * self.head_dim
            if self.v_proj_layout_padded
            else self.total_num_kv_heads * self.v_head_dim
        )

        # EXL3 pads every TP shard's matrix dims to 128 and refuses padded
        # geometry when tp_size > 1 (exl3.py:3863-3876).  Tiny fixtures have
        # per-rank q/k/v widths below 128 (or not 128-aligned), which cannot
        # be fixed from inside a ColumnParallelLinear.  In that case build
        # q/k/v replicated (full width) and slice this rank's heads out of
        # the full output in forward(); o_proj stays row-parallel, so
        # attention keeps its head-wise TP sharding.  Real-model dims are
        # 128-aligned per rank, so this path is tiny-fixture-only.
        self.qkv_is_exl3 = _linear_is_exl3(quant_config, None, f"{prefix}.q_proj")
        self.qkv_replicated = (
            tp_size > 1
            and self.qkv_is_exl3
            and any(
                x % (128 * tp_size)
                for x in (
                    self.total_num_heads * self.head_dim,
                    self.total_num_kv_heads * self.head_dim,
                    v_proj_out,
                )
            )
        )

        # Deviation vs mimo: split q/k/v projections.
        qkv_extra = {"disable_tp": True} if self.qkv_replicated else {}
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
            **qkv_extra,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
            **qkv_extra,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            v_proj_out,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
            **qkv_extra,
        )
        if self.qkv_replicated:
            # Same contract the indexer pins for EXL3 (dsa.py:77-81): the
            # quant method resolves geometry from layer/param attributes.
            for lin in (self.q_proj, self.k_proj, self.v_proj):
                lin.tp_rank, lin.tp_size = 0, 1
                update_tp = getattr(lin, "update_param_tp_status", None)
                if callable(update_tp):
                    update_tp()
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        # R1: module flag used by forward() to avoid double-applying v_scale.
        self.o_proj_is_exl3 = _linear_is_exl3(
            quant_config, self.o_proj, f"{prefix}.o_proj"
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 32768),
            rope_parameters={
                "rope_type": "default",
                "rope_theta": getattr(
                    config, p + "rope_theta", getattr(config, "rope_theta", 1000000)
                ),
                "partial_rotary_factor": getattr(config, "partial_rotary_factor", 1.0),
            },
        )

        # Blueprint 2.2: create the sink param for SWA layers only. DSA layers
        # always get sinks=None in Stage A (add_full_attention_sink_bias is
        # False in every shipped config).
        if self.is_swa and getattr(config, "add_swa_attention_sink_bias", False):
            self.attention_sink_bias = nn.Parameter(
                torch.empty(self.num_heads), requires_grad=False
            )
            # Manual head-shard on load (mimo_v2.py:752-759) via the param's
            # weight_loader so AutoWeightsLoader picks it up.
            self.attention_sink_bias.weight_loader = _make_sink_bias_weight_loader(
                get_tensor_model_parallel_rank(), tp_size
            )
        else:
            self.attention_sink_bias = None

        sliding_window = (
            getattr(config, "sliding_window", None) if self.is_swa else None
        )

        # --- DiffKV backend auto-selection, copied from mimo_v2.py:296-318 ---
        # Use DiffKV backend when V has a different head dim than K.
        # Auto-pick FA-DiffKV when FA3/4 is usable on this device, else fall
        # back to TRITON_ATTN_DIFFKV.  Users can force a choice via
        # `--attention-backend <FLASH_ATTN_DIFFKV|TRITON_ATTN_DIFFKV>`.
        if self.v_head_dim != self.head_dim:
            requested = get_current_vllm_config().attention_config.backend
            if requested is not None and requested.name.endswith("_DIFFKV"):
                backend_enum = requested
            else:
                fa_backend = AttentionBackendEnum.FLASH_ATTN_DIFFKV.get_class()
                if fa_backend.is_supported_on_current_device(
                    head_size=self.head_dim,
                    head_size_v=self.v_head_dim,
                    has_sinks=self.attention_sink_bias is not None,
                ):
                    backend_enum = AttentionBackendEnum.FLASH_ATTN_DIFFKV
                else:
                    backend_enum = AttentionBackendEnum.TRITON_ATTN_DIFFKV
            attn_backend = backend_enum.get_class()
            attn_backend.set_head_size_v(self.v_head_dim)
            logger.info_once("Using %s for attention.", attn_backend.get_name())
        else:
            attn_backend = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.attention_sink_bias,
            attn_backend=attn_backend,
            head_size_v=self.v_head_dim,
        )

        # Stage A: indexer is constructed and loaded on DSA layers only, and
        # never called. Stage B/C hook lives in vllm_naive_n05/dsa.py.
        self.indexer = (
            None
            if self.is_swa
            else NaiveN05FlashIndexer(
                config, prefix=f"{prefix}.indexer", quant_config=quant_config
            )
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # Replicated (full-width) q/k/v: keep only this rank's heads.
        if self.qkv_replicated:
            b = q.shape[0]
            q = q.view(b, self.total_num_heads, self.head_dim)[
                :,
                self.tp_rank * self.num_heads : (self.tp_rank + 1) * self.num_heads,
            ].reshape(b, self.q_size)
            kv = slice(
                self.tp_rank * self.num_kv_heads,
                (self.tp_rank + 1) * self.num_kv_heads,
            )
            k = k.view(b, self.total_num_kv_heads, self.head_dim)[:, kv].reshape(
                b, self.k_size
            )
            v = v.view(b, self.total_num_kv_heads, self.head_dim)[:, kv].reshape(b, -1)

        # exllamav3 packs keep V per head padded to head_dim; compact back to
        # the true [num_kv_heads, v_head_dim] layout vLLM's Attention expects
        # (head_size_v). The per-head block order is [true | pad], so this is
        # a strided select, not an end trim.
        if self.v_proj_layout_padded:
            v = v.reshape(-1, self.num_kv_heads, self.head_dim)[..., : self.v_head_dim]
            v = v.reshape(-1, self.num_kv_heads * self.v_head_dim)

        q, k = self.rotary_emb(positions, q, k)

        # R1: EXL3 o_proj had attention_value_scale baked into the stored
        # weight at conversion (exllamav3/modules/linear.py:236-237), so
        # applying it here would double-scale by 0.707. Native/BF16 o_proj
        # needs the forward-side scale (reference modeling:141-142).
        if self.v_scale is not None and not self.o_proj_is_exl3:
            v = v * self.v_scale

        # Stage A: dense causal GQA, exact for context <= config.index_top_k
        # (top-k then selects the entire visible history).
        # Stage B/C hook: DSA layers replace this call with the indexer
        # score -> stable argsort top-k -> block-mask/backend path.
        attn_output = self.attn(q, k, v)

        output, _ = self.o_proj(attn_output)
        return output


class NaiveN05FlashDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id

        # Deviation vs mimo: branch on hybrid_layer_pattern (0 = DSA, 1 = SWA)
        # inside NaiveN05FlashAttention; no compressed-softmax naming.
        self.self_attn = NaiveN05FlashAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.is_layer_sparse = self.is_moe_layer(layer_id)
        if self.is_layer_sparse:
            self.mlp = NaiveN05FlashMoE(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = NaiveN05FlashMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            hasattr(self.config, "moe_layer_freq")
            and layer_idx >= 0
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )


@support_torch_compile
class NaiveN05FlashModel(nn.Module, EagleModelMixin):
    # Deviation vs mimo_v2.MiMoV2Model (R4): no custom load_weights expert
    # loop. This mapper only fuses the dense layer-0 gate_proj/up_proj into
    # gate_up_proj; `.mlp.experts.{id}.*` names are untouched and reach
    # MoERunner/RoutedExperts (or the EXL3 replacement loader) through
    # AutoWeightsLoader recursion.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self._aux_dump_counter = 0
        if _AUX_FORCE_LAYERS:
            parsed = tuple(int(x) for x in _AUX_FORCE_LAYERS.split(",") if x != "")
            self._set_aux_hidden_state_layers(parsed)
            logger.info("NaiveN05Flash: forced aux layers %s (debug env)", parsed)

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: NaiveN05FlashDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if _AUX_DUMP_DIR and len(aux_hidden_states) > 0:
            dump_idx = self._aux_dump_counter
            self._aux_dump_counter += 1
            if dump_idx < _AUX_DUMP_MAX:
                os.makedirs(_AUX_DUMP_DIR, exist_ok=True)
                path = os.path.join(_AUX_DUMP_DIR, f"aux_dump.{dump_idx:03d}.pt")
                torch.save(
                    {
                        "input_ids": None if input_ids is None else input_ids.cpu(),
                        "positions": positions.cpu(),
                        "aux_layers": tuple(self.aux_hidden_state_layers),
                        "aux": [x.detach().float().cpu() for x in aux_hidden_states],
                        "hidden_pre_norm": hidden_states.detach().float().cpu(),
                    },
                    path,
                )
            if _AUX_DUMP_ONLY:
                # Debug-only: the V2 runner asserts a tensor output when no
                # speculative config requests aux states; emit a tensor so the
                # generation proceeds while we keep the captured aux dump.
                return hidden_states

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Non-expert remap only (R4). Expert tensors are delegated to
        # MoERunner.load_weights / RoutedExperts (or the EXL3 replacement
        # loader) during recursion; nothing here resolves gate_proj to
        # w13_weight, so EXL3 trellis tensors are not dropped.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class NaiveN05FlashForCausalLM(
    nn.Module, SupportsPP, MixtureOfExperts, SupportsEagle3
):
    # Deviation vs mimo: no qkv_proj entry (q/k/v are split modules, so the
    # checkpoint names match directly).
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = NaiveN05FlashModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = NaiveN05FlashModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Stage A: skip any MTP/drafter tensors, matching mimo's
        # `if "mtp" in name: continue`; the DSpark drafter is a
        # separate model (dspark_draft.py). AutoWeightsLoader recurses into NaiveN05FlashModel,
        # whose load_weights applies the dense gate/up mapper and whose
        # MoERunner children consume the expert tensors, including EXL3
        # trellis weights (R4).
        # vLLM 0.29's AutoWeightsLoader has no skip_substrs argument, so the
        # mtp filter is applied to the iterator here.
        weights = ((name, w) for name, w in weights if "mtp" not in name)
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


__all__ = ["NaiveN05FlashForCausalLM", "NaiveN05FlashModel"]
