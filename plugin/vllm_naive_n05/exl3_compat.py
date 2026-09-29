"""Fractional-K compatibility shim for vllm-exl3 packs (Naive-N0.5-Flash 3.5bpw).

The real pack stores every EXL3 trellis with 56 int16 words per 16x16 tile
(K = 3.5), plus lm_head at 96 words (K = 6).  vllm-exl3 sizes non-routed
*linear* trellises from the integer ``bits`` in ``quantization_config``
(``k_words = bits * 16``) and validates ``bits`` against {2,3,4,5,6}, so a
uniform 3.5 pack cannot be expressed in its vocabulary.  Routed expert
trellis storage is ragged and sized from the checkpoint on load, but the
loader guard and the fused apply paths still assume integer K.

The pack rewriter (``tools/naive_pack_config.py``) therefore keeps the integer
bits for validation and additionally emits the exact word count:
``non_routed_exl3.layers[prefix]["k_words"]`` per dense prefix, plus a default
``non_routed_exl3["k_words"]`` for suffix-matched prefixes.  This shim wraps
``Exl3Config._bits_for_non_routed`` to hand ``Exl3LinearMethod`` an
int-compatible bits carrier whose ``bits * 16`` yields the true ``k_words``,
and preserves that carrier through ``Exl3LinearMethod.__init__`` (which
otherwise ``int()``-casts it away).

The fractional runtime lives in the vcruz305 exllamav3 fork (float ``K``,
``frac.cuh``, HALF kernel instances); vllm-exl3 already builds its linears and
calls its MoE kernels through that extension, so only the plugin's own
integer-K assumptions need patching:

  * the expert-loader width guard accepts ``16*K + 8`` (half-integer) words,
  * ``physical_k_compat`` keeps the float K from ``LinearEXL3.K``,
  * the fused/native apply paths stop ``int()``-truncating K (3.5 -> 3),
  * the fused-state builder derives K from the physical trellis width.

Nothing in vllm-exl3/exllamav3 is modified on disk; the patches are applied
in-process by :func:`register` and are behavior-preserving for integer-K
packs (all replacement anchors are no-ops for int K).
"""

import logging
import os

logger = logging.getLogger(__name__)


class FractionalKBits(int):
    """Integer bits carrying the exact packed word count for ``bits * 16``."""

    def __new__(cls, bits, k_words):
        obj = int.__new__(cls, int(bits))
        obj.k_words = int(k_words)
        return obj

    def __mul__(self, other):
        if other == 16:
            return self.k_words
        return int.__mul__(self, other)

    __rmul__ = __mul__


def _patch_config_bits(exl3) -> None:
    cfg_cls = getattr(exl3, "Exl3Config", None)
    lin_cls = getattr(exl3, "Exl3LinearMethod", None)
    if cfg_cls is None or lin_cls is None or getattr(cfg_cls, "_naive_fk_patch", False):
        return

    orig_bits_for_non_routed = cfg_cls._bits_for_non_routed

    def _k_words_for(self, prefix):
        nr = self.non_routed_exl3 or {}
        entry = (nr.get("layers") or {}).get(prefix)
        if isinstance(entry, dict) and "k_words" in entry:
            return int(entry["k_words"])
        if "k_words" in nr:
            return int(nr["k_words"])
        return None

    def _bits_for_non_routed(self, prefix):
        bits = orig_bits_for_non_routed(self, prefix)
        k_words = _k_words_for(self, prefix)
        if k_words is not None and k_words != int(bits) * 16:
            return FractionalKBits(int(bits), k_words)
        return bits

    orig_init = lin_cls.__init__

    def __init__(self, quant_config, bits=None):
        if isinstance(bits, FractionalKBits):
            orig_init(self, quant_config, bits=None)
            self.bits = bits
        else:
            orig_init(self, quant_config, bits=bits)

    cfg_cls._bits_for_non_routed = _bits_for_non_routed
    lin_cls.__init__ = __init__
    cfg_cls._naive_fk_patch = True


def _source_patch(fn, replacements):
    """Rebuild ``fn`` with textual replacements and install it in its module."""
    import inspect
    import textwrap

    source = inspect.getsource(fn)
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(
                f"frac-K shim anchor missing in {fn.__qualname__}: {old!r}"
            )
        source = source.replace(old, new)
    namespace = fn.__globals__
    filename = inspect.getsourcefile(fn) or "<exl3>"
    exec(compile(textwrap.dedent(source), filename, "exec"), namespace)
    return namespace[fn.__name__]


def _frac_k_from_inners(inners, default):
    """Uniform physical K from the fork's LinearEXL3 objects, float-preserving."""
    values = []
    for pack in inners:
        for projection in ("gate", "up", "down"):
            linear = pack[projection]
            k = getattr(linear, "K", None)
            if k is None:
                trellis = getattr(linear, "trellis", None)
                if trellis is None or getattr(trellis, "ndim", 0) != 3:
                    return default
                k = trellis.shape[-1] / 16
            values.append(k)
    if not values:
        return default
    first = values[0]
    if all(v == first for v in values):
        return int(first) if float(first).is_integer() else float(first)
    return default


def _physical_k_values(inners):
    """Float-preserving twin of vllm_exl3.physical_k_compat._physical_k_values."""
    values = set()
    for pack in inners:
        for projection in ("gate", "up", "down"):
            linear = pack[projection]
            k = getattr(linear, "K", None)
            if k is None:
                trellis = getattr(linear, "trellis", None)
                if trellis is None or getattr(trellis, "ndim", 0) != 3:
                    raise RuntimeError(
                        f"EXL3 fused state cannot determine physical K for {projection}"
                    )
                words = int(trellis.shape[-1])
                if words <= 0 or words % 8:
                    raise RuntimeError(
                        f"EXL3 fused state invalid trellis width {words} for {projection}"
                    )
                k = words / 16
            values.add(int(k) if float(k).is_integer() else float(k))
    return values


def _install_frac_runtime(exl3) -> None:
    """Patch the plugin's integer-K assumptions; idempotent."""
    if getattr(exl3, "_naive_frac_k_patch", False):
        return
    moe_cls = getattr(exl3, "Exl3MoEMethod", None)
    fused_apply = getattr(exl3, "apply_exl3_fused_moe", None)
    fat_apply = getattr(exl3, "apply_exl3_batched_fat", None)
    native_apply = getattr(exl3, "_apply_native_fused_moe", None)
    native_dims = getattr(exl3, "_native_moe_dimensions_supported", None)
    if None in (moe_cls, fused_apply, fat_apply, native_apply, native_dims):
        return

    try:
        from vllm_exl3 import physical_k_compat

        physical_k_compat._physical_k_values = _physical_k_values
    except Exception:
        pass

    patched_load = _source_patch(
        moe_cls._load_exl3,
        [(
            "if int(sharded.shape[-1]) % 16 != 0:",
            "if int(sharded.shape[-1]) % 16 not in (0, 8):",
        )],
    )
    moe_cls._load_exl3 = patched_load
    if getattr(exl3, "_load_exl3", None) is patched_load:
        delattr(exl3, "_load_exl3")

    fused_replacements = [
        ('int(getattr(layer, "_exl3_k", 4))', 'getattr(layer, "_exl3_k", 4)')
    ]
    # Decode launch shape. The plugin passes num_active=-1 (no .item() sync), which
    # makes exl3_moe launch max-concurrency 8-SM groups (6 x 8 on GB10) even when a
    # decode step routes only ~4 experts to this rank, idling a third of the SMs.
    # num_active only sizes the grid (exl3_moe.cu:186,277-281; experts are claimed by
    # ticket, so correctness does not depend on it), so a fixed positive hint gives
    # min(concurrency, hint) groups widened to num_sms / groups each.
    hint = os.environ.get("EXL3_MOE_NUM_ACTIVE_HINT", "")
    if hint.strip().isdigit() and int(hint) > 0:
        fused_replacements.append((
            "n_active_host = -1 if _exl3_moe_accepts_num_active(fn) else None",
            f"n_active_host = {int(hint)} if _exl3_moe_accepts_num_active(fn) else None",
        ))
        logger.info("vllm_naive_n05: exl3_moe num_active launch hint = %s", int(hint))
    exl3.apply_exl3_fused_moe = _source_patch(fused_apply, fused_replacements)
    exl3.apply_exl3_batched_fat = _source_patch(
        fat_apply,
        [('int(getattr(gate, "K", 4))', 'getattr(gate, "K", 4)')],
    )
    exl3._apply_native_fused_moe = _source_patch(
        native_apply,
        [(
            'int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 4)))',
            'getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", 4))',
        )],
    )
    exl3._native_moe_dimensions_supported = _source_patch(
        native_dims,
        [(
            'bits = int(getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", -1)))',
            'bits = getattr(layer, "_exl3_k", getattr(layer, "_exl3_bits", -1))',
        )],
    )

    builder = getattr(exl3, "build_exl3_fused_state", None)
    if builder is not None and getattr(builder, "__module__", "") == exl3.__name__:
        exl3._naive_frac_k_from_inners = _frac_k_from_inners
        exl3.build_exl3_fused_state = _source_patch(
            builder,
            [(
                "layer._exl3_k = int(layer._exl3_bits)",
                "layer._exl3_k = _naive_frac_k_from_inners("
                "inners, layer._exl3_bits)",
            )],
        )

    exl3._naive_frac_k_patch = True


def apply() -> None:
    """Patch vllm-exl3 in-process; idempotent and a no-op without the plugin."""
    try:
        from vllm_exl3 import exl3
    except Exception:
        return

    try:
        _patch_config_bits(exl3)
        _install_frac_runtime(exl3)
    except Exception as exc:
        logger.warning("vllm_naive_n05: fractional-K compat shim incomplete: %s", exc)
