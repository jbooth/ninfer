"""JB-NVFP4 ``.ninfer`` artifact builder for Qwen3.8-Flash-Next (qwen4exp).

Source of truth
---------------
* canonical BF16 GGUF (``Qwen3.8-Flash-Next-BF16.gguf``, 354 GB);
* unsloth imatrix file (``imatrix_unsloth_qwen38next.gguf_file``) — per-tensor
  input second-moment bands (``in_sum2``) and per-expert MoE bands;
* HF frontend resources (six ``frontend/*`` files).

Numerics
--------
The encoder is the shared pure-numpy imatrix-aware selection from
``tools/convert/common/encoder_imatrix.py``: the per-object objective is the
importance-weighted MSE of the quantization error (plan §2.4 — "the MSE of
importance × error").  No llama-quantize / C++ on this path.

Canonical invocation
--------------------
::

    /home/robot/workplace/unsloth_env/bin/python -m \
        tools.convert.qwen4exp.convert_jbnvfp4 \
        --model /llm/models/Qwen3.8-Flash-Next-BF16.gguf \
        --imatrix /llm/models/imatrix_unsloth_qwen38next.gguf_file \
        --resources /llm/models/Qwen3.8-Flash-Next/master \
        --out /llm/models/Qwen3.8-Flash-Next-JB-NVFP4.ninfer \
        --workers 32

Layout conventions
------------------
* GGUF stores linears as ``(in, out)`` and MoE experts as
  ``(in, out, E)``; artifact objects are stored ``(out, in)``.
* ``moe/routed_gate_up``: e-major parent, per-expert row block
  ``[gate 640 | up 640]``, K = 2560 — NVFP4 (blockscale-k16-m128x4-v1).
* ``moe/routed_down``: e-major rows, ``[out 2560, in 640]`` — Q6G64_F16S.
* ``moe/router_gate``: ``[routed 512 | shared-gate 1]`` rows, FP32.
* GDN layers (l % 4 != 3): ``attn_qkv`` splits into q|k (4096 rows) and v
  (6144 rows); ``gdn/value_z`` = v rows then the 6144 ``attn_gate`` (z) rows.
* Full-attn layers (l % 4 == 3): source ``attn_q`` interleaves q|gate per
  head (head h: q = rows ``h*512 .. h*512+255``, gate = ``h*512+256 ..
  (h+1)*512-1``); the parent ships ``q | k | gate | v`` de-interleaved.
* HC ``up/down/inject`` are raw weights; HC ``norm`` rows are copied
  verbatim (the GGUF converter already folded ``1 + w``).  The head mixer has
  no inject tensor.
* PLE u64 metadata (prime multipliers, head offsets, head vocab sizes) is
  stored as I32 lo/hi pairs, lo first (the artifact has no I64; plan §4.6).
* Per-layer object write order: ``hc_attn`` (dataflow: the mixer turns the
  4-stream residual into the block input) -> attention/GDN -> ``moe`` ->
  ``hc_ffn`` -> PLE (layer 1 only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact import layouts
from tools.convert.common import encoder_imatrix as enc
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common.inventory import ResourceSpec, TensorSpec

MODEL_ID = "qwen3.8-flash-next"
WEIGHTS_ID = "jbnvfp4"

# --- model geometry (GGUF qwen4exp KV; asserted at preflight) --------------
HIDDEN = 2560
LAYERS = 48
E = 512                 # routed experts
I = 640                 # expert FF width
TOPK = 10
VOCAB = 248320
PLE_ROWS = 320001536
PLE_DIM = 160
HC_DIM = 10240          # 4 streams x 2560
HC_LR = 320
HEADS = 24
HEAD_DIM = 256
KV_DIM = 512            # 2 kv heads x 256
GATE_DIM = 6144         # q rows (24x256) == v rows (48x128) == z rows
V_DIM = 6144
QK_DIM = 4096
ATTN_Q_DIM = 12288      # 24 heads x (256 q + 256 gate)
IDX_K_DIM = 128
NV_ROWS = E * 2 * I     # 655360
NV_K = HIDDEN
Q6_ROWS = E * HIDDEN    # 1310720
Q6_K = I
QKQV_ROWS = GATE_DIM + KV_DIM + GATE_DIM + KV_DIM  # 13312

# --- artifact format/layout names ------------------------------------------
CONTIG = "contiguous-le-v1"
ROWSPLIT = "row-split-k128-v1"
BLOCKSCALE = "blockscale-k16-m128x4-v1"
BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
W8 = "W8G32_F16S"
Q6 = "Q6G64_F16S"
NVFP4 = "NVFP4"

NV_CHUNK = 256          # encode_nvfp4 row chunk (worker RAM bound)
PLE_CHUNK = 2_000_000   # per_layer_token_embedding streaming rows


def _spec(name: str, shape: tuple[int, ...], fmt: str) -> TensorSpec:
    """TensorSpec with the canonical layout for its numeric format."""
    if fmt in (BF16, FP32, I32):
        layout = CONTIG
    elif fmt in (W8, Q6):
        layout = ROWSPLIT
    elif fmt == NVFP4:
        layout = BLOCKSCALE
    else:
        raise ValueError(f"unsupported format {fmt!r}")
    return TensorSpec(name=name, shape=tuple(shape), format=fmt, layout=layout)

# --- fork-inherited reader state (workers) ---------------------------------
G_SRC: GGUFReader
G_IMAT: GGUFReader
SRC_IDS: dict[str, int]
IMAT_IDS: dict[str, int]

# q|gate de-interleave row indexes on the source attn_q output axis (24 heads)
_Q_IDX = np.repeat(np.arange(HEADS) * 2 * HEAD_DIM, HEAD_DIM)          # q rows
_G_IDX = _Q_IDX + HEAD_DIM                                            # gate rows


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def _bf16_f32(raw_u8: np.ndarray) -> np.ndarray:
    """Bit-exact BF16 -> FP32 (bf16 word << 16 as f32 bits)."""
    u16 = raw_u8.view(np.uint16)
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _src(name: str) -> np.ndarray:
    """Source GGUF tensor as a raw memmap view (uint8 for BF16, f32 as-is)."""
    return G_SRC.get_tensor(SRC_IDS[name]).data


def _imat(name: str) -> np.ndarray:
    return G_IMAT.get_tensor(IMAT_IDS[name]).data


def _kv(name: str):
    return G_SRC.get_field(name).contents()


def _check_shape(where: str, shape: tuple, want: tuple) -> None:
    if tuple(shape) != tuple(want):
        raise ValueError(f"unexpected shape for {where}: {tuple(shape)}, want {tuple(want)}")


def _orient(where: str, data: np.ndarray, want: tuple) -> np.ndarray:
    """Return data in artifact orientation.

    The gguf reader returns C-order data with the innermost (first metadata)
    dimension last, so 2-D linears arrive already as (out, in) and 3-D expert
    stacks as (E, out, in).  Only 2-D conv kernels (metadata (K, C)) arrive
    transposed relative to their artifact shape; transpose in that case.
    """
    if tuple(data.shape) == tuple(want):
        return data
    if data.ndim >= 2 and tuple(data.shape[::-1]) == tuple(want):
        return np.ascontiguousarray(data.T)
    raise ValueError(f"cannot orient {where}: {tuple(data.shape)}, want {tuple(want)}")


# ---------------------------------------------------------------------------
# worker tasks (module level so they fork-inherit the readers)
# ---------------------------------------------------------------------------

def _task_amax(args: tuple[int, str, int]) -> float:
    layer, src, e = args
    raw = np.ascontiguousarray(_src(src)[e])
    return float(np.abs(_bf16_f32(raw)).max())


def _task_gate_up(args: tuple[int, int, np.float32]) -> tuple[bytes, bytes]:
    """One expert's fused gate|up NVFP4 encode (rows gate-first)."""
    layer, e, dw = args
    g = _bf16_f32(np.ascontiguousarray(_src(f"blk.{layer}.ffn_gate_exps.weight")[e]))
    u = _bf16_f32(np.ascontiguousarray(_src(f"blk.{layer}.ffn_up_exps.weight")[e]))
    t = np.empty((2 * I, HIDDEN), np.float32)
    t[:I] = g * dw
    t[I:] = u * dw
    band = np.empty((2 * I, HIDDEN), np.float32)
    band[:I] = _imat(f"blk.{layer}.ffn_gate_exps.weight.in_sum2")[e]
    band[I:] = _imat(f"blk.{layer}.ffn_up_exps.weight.in_sum2")[e]
    packed, words = enc.encode_nvfp4(t, band, row_chunk=NV_CHUNK)
    return packed.tobytes(), words.tobytes()


def _task_down(args: tuple[int, int]) -> tuple[bytes, bytes]:
    """One expert's down Q6 encode."""
    layer, e = args
    d = _bf16_f32(np.ascontiguousarray(_src(f"blk.{layer}.ffn_down_exps.weight")[e]))
    band = _imat(f"blk.{layer}.ffn_down_exps.weight.in_sum2")[e]
    codes, scales = enc.encode_row_split(d, band, gs=64, qmax=31)
    return codes.tobytes(), scales.tobytes()


def _task_w8_part(args: tuple[int, str, int, int, int, str]) -> tuple[bytes, bytes, int]:
    """A W8G32 part: source rows (mode) + 1-D imatrix band."""
    layer, src, mode, r0, r1, band_src = args
    u16 = _src(src).view(np.uint16)                       # already (out, in) bf16 words
    if mode == 0:
        blk = np.ascontiguousarray(u16)
    elif mode == 1:
        blk = np.ascontiguousarray(u16[r0:r1])
    elif mode == 2:
        blk = np.ascontiguousarray(u16[_Q_IDX])
    elif mode == 3:
        blk = np.ascontiguousarray(u16[_G_IDX])
    else:
        raise ValueError(f"unknown row mode {mode}")
    w = _bf16_f32(blk)
    band = _imat(f"blk.{layer}.{band_src}.in_sum2")
    _check_shape(f"band {band_src} layer {layer}", band.shape, (w.shape[1],))
    codes, scales = enc.encode_row_split(w, band, gs=32, qmax=127)
    return codes.tobytes(), scales.tobytes(), int(w.shape[0])


# ---------------------------------------------------------------------------
# payload producers (parent process)
# ---------------------------------------------------------------------------

def _bf16_direct(name: str, shape: tuple[int, int]) -> bytes:
    u16 = _src(name).view(np.uint16)
    blk = _orient(name, u16, shape)
    return layouts.encode_direct(torch.from_numpy(blk).view(torch.bfloat16), BF16)


def _f32_direct(name: str, shape: tuple[int, ...]) -> bytes:
    blk = _orient(name, np.asarray(_src(name)), shape)
    return layouts.encode_direct(torch.from_numpy(blk), FP32)


def _i32_u64_pairs(key: str, count: int) -> bytes:
    v = np.asarray(_kv(key)).astype(np.uint64)
    _check_shape(key, v.shape, (count,))
    lo = (v & np.uint64(0xFFFFFFFF)).astype(np.int32)
    hi = (v >> np.uint64(32)).astype(np.int32)
    out = np.empty(2 * count, np.int32)
    out[0::2] = lo
    out[1::2] = hi
    return layouts.encode_direct(torch.from_numpy(out), I32)


def _ple_table_chunks(n_rows: int):
    u16 = _src("per_layer_token_embd.weight").view(np.uint16)   # (rows, 160): already token-major
    _check_shape("per_layer_token_embd.weight", u16.shape, (PLE_ROWS, PLE_DIM))
    for i0 in range(0, n_rows, PLE_CHUNK):
        i1 = min(i0 + PLE_CHUNK, n_rows)
        yield u16[i0:i1].tobytes()


# ---------------------------------------------------------------------------
# tensor existence preflight
# ---------------------------------------------------------------------------

_HC_NAMES = (
    "hc_attn_down.weight", "hc_attn_inject.weight", "hc_attn_norm.weight",
    "hc_attn_up.weight", "hc_ffn_down.weight", "hc_ffn_inject.weight",
    "hc_ffn_norm.weight", "hc_ffn_up.weight",
)
_MOE_NAMES = (
    "ffn_down_exps.weight", "ffn_down_shexp.weight", "ffn_gate_exps.weight",
    "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
    "ffn_up_exps.weight", "ffn_up_shexp.weight",
)


def _layer_tensor_names(layer: int) -> tuple[str, ...]:
    base = (
        "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_alpha.weight",
        "ssm_beta.weight", "attn_qkv.weight", "attn_gate.weight",
        "ssm_norm.weight", "ssm_out.weight",
    ) if layer % 4 != 3 else (
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "attn_q_norm.weight", "attn_k_norm.weight",
        "indexer.q_proj.weight", "indexer.k_proj.weight",
        "indexer.q_norm.weight", "indexer.k_norm.weight",
    )
    return base + _MOE_NAMES + _HC_NAMES


def _layer_band_names(layer: int) -> tuple[str, ...]:
    base = ("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight") \
        if layer % 4 != 3 else (
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")
    base = base + ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")
    return base + ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


def _preflight(layers: tuple[int, ...]) -> dict[str, object]:
    expect = {
        "qwen4exp.block_count": LAYERS,
        "qwen4exp.embedding_length": HIDDEN,
        "qwen4exp.expert_count": E,
        "qwen4exp.expert_feed_forward_length": I,
        "qwen4exp.expert_used_count": TOPK,
        "qwen4exp.hyper_connection.count": HC_DIM // HIDDEN,
        "qwen4exp.hyper_connection.low_rank": HC_LR,
        "qwen4exp.ple.heads_per_ngram": 8,
        "qwen4exp.ple.ngram_size": 3,
    }
    kv: dict[str, object] = {}
    for key, want in expect.items():
        got = _kv(key)
        kv[key] = got
        if got != want:
            raise ValueError(f"KV mismatch: {key} = {got!r}, want {want!r}")
    if str(_kv("general.architecture")) != "qwen4exp":
        raise ValueError("architecture is not qwen4exp")
    kv["general.architecture"] = "qwen4exp"
    for key, want in (("qwen4exp.ple.layer_multipliers", 3),
                      ("qwen4exp.ple.head_offsets", 16),
                      ("qwen4exp.ple.head_vocab_sizes", 16)):
        arr = np.asarray(_kv(key))
        _check_shape(key, arr.shape, (want,))
        kv[key] = f"u64[{want}]"
    # tensor + band existence for every planned layer
    for layer in layers:
        prefix = f"blk.{layer}."
        for base in _layer_tensor_names(layer):
            if prefix + base not in SRC_IDS:
                raise ValueError(f"missing source tensor {prefix}{base}")
        for base in _layer_band_names(layer):
            if prefix + base + ".in_sum2" not in IMAT_IDS:
                raise ValueError(f"missing imatrix band {prefix}{base}.in_sum2")
    for name in ("token_embd.weight", "per_layer_token_embd.weight", "output.weight",
                 "output_hc_up.weight", "output_hc_down.weight", "output_hc_norm.weight",
                 "blk.1.ple_key.weight", "blk.1.ple_value.weight", "blk.1.ple_conv1d.weight",
                 "blk.1.ple_norm_query.weight", "blk.1.ple_norm_key.weight",
                 "blk.1.ple_norm_conv.weight"):
        if name not in SRC_IDS:
            raise ValueError(f"missing source tensor {name}")
    return kv


# ---------------------------------------------------------------------------
# object plan + producer jobs
# ---------------------------------------------------------------------------

LAYER_STATES: dict[int, "_LayerState"] = {}
RESOURCE_NAMES = (
    "frontend/tokenizer.json",
    "frontend/tokenizer_config.json",
    "frontend/chat_template.jinja",
    "frontend/generation_config.json",
    "frontend/preprocessor_config.json",
    "frontend/video_preprocessor_config.json",
)


class _LayerState:
    """Per-layer pool state (lazy).

    ensure() runs the 1024-task amax pre-pass, computes the layer d_w and
    the fused gate|up input divisor, and submits the three encode maps
    (gate_up NVFP4, down Q6, W8 parts).  Producers pull from the maps in
    task order; the W8 part order matches the object row order:

      GDN:    0 qk | 1 v | 2 z | 3 ssm_out | 4,5 shared gate|up | 6 shared down
      full:   0 q | 1 k | 2 gate | 3 v | 4 attn_out | 5,6 shared gate|up | 7 shared down
    """

    def __init__(self, layer: int, pool: ProcessPoolExecutor) -> None:
        self.layer = layer
        self.pool = pool
        self.it_gate_up = None
        self.it_down = None
        self.it_w8 = None
        self.d_w = None
        self.d_w_f = None
        self.in_div = None
        self.report: dict[str, object] = {}

    def ensure(self) -> None:
        if self.d_w is not None:
            return
        layer = self.layer
        amax_tasks = [
            (layer, f"blk.{layer}.{base}", e)
            for base in ("ffn_gate_exps.weight", "ffn_up_exps.weight")
            for e in range(E)
        ]
        a = float(max(self.pool.map(_task_amax, amax_tasks, chunksize=16)))
        dw = np.float32(np.float32(2688.0) / np.float32(a))
        self.d_w = struct.pack("<f", dw)
        self.d_w_f = float(dw)
        band_g = _imat(f"blk.{layer}.ffn_gate_exps.weight.in_sum2")
        band_u = _imat(f"blk.{layer}.ffn_up_exps.weight.in_sum2")
        _check_shape(f"gate band layer {layer}", band_g.shape, (E, HIDDEN))
        _check_shape(f"up band layer {layer}", band_u.shape, (E, HIDDEN))
        self.in_div = enc.input_divisor(np.concatenate([band_g, band_u]))
        self.it_gate_up = self.pool.map(
            _task_gate_up, [(layer, e, dw) for e in range(E)], chunksize=8)
        self.it_down = self.pool.map(
            _task_down, [(layer, e) for e in range(E)], chunksize=8)
        if layer % 4 != 3:
            w8_tasks = [
                (layer, f"blk.{layer}.attn_qkv.weight", 1, 0, QK_DIM, "attn_qkv.weight"),
                (layer, f"blk.{layer}.attn_qkv.weight", 1, QK_DIM, HC_DIM, "attn_qkv.weight"),
                (layer, f"blk.{layer}.attn_gate.weight", 0, 0, 0, "attn_gate.weight"),
                (layer, f"blk.{layer}.ssm_out.weight", 0, 0, 0, "ssm_out.weight"),
            ]
        else:
            w8_tasks = [
                (layer, f"blk.{layer}.attn_q.weight", 2, 0, 0, "attn_q.weight"),
                (layer, f"blk.{layer}.attn_k.weight", 0, 0, 0, "attn_k.weight"),
                (layer, f"blk.{layer}.attn_q.weight", 3, 0, 0, "attn_q.weight"),
                (layer, f"blk.{layer}.attn_v.weight", 0, 0, 0, "attn_v.weight"),
                (layer, f"blk.{layer}.attn_output.weight", 0, 0, 0, "attn_output.weight"),
            ]
        w8_tasks += [
            (layer, f"blk.{layer}.ffn_gate_shexp.weight", 0, 0, 0, "ffn_gate_shexp.weight"),
            (layer, f"blk.{layer}.ffn_up_shexp.weight", 0, 0, 0, "ffn_up_shexp.weight"),
            (layer, f"blk.{layer}.ffn_down_shexp.weight", 0, 0, 0, "ffn_down_shexp.weight"),
        ]
        self.it_w8 = self.pool.map(_task_w8_part, w8_tasks, chunksize=1)
        self.report = {
            "amax": a,
            "d_w": self.d_w_f,
            "in_div": struct.unpack("<f", self.in_div)[0],
        }

    def w8(self, k: int):
        self.ensure()
        return _iter_take(self.it_w8, k)


def _iter_take(it, n: int):
    for _ in range(n):
        yield next(it)


def _w8_payload(shape: tuple[int, int], it) -> bytes:
    n, k = shape
    g = k // 32
    codes = np.empty((n, g, 32), np.int8)
    scales = np.empty((n, g), np.float16)
    r = 0
    for cb, sb, nr in it:
        codes[r:r + nr] = np.frombuffer(cb, np.int8).reshape(nr, g, 32)
        scales[r:r + nr] = np.frombuffer(sb, np.float16).reshape(nr, g)
        r += nr
    if r != n:
        raise RuntimeError(f"w8 assembly short: {r}/{n} rows")
    return layouts.encode_row_split(
        torch.from_numpy(codes), torch.from_numpy(scales), W8, shape)


def _parity_expert0_nvfp4(st: "_LayerState", packed: np.ndarray, words: np.ndarray) -> None:
    """Re-encode expert 0 in the parent; require exact byte parity."""
    layer = st.layer
    g = _bf16_f32(np.ascontiguousarray(_src(f"blk.{layer}.ffn_gate_exps.weight")[0]))
    u = _bf16_f32(np.ascontiguousarray(_src(f"blk.{layer}.ffn_up_exps.weight")[0]))
    t = np.empty((2 * I, HIDDEN), np.float32)
    t[:I] = g * np.float32(st.d_w_f)
    t[I:] = u * np.float32(st.d_w_f)
    band = np.empty((2 * I, HIDDEN), np.float32)
    band[:I] = _imat(f"blk.{layer}.ffn_gate_exps.weight.in_sum2")[0]
    band[I:] = _imat(f"blk.{layer}.ffn_up_exps.weight.in_sum2")[0]
    p0, w0 = enc.encode_nvfp4(t, band, row_chunk=NV_CHUNK)
    if not (np.array_equal(packed[:2 * I], p0) and np.array_equal(words[:2 * I], w0)):
        raise RuntimeError(f"nvfp4 expert-0 byte parity failed (layer {layer})")


def _parity_expert0_down(st: "_LayerState", codes: np.ndarray, scales: np.ndarray) -> None:
    layer = st.layer
    d = _bf16_f32(np.ascontiguousarray(_src(f"blk.{layer}.ffn_down_exps.weight")[0]))
    band = _imat(f"blk.{layer}.ffn_down_exps.weight.in_sum2")[0]
    c0, s0 = enc.encode_row_split(d, band, gs=64, qmax=31)
    if not (np.array_equal(codes[:HIDDEN], c0) and np.array_equal(scales[:HIDDEN], s0)):
        raise RuntimeError(f"q6 expert-0 byte parity failed (layer {layer})")


def _gate_up_producer(st: "_LayerState") -> bytes:
    st.ensure()
    packed = np.empty((NV_ROWS, NV_K // 2), np.uint8)
    words = np.empty((NV_ROWS, NV_K // 16), np.uint8)
    for e in range(E):
        pb, wb = next(st.it_gate_up)
        packed[e * 2 * I:(e + 1) * 2 * I] = np.frombuffer(pb, np.uint8).reshape(2 * I, NV_K // 2)
        words[e * 2 * I:(e + 1) * 2 * I] = np.frombuffer(wb, np.uint8).reshape(2 * I, NV_K // 16)
    _parity_expert0_nvfp4(st, packed, words)
    return layouts.encode_nvfp4(
        torch.from_numpy(packed), torch.from_numpy(words), st.d_w, (NV_ROWS, NV_K))


def _down_producer(st: "_LayerState") -> bytes:
    st.ensure()
    codes = np.empty((Q6_ROWS, Q6_K // 64, 64), np.int8)
    scales = np.empty((Q6_ROWS, Q6_K // 64), np.float16)
    for e in range(E):
        cb, sb = next(st.it_down)
        r0 = e * HIDDEN
        codes[r0:r0 + HIDDEN] = np.frombuffer(cb, np.int8).reshape(HIDDEN, Q6_K // 64, 64)
        scales[r0:r0 + HIDDEN] = np.frombuffer(sb, np.float16).reshape(HIDDEN, Q6_K // 64)
    _parity_expert0_down(st, codes, scales)
    return layouts.encode_row_split(
        torch.from_numpy(codes), torch.from_numpy(scales), Q6, (Q6_ROWS, Q6_K))


def _divisor_producer(st: "_LayerState") -> bytes:
    st.ensure()
    return layouts.encode_direct(torch.from_numpy(np.frombuffer(st.in_div, np.float32)), FP32)


def _router_gate(layer: int) -> bytes:
    r = np.ascontiguousarray(_src(f"blk.{layer}.ffn_gate_inp.weight"))
    s = np.ascontiguousarray(_src(f"blk.{layer}.ffn_gate_inp_shexp.weight"))
    _check_shape(f"router layer {layer}", r.shape, (E, HIDDEN))
    _check_shape(f"router shared layer {layer}", s.shape, (1, HIDDEN))
    return layouts.encode_direct(torch.from_numpy(np.concatenate([r, s], axis=0)), FP32)


def _stack_bf16(names: tuple[str, ...], shape: tuple[int, int]) -> bytes:
    blk = np.ascontiguousarray(
        np.concatenate([_src(n).view(np.uint16) for n in names], axis=0))
    _check_shape(" + ".join(names), blk.shape, shape)
    return layouts.encode_direct(torch.from_numpy(blk).view(torch.bfloat16), BF16)


def _hc_units(layer: int, section: str) -> list[tuple[TensorSpec, object]]:
    L = f"text/layers/{layer}"
    S = f"blk.{layer}.hc_{section}"
    units: list[tuple[TensorSpec, object]] = []
    for name, shape in (("up", (HC_DIM, HC_LR)), ("down", (HC_LR, HC_DIM)),
                        ("inject", (HC_DIM // HIDDEN, HC_DIM))):
        units.append((_spec(f"{L}/hc_{section}/{name}", shape, BF16),
                      lambda n=S + "_" + name + ".weight", sh=shape: _bf16_direct(n, sh)))
    units.append((_spec(f"{L}/hc_{section}/norm", (HC_DIM,), FP32),
                  lambda n=S + "_norm.weight": _f32_direct(n, (HC_DIM,))))
    return units


def _layer_units(layer: int, st: _LayerState) -> list[tuple[TensorSpec, object]]:
    L = f"text/layers/{layer}"
    units = _hc_units(layer, "attn")
    if layer % 4 != 3:
        units += [
            (_spec(f"{L}/gdn/query_key", (QK_DIM, HIDDEN), W8),
             lambda s=st: _w8_payload((QK_DIM, HIDDEN), s.w8(1))),
            (_spec(f"{L}/gdn/value_z", (V_DIM + GATE_DIM, HIDDEN), W8),
             lambda s=st: _w8_payload((V_DIM + GATE_DIM, HIDDEN), s.w8(2))),
            (_spec(f"{L}/gdn/a_b_projection", (2 * 48, HIDDEN), BF16),
             lambda l=layer: _stack_bf16(
                 (f"blk.{l}.ssm_alpha.weight", f"blk.{l}.ssm_beta.weight"), (2 * 48, HIDDEN))),
            (_spec(f"{L}/gdn/a_log", (48,), FP32),
             lambda l=layer: _f32_direct(f"blk.{l}.ssm_a", (48,))),
            (_spec(f"{L}/gdn/dt_bias", (48,), FP32),
             lambda l=layer: _f32_direct(f"blk.{l}.ssm_dt.bias", (48,))),
            (_spec(f"{L}/gdn/convolution", (4, HC_DIM), FP32),
             lambda l=layer: _f32_direct(f"blk.{l}.ssm_conv1d.weight", (4, HC_DIM))),
            (_spec(f"{L}/gdn/norm", (128,), FP32),
             lambda l=layer: _f32_direct(f"blk.{l}.ssm_norm.weight", (128,))),
            (_spec(f"{L}/gdn/output", (HIDDEN, V_DIM), W8),
             lambda s=st: _w8_payload((HIDDEN, V_DIM), s.w8(1))),
        ]
    else:
        units += [
            (_spec(f"{L}/attention/query_key_gate_value", (QKQV_ROWS, HIDDEN), W8),
             lambda s=st: _w8_payload((QKQV_ROWS, HIDDEN), s.w8(4))),
            (_spec(f"{L}/attention/query_norm", (HEAD_DIM,), FP32),
             lambda l=layer: _f32_direct(f"blk.{l}.attn_q_norm.weight", (HEAD_DIM,))),
            (_spec(f"{L}/attention/key_norm", (HEAD_DIM,), FP32),
             lambda l=layer: _f32_direct(f"blk.{l}.attn_k_norm.weight", (HEAD_DIM,))),
            (_spec(f"{L}/attention/output", (HIDDEN, V_DIM), W8),
             lambda s=st: _w8_payload((HIDDEN, V_DIM), s.w8(1))),
            (_spec(f"{L}/indexer/query_proj", (KV_DIM, HIDDEN), BF16),
             lambda l=layer: _bf16_direct(f"blk.{l}.indexer.q_proj.weight", (KV_DIM, HIDDEN))),
            (_spec(f"{L}/indexer/key_proj", (IDX_K_DIM, HIDDEN), BF16),
             lambda l=layer: _bf16_direct(f"blk.{l}.indexer.k_proj.weight", (IDX_K_DIM, HIDDEN))),
            (_spec(f"{L}/indexer/query_norm", (IDX_K_DIM,), FP32),
             lambda l=layer: _f32_direct(f"blk.{l}.indexer.q_norm.weight", (IDX_K_DIM,))),
            (_spec(f"{L}/indexer/key_norm", (IDX_K_DIM,), FP32),
             lambda l=layer: _f32_direct(f"blk.{l}.indexer.k_norm.weight", (IDX_K_DIM,))),
        ]
    units += [
        (_spec(f"{L}/moe/routed_gate_up", (NV_ROWS, NV_K), NVFP4),
         lambda s=st: _gate_up_producer(s)),
        (_spec(f"{L}/moe/gate_up_input_divisor", (), FP32),
         lambda s=st: _divisor_producer(s)),
        (_spec(f"{L}/moe/routed_down", (Q6_ROWS, Q6_K), Q6),
         lambda s=st: _down_producer(s)),
        (_spec(f"{L}/moe/router_gate", (E + 1, HIDDEN), FP32),
         lambda l=layer: _router_gate(l)),
        (_spec(f"{L}/moe/shared_gate_up", (2 * I, HIDDEN), W8),
         lambda s=st: _w8_payload((2 * I, HIDDEN), s.w8(2))),
        (_spec(f"{L}/moe/shared_down", (HIDDEN, I), W8),
         lambda s=st: _w8_payload((HIDDEN, I), s.w8(1))),
    ]
    units += _hc_units(layer, "ffn")
    if layer == 1:
        units += [
            (_spec("text/ple/key", (HC_DIM, HIDDEN), BF16),
             lambda: _bf16_direct("blk.1.ple_key.weight", (HC_DIM, HIDDEN))),
            (_spec("text/ple/value", (HIDDEN, HIDDEN), BF16),
             lambda: _bf16_direct("blk.1.ple_value.weight", (HIDDEN, HIDDEN))),
            (_spec("text/ple/convolution", (4, HC_DIM), BF16),
             lambda: _bf16_direct("blk.1.ple_conv1d.weight", (4, HC_DIM))),
            (_spec("text/ple/norm_query", (HC_DIM,), FP32),
             lambda: _f32_direct("blk.1.ple_norm_query.weight", (HC_DIM,))),
            (_spec("text/ple/norm_key", (HC_DIM,), FP32),
             lambda: _f32_direct("blk.1.ple_norm_key.weight", (HC_DIM,))),
            (_spec("text/ple/norm_conv", (HC_DIM,), FP32),
             lambda: _f32_direct("blk.1.ple_norm_conv.weight", (HC_DIM,))),
        ]
    return units


def build_plan(layers: tuple[int, ...], smoke: bool):
    """Return (specs, producers) in exact write order."""
    specs: list = [ResourceSpec(r) for r in RESOURCE_NAMES]
    producers: dict[str, object] = {}

    def add(spec: TensorSpec, producer):
        specs.append(spec)
        producers[spec.name] = producer

    add(_spec("text/token_embedding", (VOCAB, HIDDEN), BF16),
        lambda: _bf16_direct("token_embd.weight", (VOCAB, HIDDEN)))
    add(_spec("text/ple/layer_multipliers", (6,), I32),
        lambda: _i32_u64_pairs("qwen4exp.ple.layer_multipliers", 3))
    add(_spec("text/ple/head_offsets", (32,), I32),
        lambda: _i32_u64_pairs("qwen4exp.ple.head_offsets", 16))
    add(_spec("text/ple/head_vocab_sizes", (32,), I32),
        lambda: _i32_u64_pairs("qwen4exp.ple.head_vocab_sizes", 16))
    if not smoke:
        add(_spec("text/per_layer_token_embedding", (PLE_ROWS, PLE_DIM), BF16),
            lambda: _ple_table_chunks(PLE_ROWS))
    for layer in layers:
        for spec, producer in _layer_units(layer, LAYER_STATES[layer]):
            add(spec, producer)
    add(_spec("text/output_hc/up", (HC_DIM, HC_LR), BF16),
        lambda: _bf16_direct("output_hc_up.weight", (HC_DIM, HC_LR)))
    add(_spec("text/output_hc/down", (HC_LR, HC_DIM), BF16),
        lambda: _bf16_direct("output_hc_down.weight", (HC_LR, HC_DIM)))
    add(_spec("text/output_hc/norm", (HC_DIM,), FP32),
        lambda: _f32_direct("output_hc_norm.weight", (HC_DIM,)))
    add(_spec("text/output_head", (VOCAB, HIDDEN), BF16),
        lambda: _bf16_direct("output.weight", (VOCAB, HIDDEN)))
    return specs, producers


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parse_layers(arg: str | None) -> tuple[int, ...]:
    if not arg:
        return tuple(range(LAYERS))
    out: set[int] = set()
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    layers = tuple(sorted(out))
    if not layers or min(layers) < 0 or max(layers) >= LAYERS:
        raise ValueError(f"layer selection out of range [0, {LAYERS}): {arg!r}")
    return layers


def convert(args: argparse.Namespace) -> int:
    global G_SRC, G_IMAT, SRC_IDS, IMAT_IDS, LAYER_STATES
    t0 = time.time()
    print(f"[jbnvfp4] opening {args.model}", flush=True)
    G_SRC = GGUFReader(args.model)
    G_IMAT = GGUFReader(args.imatrix)
    SRC_IDS = {t.name: i for i, t in enumerate(G_SRC.tensors)}
    IMAT_IDS = {t.name: i for i, t in enumerate(G_IMAT.tensors)}
    layers = _parse_layers(args.layers)
    if args.smoke and not args.layers:
        layers = (0, 1, 3)
    print(f"[jbnvfp4] layers={layers} workers={args.workers} smoke={args.smoke}", flush=True)
    kv = _preflight(layers)
    print(f"[jbnvfp4] preflight OK ({len(SRC_IDS)} source tensors, "
          f"{len(IMAT_IDS)} imatrix tensors)", flush=True)

    resource_map = {}
    for name in RESOURCE_NAMES:
        data = Path(args.resources, name.removeprefix("frontend/")).read_bytes()
        if not data:
            raise ValueError(f"empty frontend resource {name}")
        resource_map[name] = data

    pool = ProcessPoolExecutor(max_workers=args.workers)
    ple_check = None
    try:
        LAYER_STATES = {l: _LayerState(l, pool) for l in layers}
        specs, producers = build_plan(layers, args.smoke)
        plan = family_conversion.build_object_plan(specs, resource_map)
        total_bytes = plan.payload_span_bytes
        by_format: dict[str, int] = {}
        for obj in plan.objects:
            fmt = getattr(obj, "format", "resource")
            by_format[fmt] = by_format.get(fmt, 0) + 1
        print(f"[jbnvfp4] plan: {len(specs)} objects, {total_bytes / 1e9:.3f} GB payload; "
              + " ".join(f"{k}:{v}" for k, v in sorted(by_format.items())), flush=True)
        if args.plan_only:
            print("[jbnvfp4] --plan-only: no artifact written", flush=True)
            return 0

        if args.smoke:
            n_check = 1_048_576
            buf = bytearray()
            for chunk in _ple_table_chunks(n_check):
                buf += chunk
            ple_check = {
                "rows": n_check,
                "bytes": len(buf),
                "sha256": hashlib.sha256(buf).hexdigest(),
            }
            print(f"[jbnvfp4] ple_table path check: {n_check} rows, "
                  f"sha256={ple_check['sha256'][:16]}...", flush=True)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(out_path, ArtifactIdentity(MODEL_ID, WEIGHTS_ID), plan.specs) as writer:
            if writer.objects != plan.objects:
                raise RuntimeError("writer object plan differs from build plan")
            for i, spec in enumerate(specs, start=1):
                name = spec.name
                payload = resource_map[name] if isinstance(spec, ResourceSpec) else producers[name]()
                writer.write(name, payload)
                print(f"[{i}/{len(specs)}] {name}  t={time.time() - t0:7.1f}s", flush=True)
        elapsed = time.time() - t0
        file_bytes = out_path.stat().st_size
        report = {
            "identity": {"model_id": MODEL_ID, "weights_id": WEIGHTS_ID},
            "inputs": {
                "model": str(args.model),
                "imatrix": str(args.imatrix),
                "resources": str(args.resources),
                "out": str(out_path),
            },
            "kv_checks": {k: str(v) for k, v in kv.items()},
            "layers": {str(l): LAYER_STATES[l].report for l in layers},
            "objects": len(specs),
            "payload_bytes": total_bytes,
            "file_bytes": file_bytes,
            "workers": args.workers,
            "smoke": bool(args.smoke),
            "ple_table_check": ple_check,
            "elapsed_s": round(elapsed, 1),
            "encoder": "tools/convert/common/encoder_imatrix.py (pure numpy, imatrix-aware)",
            "conventions": {
                "gguf_layout": "source (in,out) / (in,out,E); artifact (out,in)",
                "routed_gate_up": "e-major, per-expert [gate 640 | up 640], K=2560, NVFP4, d_w=f32(2688/amax)",
                "routed_down": "e-major, Q6G64_F16S (int6, group-64)",
                "input_divisor": "f32(2688 / max_j sqrt(band[j])) over gate+up bands",
                "w8": "W8G32_F16S (int8, group-32) with 1-D per-source imatrix bands",
                "hc_norm": "verbatim from GGUF (1+w already folded by the GGUF converter)",
                "hc_head": "no inject tensor at the head (graph passes nullptr)",
                "ple_u64": "I32 lo/hi pairs, lo first (no I64 in the artifact)",
                "ple_table": "BF16 [320001536,160] = transpose of GGUF [160,320001536] (inner-first storage)",
                "full_attn_deinterleave": "head h: q=rows h*512..h*512+255, gate=h*512+256..(h+1)*512-1",
            },
        }
        report_path = out_path.with_suffix(out_path.suffix + ".conversion.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[jbnvfp4] done in {elapsed:.1f}s: {file_bytes / 1e9:.3f} GB -> {out_path}", flush=True)
        print(f"[jbnvfp4] report -> {report_path}", flush=True)
        return 0
    finally:
        pool.shutdown(wait=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Build the JB-NVFP4 .ninfer artifact for Qwen3.8-Flash-Next "
                    "(qwen4exp) from the canonical BF16 GGUF + unsloth imatrix.")
    ap.add_argument("--model", required=True, help="BF16 GGUF (354 GB)")
    ap.add_argument("--imatrix", required=True, help="unsloth imatrix GGUF")
    ap.add_argument("--resources", required=True,
                    help="HF dir with the six frontend resource files")
    ap.add_argument("--out", required=True, help="output .ninfer path")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--layers", default=None,
                    help="layer subset, e.g. '0-3' or '0,1,3' (default: all 48)")
    ap.add_argument("--smoke", action="store_true",
                    help="small subset run (layers 0,1,3; skips the 102 GB PLE table)")
    ap.add_argument("--plan-only", action="store_true",
                    help="build and print the object plan; write nothing")
    args = ap.parse_args(argv)
    sys.exit(convert(args))


if __name__ == "__main__":
    main()
