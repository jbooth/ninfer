"""JB-NVFP4 ``.ninfer`` artifact builder for Qwen3.8-Flash-Next (qwen4exp).

Source of truth
---------------
* HF safetensors checkpoint directory (``Qwen3.8-Flash-Next/master``, 131
  shards, ~351 GB); the canonical weight source;
* unsloth imatrix file (``imatrix_unsloth_qwen38next.gguf_file``) — per-tensor
  input second-moment bands (``in_sum2``) and per-expert MoE bands (GGUF
  tensor names, consumed unchanged);
* HF frontend resources (six ``frontend/*`` files).

Numerics
--------
The encoder is the shared pure-numpy imatrix-aware selection from
``tools/convert/common/encoder_imatrix.py``: the per-object objective is the
importance-weighted MSE of the quantization error (plan §2.4 — "the MSE of
importance × error").  No llama-quantize / C++ on this path.

Source transforms
-----------------
The checkpoint tensors are the raw HF values; the artifact layout matches the
llama.cpp ``qwen4exp`` GGUF conventions (``llama.cpp/conversion/qwen4exp.py``
+ ``qwen.py`` ``_LinearAttentionVReorderBase``), which were verified byte-
exact against the shipped BF16 GGUF built from this checkpoint:

* GDN V-head tiling: with ``num_k_heads=16 < num_v_heads=48`` (``r=3``), the
  HF weights store V heads grouped by K head ``[G0_v0..G0_v2, G1_v0..]``;
  the artifact stores them tiled ``[G0_v0, G1_v0, G2_v0, G0_v1, ...]``.
  Applied to the V rows of ``in_proj_qkv`` (rows 4096:10240), all rows of
  ``in_proj_z``, all 48 rows of ``in_proj_a/b``, the 48 elements of
  ``A_log``/``dt_bias``, the V channel block (6144 of 10240) of ``conv1d``,
  and the 6144 input columns of ``out_proj``.
* ``gdn/a_log`` = ``-exp(reordered(A_log))`` (torch FP32 exp), i.e. the
  linear decay rate already sign-flipped.
* Every gamma norm stored as zero-centred in the checkpoint is folded to
  ``f32(1.0) + f32(bf16(w))``: per-layer HC norms, head mixer HC norm, full-
  attention q/k norms, indexer q/k norms, PLE query/key/conv norms.  The GDN
  internal 128-dim norm is NOT folded.
* Full-attention ``q_proj`` interleaves q|gate per head (head h: q = rows
  ``h*512 .. h*512+255``, gate = ``h*512+256 .. (h+1)*512-1``); the parent
  ships ``q | k | gate | v`` de-interleaved.
* MoE experts arrive stacked ``[E, 1280, 2560]``; per expert the first 640
  rows are gate and the last 640 are up (verified byte-exact against the
  GGUF split).
* PLE u64 metadata (prime multipliers, head offsets, head vocab sizes) is
  read from the checkpoint I64 tensors ``layers.1.ple.ple_embedding.*``
  and stored as I32 lo/hi pairs, lo first (the artifact has no I64).
* The PLE table is the 128 shards ``ngram_embedding.shard_{0..127}.weight``
  concatenated in ascending shard index — the canonical row layout.
* Head mixer has no inject tensor (graph passes nullptr).

Canonical invocation
--------------------
::

    /home/robot/workplace/unsloth_env/bin/python -m \
        tools.convert.qwen4exp.convert_jbnvfp4 \
        --model /llm/models/Qwen3.8-Flash-Next/master \
        --imatrix /llm/models/imatrix_unsloth_qwen38next.gguf_file \
        --resources /llm/models/Qwen3.8-Flash-Next/master \
        --out /llm/models/Qwen3.8-Flash-Next-JB-NVFP4.ninfer \
        --workers 32

Layout conventions
------------------
* HF linears are ``(out, in)`` and MoE experts ``[E, out*2|out, in]``;
  artifact objects are stored ``(out, in)``.
* ``moe/routed_gate_up``: e-major parent, per-expert row block
  ``[gate 640 | up 640]``, K = 2560 — NVFP4 (blockscale-k16-m128x4-v1).
* ``moe/routed_down``: e-major rows, ``[out 2560, in 640]`` — Q6G64_F16S.
* ``moe/router_gate``: ``[routed 512 | shared-gate 1]`` rows, FP32.
* GDN layers (l % 4 != 3): ``in_proj_qkv`` splits into q|k (4096 rows) and v
  (6144 rows, tiled); ``gdn/value_z`` = v rows then the 6144 ``in_proj_z``
  (z) rows (tiled).
* Per-layer object write order: ``hc_attn`` (dataflow: the mixer turns the
  4-stream residual into the block input) -> attention/GDN -> ``moe`` ->
  ``hc_ffn`` -> PLE (layer 1 only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

# --- model geometry (asserted at preflight) --------------------------------
HIDDEN = 2560
LAYERS = 48
E = 512                 # routed experts
I = 640                 # expert FF width
TOPK = 10
VOCAB = 248320
PLE_ROWS = 320001536
PLE_DIM = 160
PLE_SHARDS = 128
PLE_LAYER = 1           # 0-based (checkpoint ple_layer_ids = [2], 1-based)
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
GDN_HQK = 16            # GDN k heads
GDN_HV = 48             # GDN v heads
GDN_R = GDN_HV // GDN_HQK   # 3
NV_ROWS = E * 2 * I     # 655360
NV_K = HIDDEN
Q6_ROWS = E * HIDDEN    # 1310720
Q6_K = I
QKQV_ROWS = GATE_DIM + KV_DIM + GATE_DIM + KV_DIM  # 13312

LM = "model.language_model.layers."
PLE_P = LM + f"{PLE_LAYER}.ple."
PLE_SHARD_P = PLE_P + "ple_embedding.ngram_embedding.shard_"   # + {i}.weight

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
G_HF: "_HFSource"
G_IMAT: GGUFReader
IMAT_IDS: dict[str, int]

# q|gate de-interleave row indexes on the q_proj output axis (24 heads)
_Q_IDX = np.repeat(np.arange(HEADS) * 2 * HEAD_DIM, HEAD_DIM)          # q rows
_G_IDX = _Q_IDX + HEAD_DIM                                            # gate rows

# GDN V-head tiling permutations (grouped -> tiled; see module docstring)
_V_PERM: np.ndarray = (
    np.arange(GDN_HQK * GDN_R * 128).reshape(GDN_HQK, GDN_R, 128)
    .transpose(1, 0, 2).reshape(-1)
)                                              # 6144 entries (v rows/channels)
_AP48: np.ndarray = (
    np.arange(GDN_HQK * GDN_R).reshape(GDN_HQK, GDN_R).transpose(1, 0).reshape(-1)
)                                              # 48 entries (a/b/A_log/dt rows)


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def _bf16_f32(raw_u16: np.ndarray) -> np.ndarray:
    """Bit-exact BF16 -> FP32 (bf16 word << 16 as f32 bits)."""
    return (raw_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _bf16_bytes(blk_u16: np.ndarray) -> bytes:
    return layouts.encode_direct(torch.from_numpy(blk_u16.copy()).view(torch.bfloat16), BF16)


def _gdn_conv_f32(layer: int) -> bytes:
    """GDN conv kernel in artifact tap-major [4, C] order.

    HF stores [C, 1, 4] (channel-major, 4 taps per channel); the V channel
    block (channels QK_DIM..) is V-head-tiled. The artifact stores the
    transposed [K, C] form (tap-major), matching the GGUF converter's output.
    """
    c = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.linear_attn.conv1d.weight").reshape(HC_DIM, 4))
    c = np.concatenate([c[:QK_DIM], c[QK_DIM:][_V_PERM]], axis=0)   # [C, 4] channel-major
    return _f32_bytes(c.T)                                         # -> [4, C] tap-major


def _gdn_ab_bf16(layer: int) -> bytes:
    """GDN a|b projection [2*GDN_HV, HIDDEN], both V-head-tiled row order."""
    a = G_HF.rows_u16(f"{LM}{layer}.linear_attn.in_proj_a.weight")[_AP48]
    b = G_HF.rows_u16(f"{LM}{layer}.linear_attn.in_proj_b.weight")[_AP48]
    return _bf16_bytes(np.concatenate([a, b], axis=0))


def _gdn_a_log_f32(layer: int) -> bytes:
    """GDN a_log = -exp(reordered(A_log)) via torch FP32 exp (sign-flipped decay)."""
    al = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.linear_attn.A_log"))[_AP48]
    return _f32_bytes((-torch.exp(torch.from_numpy(al))).numpy())


def _gdn_dt_f32(layer: int) -> bytes:
    """GDN dt_bias in reordered (V-tiled) row order."""
    return _f32_bytes(_bf16_f32(G_HF.rows_u16(f"{LM}{layer}.linear_attn.dt_bias"))[_AP48])


def _f32_bytes(blk_f32: np.ndarray) -> bytes:
    return layouts.encode_direct(torch.from_numpy(np.ascontiguousarray(blk_f32, dtype=np.float32)), FP32)


def _imat(name: str) -> np.ndarray:
    return G_IMAT.get_tensor(IMAT_IDS[name]).data


def _check_shape(where: str, shape: tuple, want: tuple) -> None:
    if tuple(shape) != tuple(want):
        raise ValueError(f"unexpected shape for {where}: {tuple(shape)}, want {tuple(want)}")


class _HFSource:
    """Position-independent safetensors reader (pread; fork-safe).

    The index json is read once in the parent; shard headers and file
    descriptors are cached lazily per process.  Workers fork-inherit the
    parent's caches and open their own descriptors (os.pread never shares a
    file offset).
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.wm: dict[str, str] = idx["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}
        self._fds: dict[str, int] = {}

    def _header(self, shard: str) -> tuple[int, dict]:
        h = self._headers.get(shard)
        if h is None:
            with open(self.model_dir / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = (8 + n, json.loads(f.read(n)))
            self._headers[shard] = h
        return h

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            fd = os.open(str(self.model_dir / shard), os.O_RDONLY)
            self._fds[shard] = fd
        return fd

    def _entry(self, name: str) -> tuple[str, int, dict]:
        shard = self.wm[name]
        base, h = self._header(shard)
        return shard, base + h[name]["data_offsets"][0], h[name]

    def shape(self, name: str) -> tuple[tuple[int, ...], str]:
        """Metadata-only (shape, dtype)."""
        return tuple(self._entry(name)[2]["shape"]), self._entry(name)[2]["dtype"]

    def rows_u16(self, name: str, r0: int = 0, r1: int | None = None) -> np.ndarray:
        """Contiguous row range of a BF16 tensor, any rank (rows of dim 0)."""
        shard, off, ent = self._entry(name)
        s = ent["shape"]
        if ent["dtype"] != "BF16":
            raise ValueError(f"{name} is {ent['dtype']}, expected BF16")
        k = 1
        for x in s[1:]:
            k *= x
        r1 = s[0] if r1 is None else r1
        buf = os.pread(self._fd(shard), (r1 - r0) * k * 2, off + r0 * k * 2)
        return np.frombuffer(buf, np.uint16).reshape(r1 - r0, *s[1:])

    def tensor_i64(self, name: str) -> np.ndarray:
        shard, off, ent = self._entry(name)
        if ent["dtype"] != "I64":
            raise ValueError(f"{name} is {ent['dtype']}, expected I64")
        numel = 1
        for x in ent["shape"]:
            numel *= x
        buf = os.pread(self._fd(shard), numel * 8, off)
        return np.frombuffer(buf, "<i8").copy()


# ---------------------------------------------------------------------------
# worker tasks (module level so they fork-inherit the readers)
# ---------------------------------------------------------------------------

def _task_amax(args: tuple[int, int]) -> float:
    layer, e = args
    blk = G_HF.rows_u16(f"{LM}{layer}.mlp.experts.gate_up_proj", e, e + 1).reshape(2 * I, HIDDEN)
    return float(np.abs(_bf16_f32(blk)).max())


def _task_gate_up(args: tuple[int, int, np.float32]) -> tuple[bytes, bytes]:
    """One expert's fused gate|up NVFP4 encode (rows gate-first)."""
    layer, e, dw = args
    blk = G_HF.rows_u16(f"{LM}{layer}.mlp.experts.gate_up_proj", e, e + 1).reshape(2 * I, HIDDEN)
    g = _bf16_f32(blk[:I])
    u = _bf16_f32(blk[I:])
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
    d = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.mlp.experts.down_proj", e, e + 1).reshape(HIDDEN, I))
    band = _imat(f"blk.{layer}.ffn_down_exps.weight.in_sum2")[e]
    codes, scales = enc.encode_row_split(d, band, gs=64, qmax=31)
    return codes.tobytes(), scales.tobytes()


def _task_w8_part(args: tuple[int, str, int, int, int, str]) -> tuple[bytes, bytes, int]:
    """A W8G32 part: HF source rows (mode) + 1-D imatrix band.

    modes: 0 full rows | 1 row slice [r0:r1) | 2/3 full q_proj de-interleaved
    q/gate rows | 4 full rows V-permuted | 5 row slice [r0:r1) V-permuted
    | 6 full rows with V-permuted columns.
    """
    layer, hf_name, mode, r0, r1, band_name = args
    if mode in (1, 5):
        u16 = G_HF.rows_u16(hf_name, r0, r1)
    elif mode in (0, 2, 3, 4, 6):
        u16 = G_HF.rows_u16(hf_name)
    else:
        raise ValueError(f"unknown row mode {mode}")
    if mode == 2:
        blk = np.ascontiguousarray(u16[_Q_IDX])
    elif mode == 3:
        blk = np.ascontiguousarray(u16[_G_IDX])
    elif mode == 4:
        blk = np.ascontiguousarray(u16[_V_PERM])
    elif mode == 5:
        blk = np.ascontiguousarray(u16[_V_PERM])
    elif mode == 6:
        blk = np.ascontiguousarray(u16[:, _V_PERM])
    else:
        blk = np.ascontiguousarray(u16)
    w = _bf16_f32(blk)
    band = _imat(f"blk.{layer}.{band_name}.in_sum2")
    _check_shape(f"band {band_name} layer {layer}", band.shape, (w.shape[1],))
    codes, scales = enc.encode_row_split(w, band, gs=32, qmax=127)
    return codes.tobytes(), scales.tobytes(), int(w.shape[0])


# ---------------------------------------------------------------------------
# payload producers (parent process)
# ---------------------------------------------------------------------------

def _i32_u64_pairs(i64: np.ndarray, count: int) -> bytes:
    _check_shape("ple u64 metadata", i64.shape, (count,))
    v = i64.astype(np.uint64)
    lo = (v & np.uint64(0xFFFFFFFF)).astype(np.int32)
    hi = (v >> np.uint64(32)).astype(np.int32)
    out = np.empty(2 * count, np.int32)
    out[0::2] = lo
    out[1::2] = hi
    return layouts.encode_direct(torch.from_numpy(out), I32)


def _ple_table_chunks(n_rows: int):
    """Stream the PLE table: 128 shards in ascending index order."""
    src = G_HF
    shard_names = [f"{PLE_SHARD_P}{i}.weight" for i in range(PLE_SHARDS)]
    rows_per: list[int] = []
    for nm in shard_names:
        s, dt = src.shape(nm)
        _check_shape(nm, s, (PLE_ROWS // PLE_SHARDS, PLE_DIM) if False else s)
        if dt != "BF16" or len(s) != 2 or s[1] != PLE_DIM:
            raise ValueError(f"bad PLE shard {nm}: {s} {dt}")
        rows_per.append(s[0])
    if sum(rows_per) != PLE_ROWS:
        raise ValueError(f"PLE shard rows {sum(rows_per)} != {PLE_ROWS}")
    emitted = 0
    for nm, nrows in zip(shard_names, rows_per):
        s0 = 0
        while s0 < nrows and emitted < n_rows:
            c = min(PLE_CHUNK, nrows - s0, n_rows - emitted)
            yield src.rows_u16(nm, s0, s0 + c).tobytes()
            s0 += c
            emitted += c


# ---------------------------------------------------------------------------
# tensor existence preflight
# ---------------------------------------------------------------------------

def _layer_tensor_names(layer: int) -> tuple[str, ...]:
    base = (
        f"{LM}{layer}.linear_attn.in_proj_qkv.weight",
        f"{LM}{layer}.linear_attn.in_proj_z.weight",
        f"{LM}{layer}.linear_attn.out_proj.weight",
        f"{LM}{layer}.linear_attn.in_proj_a.weight",
        f"{LM}{layer}.linear_attn.in_proj_b.weight",
        f"{LM}{layer}.linear_attn.A_log",
        f"{LM}{layer}.linear_attn.dt_bias",
        f"{LM}{layer}.linear_attn.conv1d.weight",
        f"{LM}{layer}.linear_attn.norm.weight",
    ) if layer % 4 != 3 else (
        f"{LM}{layer}.self_attn.q_proj.weight",
        f"{LM}{layer}.self_attn.k_proj.weight",
        f"{LM}{layer}.self_attn.v_proj.weight",
        f"{LM}{layer}.self_attn.o_proj.weight",
        f"{LM}{layer}.self_attn.q_norm.weight",
        f"{LM}{layer}.self_attn.k_norm.weight",
        f"{LM}{layer}.self_attn.indexer.index_qk_proj.weight",
        f"{LM}{layer}.self_attn.indexer.q_layernorm.weight",
        f"{LM}{layer}.self_attn.indexer.k_layernorm.weight",
    )
    moe = (
        f"{LM}{layer}.mlp.experts.gate_up_proj",
        f"{LM}{layer}.mlp.experts.down_proj",
        f"{LM}{layer}.mlp.gate.weight",
        f"{LM}{layer}.mlp.shared_expert_gate.weight",
        f"{LM}{layer}.mlp.shared_expert.gate_proj.weight",
        f"{LM}{layer}.mlp.shared_expert.up_proj.weight",
        f"{LM}{layer}.mlp.shared_expert.down_proj.weight",
    )
    hc = tuple(
        f"{LM}{layer}.{sec}_hyper_connection.{t}"
        for sec in ("attn", "mlp")
        for t in ("input_mix_weight_up.weight", "input_mix_weight_down.weight",
                  "block_inject_weight.weight", "hc_norm.weight")
    )
    return base + moe + hc


def _layer_band_names(layer: int) -> tuple[str, ...]:
    base = ("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight") \
        if layer % 4 != 3 else (
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")
    base = base + ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")
    return base + ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")


def _preflight(layers: tuple[int, ...], resources_dir: Path) -> dict[str, object]:
    # checkpoint config (multimodal: text fields live under text_config, with a
    # top-level fallback for a language-only checkpoint)
    raw = json.loads((resources_dir / "config.json").read_text())
    cfg = raw["text_config"] if isinstance(raw.get("text_config"), dict) else raw
    checks: dict[str, object] = {
        "config.num_hidden_layers": cfg.get("num_hidden_layers"),
        "config.hidden_size": cfg.get("hidden_size"),
        "config.ple_layer_ids": cfg.get("ple_layer_ids"),
        "config.ngram_size": cfg.get("ngram_size"),
        "config.heads_per_ngram": cfg.get("heads_per_ngram"),
        "config.split_ngram_parts": cfg.get("split_ngram_parts"),
        "config.hc_count": cfg.get("hc_count"),
        "config.hc_lowrank": cfg.get("hc_lowrank"),
        "config.full_attention_interval": cfg.get("full_attention_interval"),
    }
    want = {
        "config.num_hidden_layers": LAYERS,
        "config.hidden_size": HIDDEN,
        "config.ple_layer_ids": [PLE_LAYER + 1],
        "config.ngram_size": 3,
        "config.heads_per_ngram": 8,
        "config.split_ngram_parts": PLE_SHARDS,
        "config.hc_count": HC_DIM // HIDDEN,
        "config.hc_lowrank": HC_LR,
        "config.full_attention_interval": 4,
    }
    for k, w in want.items():
        if checks[k] != w:
            raise ValueError(f"config mismatch: {k} = {checks[k]!r}, want {w!r}")

    # HF tensor + shape existence for every planned layer
    for layer in layers:
        for base in _layer_tensor_names(layer):
            if base not in G_HF.wm:
                raise ValueError(f"missing checkpoint tensor {base}")
        for base in _layer_band_names(layer):
            if f"blk.{layer}.{base}.in_sum2" not in IMAT_IDS:
                raise ValueError(f"missing imatrix band blk.{layer}.{base}.in_sum2")
    for name, shape in (
        (f"{LM}0.mlp.experts.gate_up_proj", (E, 2 * I, HIDDEN)),
        (f"{LM}0.mlp.experts.down_proj", (E, HIDDEN, I)),
        (f"{LM}3.self_attn.q_proj.weight", (ATTN_Q_DIM, HIDDEN)),
        (f"{LM}0.linear_attn.in_proj_qkv.weight", (HC_DIM, HIDDEN)),
        (f"{LM}0.linear_attn.in_proj_z.weight", (GATE_DIM, HIDDEN)),
        (f"{LM}0.linear_attn.out_proj.weight", (HIDDEN, V_DIM)),
        (f"{LM}0.linear_attn.A_log", (GDN_HV,)),
        (f"{PLE_SHARD_P}0.weight", (PLE_ROWS // PLE_SHARDS, PLE_DIM)),
        (f"{PLE_P}ple_embedding.layer_multipliers", (3,)),
        (f"{PLE_P}ple_embedding.ngram_heads_offsets", (16,)),
        (f"{PLE_P}ple_embedding.ngram_heads_vocab_sizes", (16,)),
        ("model.language_model.embed_tokens.weight", (VOCAB, HIDDEN)),
        ("lm_head.weight", (VOCAB, HIDDEN)),
    ):
        if name not in G_HF.wm:
            raise ValueError(f"missing checkpoint tensor {name}")
        s, dt = G_HF.shape(name)
        if s != shape:
            raise ValueError(f"checkpoint shape {name}: {s}, want {shape}")
    for i in range(PLE_SHARDS):
        nm = f"{PLE_SHARD_P}{i}.weight"
        if nm not in G_HF.wm:
            raise ValueError(f"missing PLE shard {nm}")
    for nm, dt in ((f"{PLE_P}ple_embedding.layer_multipliers", "I64"),
                   (f"{PLE_P}ple_embedding.ngram_heads_offsets", "I64"),
                   (f"{PLE_P}ple_embedding.ngram_heads_vocab_sizes", "I64")):
        if G_HF.shape(nm)[1] != dt:
            raise ValueError(f"{nm} dtype {G_HF.shape(nm)[1]}, want I64")
    for name in (
        f"{PLE_P}key_proj.weight", f"{PLE_P}value_proj.weight", f"{PLE_P}conv1d.weight",
        f"{PLE_P}norm_query.weight", f"{PLE_P}norm_key.weight", f"{PLE_P}norm_conv.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
    ):
        if name not in G_HF.wm:
            raise ValueError(f"missing checkpoint tensor {name}")
    return checks


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

    ensure() runs the 512-task amax pre-pass, computes the layer d_w and
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
        amax_tasks = [(layer, e) for e in range(E)]
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
            qkv = f"{LM}{layer}.linear_attn.in_proj_qkv.weight"
            w8_tasks = [
                (layer, qkv, 1, 0, QK_DIM, "attn_qkv.weight"),
                (layer, qkv, 5, QK_DIM, HC_DIM, "attn_qkv.weight"),
                (layer, f"{LM}{layer}.linear_attn.in_proj_z.weight", 4, 0, 0, "attn_gate.weight"),
                (layer, f"{LM}{layer}.linear_attn.out_proj.weight", 6, 0, 0, "ssm_out.weight"),
            ]
        else:
            q = f"{LM}{layer}.self_attn.q_proj.weight"
            w8_tasks = [
                (layer, q, 2, 0, 0, "attn_q.weight"),
                (layer, f"{LM}{layer}.self_attn.k_proj.weight", 0, 0, 0, "attn_k.weight"),
                (layer, q, 3, 0, 0, "attn_q.weight"),
                (layer, f"{LM}{layer}.self_attn.v_proj.weight", 0, 0, 0, "attn_v.weight"),
                (layer, f"{LM}{layer}.self_attn.o_proj.weight", 0, 0, 0, "attn_output.weight"),
            ]
        m = f"{LM}{layer}.mlp"
        w8_tasks += [
            (layer, f"{m}.shared_expert.gate_proj.weight", 0, 0, 0, "ffn_gate_shexp.weight"),
            (layer, f"{m}.shared_expert.up_proj.weight", 0, 0, 0, "ffn_up_shexp.weight"),
            (layer, f"{m}.shared_expert.down_proj.weight", 0, 0, 0, "ffn_down_shexp.weight"),
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
    blk = G_HF.rows_u16(f"{LM}{layer}.mlp.experts.gate_up_proj", 0, 1).reshape(2 * I, HIDDEN)
    g = _bf16_f32(blk[:I])
    u = _bf16_f32(blk[I:])
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
    d = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.mlp.experts.down_proj", 0, 1).reshape(HIDDEN, I))
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
    r = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.mlp.gate.weight"))
    s = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.mlp.shared_expert_gate.weight"))
    _check_shape(f"router layer {layer}", r.shape, (E, HIDDEN))
    _check_shape(f"router shared layer {layer}", s.shape, (1, HIDDEN))
    return layouts.encode_direct(
        torch.from_numpy(np.ascontiguousarray(np.concatenate([r, s], axis=0), dtype=np.float32)), FP32)


def _hc_units(layer: int, section: str) -> list[tuple[TensorSpec, object]]:
    L = f"text/layers/{layer}"
    # the FFN mixer's checkpoint prefix is "mlp", not "ffn"
    P = f"{LM}{layer}.{'attn' if section == 'attn' else 'mlp'}_hyper_connection"
    units: list[tuple[TensorSpec, object]] = []
    for name, shape, hf in (
        ("up", (HC_DIM, HC_LR), "input_mix_weight_up.weight"),
        ("down", (HC_LR, HC_DIM), "input_mix_weight_down.weight"),
        ("inject", (HC_DIM // HIDDEN, HC_DIM), "block_inject_weight.weight"),
    ):
        units.append((_spec(f"{L}/hc_{section}/{name}", shape, BF16),
                      lambda p=f"{P}.{hf}", sh=shape: _bf16_bytes(G_HF.rows_u16(p))))
    units.append((_spec(f"{L}/hc_{section}/norm", (HC_DIM,), FP32),
                  lambda p=f"{P}.hc_norm.weight":
                  _f32_bytes(np.float32(1.0) + _bf16_f32(G_HF.rows_u16(p)))))
    return units


def _layer_units(layer: int, st: _LayerState) -> list[tuple[TensorSpec, object]]:
    L = f"text/layers/{layer}"
    units = _hc_units(layer, "attn")
    if layer % 4 != 3:
        ga = f"{LM}{layer}.linear_attn"
        units += [
            (_spec(f"{L}/gdn/query_key", (QK_DIM, HIDDEN), W8),
             lambda s=st: _w8_payload((QK_DIM, HIDDEN), s.w8(1))),
            (_spec(f"{L}/gdn/value_z", (V_DIM + GATE_DIM, HIDDEN), W8),
             lambda s=st: _w8_payload((V_DIM + GATE_DIM, HIDDEN), s.w8(2))),
            (_spec(f"{L}/gdn/a_b_projection", (2 * GDN_HV, HIDDEN), BF16),
             lambda l=layer: _gdn_ab_bf16(l)),
            (_spec(f"{L}/gdn/a_log", (GDN_HV,), FP32),
             lambda l=layer: _gdn_a_log_f32(l)),
            (_spec(f"{L}/gdn/dt_bias", (GDN_HV,), FP32),
             lambda l=layer: _gdn_dt_f32(l)),
            (_spec(f"{L}/gdn/convolution", (4, HC_DIM), FP32),
             lambda l=layer: _gdn_conv_f32(l)),
            (_spec(f"{L}/gdn/norm", (128,), FP32),
             lambda l=layer: _f32_bytes(_bf16_f32(G_HF.rows_u16(f"{LM}{l}.linear_attn.norm.weight")))),
            (_spec(f"{L}/gdn/output", (HIDDEN, V_DIM), W8),
             lambda s=st: _w8_payload((HIDDEN, V_DIM), s.w8(1))),
        ]
    else:
        sa = f"{LM}{layer}.self_attn"
        units += [
            (_spec(f"{L}/attention/query_key_gate_value", (QKQV_ROWS, HIDDEN), W8),
             lambda s=st: _w8_payload((QKQV_ROWS, HIDDEN), s.w8(4))),
            (_spec(f"{L}/attention/query_norm", (HEAD_DIM,), FP32),
             lambda l=layer: _f32_bytes(np.float32(1.0) + _bf16_f32(G_HF.rows_u16(f"{LM}{l}.self_attn.q_norm.weight")))),
            (_spec(f"{L}/attention/key_norm", (HEAD_DIM,), FP32),
             lambda l=layer: _f32_bytes(np.float32(1.0) + _bf16_f32(G_HF.rows_u16(f"{LM}{l}.self_attn.k_norm.weight")))),
            (_spec(f"{L}/attention/output", (HIDDEN, V_DIM), W8),
             lambda s=st: _w8_payload((HIDDEN, V_DIM), s.w8(1))),
            (_spec(f"{L}/indexer/query_proj", (KV_DIM, HIDDEN), BF16),
             lambda l=layer: _bf16_bytes(G_HF.rows_u16(f"{LM}{l}.self_attn.indexer.index_qk_proj.weight", 0, KV_DIM))),
            (_spec(f"{L}/indexer/key_proj", (IDX_K_DIM, HIDDEN), BF16),
             lambda l=layer: _bf16_bytes(G_HF.rows_u16(f"{LM}{l}.self_attn.indexer.index_qk_proj.weight", KV_DIM, KV_DIM + IDX_K_DIM))),
            (_spec(f"{L}/indexer/query_norm", (IDX_K_DIM,), FP32),
             lambda l=layer: _f32_bytes(np.float32(1.0) + _bf16_f32(G_HF.rows_u16(f"{LM}{l}.self_attn.indexer.q_layernorm.weight")))),
            (_spec(f"{L}/indexer/key_norm", (IDX_K_DIM,), FP32),
             lambda l=layer: _f32_bytes(np.float32(1.0) + _bf16_f32(G_HF.rows_u16(f"{LM}{l}.self_attn.indexer.k_layernorm.weight")))),
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
    if layer == PLE_LAYER:
        units += [
            (_spec("text/ple/key", (HC_DIM, HIDDEN), BF16),
             lambda: _bf16_bytes(G_HF.rows_u16(f"{PLE_P}key_proj.weight"))),
            (_spec("text/ple/value", (HIDDEN, HIDDEN), BF16),
             lambda: _bf16_bytes(G_HF.rows_u16(f"{PLE_P}value_proj.weight"))),
            (_spec("text/ple/convolution", (4, HC_DIM), BF16),
             lambda: _bf16_bytes(G_HF.rows_u16(f"{PLE_P}conv1d.weight").reshape(HC_DIM, 4).T.copy())),
            (_spec("text/ple/norm_query", (HC_DIM,), FP32),
             lambda: _f32_bytes(np.float32(1.0) + _bf16_f32(G_HF.rows_u16(f"{PLE_P}norm_query.weight")))),
            (_spec("text/ple/norm_key", (HC_DIM,), FP32),
             lambda: _f32_bytes(np.float32(1.0) + _bf16_f32(G_HF.rows_u16(f"{PLE_P}norm_key.weight")))),
            (_spec("text/ple/norm_conv", (HC_DIM,), FP32),
             lambda: _f32_bytes(np.float32(1.0) + _bf16_f32(G_HF.rows_u16(f"{PLE_P}norm_conv.weight")))),
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
        lambda: _bf16_bytes(G_HF.rows_u16("model.language_model.embed_tokens.weight")))
    add(_spec("text/ple/layer_multipliers", (6,), I32),
        lambda: _i32_u64_pairs(G_HF.tensor_i64(f"{PLE_P}ple_embedding.layer_multipliers"), 3))
    add(_spec("text/ple/head_offsets", (32,), I32),
        lambda: _i32_u64_pairs(G_HF.tensor_i64(f"{PLE_P}ple_embedding.ngram_heads_offsets"), 16))
    add(_spec("text/ple/head_vocab_sizes", (32,), I32),
        lambda: _i32_u64_pairs(G_HF.tensor_i64(f"{PLE_P}ple_embedding.ngram_heads_vocab_sizes"), 16))
    if not smoke:
        add(_spec("text/per_layer_token_embedding", (PLE_ROWS, PLE_DIM), BF16),
            lambda: _ple_table_chunks(PLE_ROWS))
    for layer in layers:
        for spec, producer in _layer_units(layer, LAYER_STATES[layer]):
            add(spec, producer)
    mx = "model.language_model.hyper_connection_mixer"
    add(_spec("text/output_hc/up", (HC_DIM, HC_LR), BF16),
        lambda: _bf16_bytes(G_HF.rows_u16(f"{mx}.input_mix_weight_up.weight")))
    add(_spec("text/output_hc/down", (HC_LR, HC_DIM), BF16),
        lambda: _bf16_bytes(G_HF.rows_u16(f"{mx}.input_mix_weight_down.weight")))
    add(_spec("text/output_hc/norm", (HC_DIM,), FP32),
        lambda: _f32_bytes(np.float32(1.0) + _bf16_f32(G_HF.rows_u16(f"{mx}.hc_norm.weight"))))
    add(_spec("text/output_head", (VOCAB, HIDDEN), BF16),
        lambda: _bf16_bytes(G_HF.rows_u16("lm_head.weight")))
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
    global G_HF, G_IMAT, IMAT_IDS, LAYER_STATES
    t0 = time.time()
    print(f"[jbnvfp4] opening {args.model}", flush=True)
    G_HF = _HFSource(Path(args.model))
    G_IMAT = GGUFReader(args.imatrix)
    IMAT_IDS = {t.name: i for i, t in enumerate(G_IMAT.tensors)}
    layers = _parse_layers(args.layers)
    if args.smoke and not args.layers:
        layers = (0, 1, 3)
    print(f"[jbnvfp4] layers={layers} workers={args.workers} smoke={args.smoke}", flush=True)
    checks = _preflight(layers, Path(args.resources))
    print(f"[jbnvfp4] preflight OK ({len(G_HF.wm)} checkpoint tensors, "
          f"{len(IMAT_IDS)} imatrix tensors)", flush=True)
    for k, v in checks.items():
        print(f"[jbnvfp4]   {k} = {v}", flush=True)

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
            "source_format": "HF safetensors (weights + PLE table + PLE u64 metadata) + imatrix GGUF (bands)",
            "config_checks": {k: str(v) for k, v in checks.items()},
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
                "hf_layout": "source (out,in); artifact (out,in)",
                "gdn_v_tiling": "V heads grouped->tiled (r=3) on qkv v-rows, z, a/b, A_log/dt, conv v-channels, out cols",
                "gdn_a_log": "-exp(reordered(A_log)) via torch FP32 exp (linear decay rate, sign-flipped)",
                "norms_folded": "f32(1.0) + f32(bf16(w)) for HC/mixer/q/k/indexer/PLE norms; GDN 128-dim norm not folded",
                "experts": "[E,1280,2560] per-expert [gate 640 | up 640], K=2560, NVFP4, d_w=f32(2688/amax)",
                "routed_down": "e-major, Q6G64_F16S (int6, group-64)",
                "input_divisor": "f32(2688 / max_j sqrt(band[j])) over gate+up bands",
                "w8": "W8G32_F16S (int8, group-32) with 1-D per-source imatrix bands",
                "ple_u64": "checkpoint I64 tensors, I32 lo/hi pairs, lo first (no I64 in the artifact)",
                "ple_table": "BF16 [320001536,160] = shards 0..127 concatenated in index order",
                "full_attn_deinterleave": "head h: q=rows h*512..h*512+255, gate=h*512+256..(h+1)*512-1",
                "hc_head": "no inject tensor at the head (graph passes nullptr)",
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
                    "(qwen4exp) from the HF safetensors checkpoint + unsloth imatrix.")
    ap.add_argument("--model", required=True, help="HF safetensors checkpoint dir (131 shards)")
    ap.add_argument("--imatrix", required=True, help="unsloth imatrix GGUF")
    ap.add_argument("--resources", required=True,
                    help="checkpoint dir with the six frontend resource files")
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
