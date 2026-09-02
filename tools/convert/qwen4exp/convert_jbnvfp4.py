"""jbq4 ``.ninfer`` artifact builder for Qwen3.8-Flash-Next (qwen4exp).

All-MoE Q4/Q8 (D14: text layers {0,1,47} + MTP = Q8G64_F16S, the other 45
layers = Q4G64_F16S), MTP enabled, all-tensor preservation.  See
``q4plan.md`` (the active authority for this artifact) for the object
census, sizes, and verification gates.

Source of truth
---------------
* HF safetensors checkpoint directory (``Qwen3.8-Flash-Next``, 131 shards,
  ~351 GB); the canonical weight source (text, MTP, and vision tensors);
* unsloth imatrix file (``imatrix_unsloth_qwen38next.gguf_file``) — per-tensor
  input second-moment bands (``in_sum2``) for the TEXT layers only; the MTP
  module and all vision tensors take the no-band path (D10);
* HF frontend resources (six ``frontend/*`` files).

Numerics
--------
The encoder is the shared pure-numpy imatrix-aware selection from
``tools/convert/common/encoder_imatrix.py``: the per-object objective is the
importance-weighted MSE of the quantization error (plan §2.4 — "the MSE of
importance × error").  No llama-quantize / C++ on this path.

Codec map (D14):
* text MoE (routed + shared): Q8G64_F16S for layers {0, 1, 47}, Q4G64_F16S
  otherwise; both group-64, asymmetric code ranges (-128..127 / -8..7);
  imatrix bands for all 48 text layers.
* MTP MoE: Q8G64_F16S, no-band; MTP dense: W8G32_F16S, no-band; MTP 1-D
  norms: BF16 passthrough.
* Vision: the 27B quantized map — patch Q6G64, block qkv/fc1 Q4G64,
  block output/fc2 Q5G64, merger W8G32; all no-band; 1-D vectors BF16.

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
        --model /llm/models/Qwen3.8-Flash-Next \
        --imatrix /llm/models/imatrix_unsloth_qwen38next.gguf_file \
        --resources /llm/models/Qwen3.8-Flash-Next \
        --out /home/robot/llm/Qwen3.8-Flash-Next-JB-Q4.ninfer \
        --workers 32

Layout conventions
------------------
------------------
* HF linears are ``(out, in)`` and MoE experts ``[E, out*2|out, in]``;
  artifact objects are stored ``(out, in)``.
* ``moe/routed_gate_up``: e-major parent, per-expert row block
  ``[gate 640 | up 640]``, K = 2560 — Q8G64_F16S (layers 0, 1, 47) or
  Q4G64_F16S (the other 45) (row-split-k128-v1, group 64, FP16 scale).
* ``moe/routed_down``: e-major rows, ``[out 2560, in 640]`` — same codec as
  the layer's routed_gate_up.
* ``moe/shared_gate_up`` / ``moe/shared_down``: same codec, with the 1-D
  shared-expert imatrix bands (text layers).
* ``moe/router_gate``: ``[routed 512 | shared-gate 1]`` rows, FP32.
* MTP (``mtp/*``, 28 objects): MoE = Q8G64_F16S no-band; dense = W8G32_F16S
  no-band (fused ``query_key_gate_value`` de-interleaved like the text
  full-attention layer); 1-D norms = BF16 passthrough.
* Vision (``vision/*``, 333 objects): quantized per the 27B map (Q6 patch,
  Q4 qkv/fc1, Q5 output/fc2, W8 merger, no-band); 1-D vectors BF16
  passthrough.
* GDN layers (l % 4 != 3): ``in_proj_qkv`` splits into q|k (4096 rows) and v
  (6144 rows, tiled); ``gdn/value_z`` = v rows then the 6144 ``in_proj_z``
  (z) rows (tiled).
* Per-layer object write order: ``hc_attn`` (dataflow: the mixer turns the
  4-stream residual into the block input) -> attention/GDN -> ``moe`` ->
  ``hc_ffn`` -> PLE (layer 1 only).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
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
WEIGHTS_ID = "jbq4"

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

# --- MTP / vision geometry (asserted at preflight) -------------------------
MTP_P = "mtp.layers.0."     # single MTP layer (mtp_num_hidden_layers = 1)
MTP_HF = "mtp."
VISION_BLOCKS = 27
VISION_HIDDEN = 1152

LM = "model.language_model.layers."
PLE_P = LM + f"{PLE_LAYER}.ple."
PLE_SHARD_P = PLE_P + "ple_embedding.ngram_embedding.shard_"   # + {i}.weight

# --- artifact format/layout names ------------------------------------------
CONTIG = "contiguous-le-v1"
ROWSPLIT = "row-split-k128-v1"
BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
W8 = "W8G32_F16S"
Q4 = "Q4G64_F16S"
ROWSPLIT_FORMATS = (W8, Q4)

PLE_TABLE_NAME = "text/per_layer_token_embedding"
PLE_COPY_BYTES = 64 * 2**20   # direct-copy thread buffer size


def _spec(name: str, shape: tuple[int, ...], fmt: str) -> TensorSpec:
    """TensorSpec with the canonical layout for its numeric format."""
    if fmt in (BF16, FP32, I32):
        layout = CONTIG
    elif fmt in ROWSPLIT_FORMATS + ("Q5G64_F16S", "Q6G64_F16S", "Q8G64_F16S"):
        layout = ROWSPLIT
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

# MoE codec dispatch (D14): text layers {0, 1, 47} and the MTP module are
# Q8G64_F16S (asymmetric [-128, 127]); the other 45 text layers are
# Q4G64_F16S (asymmetric [-8, 7]).  Text layers carry imatrix bands; the
# MTP module does not (D10).

Q5 = "Q5G64_F16S"
Q6 = "Q6G64_F16S"
Q8 = "Q8G64_F16S"
Q8_LAYERS = frozenset((0, 1, 47))
GROUPS = {Q4: 64, Q5: 64, Q6: 64, Q8: 64, W8: 32}
QMAX = {Q4: 7, Q5: 15, Q6: 31, Q8: 127, W8: 127}
QMIN = {Q4: -8, Q5: -16, Q6: -32, Q8: -128, W8: -127}


def _moe_code(src: str) -> str:
    """Layer MoE codec: src is the HF prefix (LM layer or MTP_P)."""
    if src.startswith(MTP_P) or int(src[len(LM):-1]) in Q8_LAYERS:
        return Q8
    return Q4


def _moe_bands(src: str, e: int | None, role: str) -> np.ndarray | None:
    """Per-expert MoE band (text layers only) for role gate / up / down."""
    if not src.startswith(LM) or e is None:
        return None
    l = int(src[len(LM):-1])
    if role == "gate":
        return _imat(f"blk.{l}.ffn_gate_exps.weight.in_sum2")[e]
    if role == "up":
        return _imat(f"blk.{l}.ffn_up_exps.weight.in_sum2")[e]
    if role == "down":
        return _imat(f"blk.{l}.ffn_down_exps.weight.in_sum2")[e]
    raise ValueError(f"unknown band role {role!r}")


def _encode_gu_expert(src: str, e: int) -> tuple[np.ndarray, np.ndarray]:
    """Q4/Q8 encode of one expert's fused gate|up [1280, 2560] (gate-first rows).

    ``src`` is the HF prefix (``model.language_model.layers.{l}.`` for text
    with imatrix bands, or ``mtp.layers.0.`` with the no-band path, D10).
    """
    blk = G_HF.rows_u16(f"{src}mlp.experts.gate_up_proj", e, e + 1).reshape(2 * I, HIDDEN)
    w = _bf16_f32(blk)
    band = None
    bg = _moe_bands(src, e, "gate")
    if bg is not None:
        band = np.empty((2 * I, HIDDEN), np.float32)
        band[:I] = bg
        band[I:] = _moe_bands(src, e, "up")
    code = _moe_code(src)
    return enc.encode_row_split(w, band, gs=GROUPS[code], qmax=QMAX[code], qmin=QMIN[code])


def _encode_dn_expert(src: str, e: int) -> tuple[np.ndarray, np.ndarray]:
    """Q4/Q8 encode of one expert's down [2560, 640] (see _encode_gu_expert)."""
    d = _bf16_f32(G_HF.rows_u16(f"{src}mlp.experts.down_proj", e, e + 1).reshape(HIDDEN, I))
    band = _moe_bands(src, e, "down")
    code = _moe_code(src)
    return enc.encode_row_split(d, band, gs=GROUPS[code], qmax=QMAX[code], qmin=QMIN[code])


def _task_gu(args: tuple[str, int]) -> tuple[bytes, bytes]:
    c, s = _encode_gu_expert(*args)
    return c.tobytes(), s.tobytes()


def _task_dn(args: tuple[str, int]) -> tuple[bytes, bytes]:
    c, s = _encode_dn_expert(*args)
    return c.tobytes(), s.tobytes()


def _task_int_part(args: tuple[str, str, str | None]) -> tuple[bytes, bytes, int]:
    """A Q4/Q8 part over full HF rows + optional 1-D imatrix band (shared
    experts; band name None = no-band path, MTP)."""
    src, hf_name, band_name = args
    w = _bf16_f32(G_HF.rows_u16(hf_name))
    band = _imat(f"blk.{src[len(LM):-1]}.{band_name}.in_sum2") if band_name is not None else None
    if band is not None:
        _check_shape(f"band {band_name} {src}", band.shape, (w.shape[1],))
    code = _moe_code(src)
    codes, scales = enc.encode_row_split(w, band, gs=GROUPS[code], qmax=QMAX[code], qmin=QMIN[code])
    return codes.tobytes(), scales.tobytes(), int(w.shape[0])


def _task_w8_part(args: tuple[str, int, int, int, str | None]) -> tuple[bytes, bytes, int]:
    """A W8G32 part: HF source rows (mode) + optional 1-D imatrix band (None
    = no-band, MTP dense).

    modes: 0 full rows | 1 row slice [r0:r1) | 2/3 full q_proj de-interleaved
    q/gate rows | 4 full rows V-permuted | 5 row slice [r0:r1) V-permuted
    | 6 full rows with V-permuted columns.
    """
    hf_name, mode, r0, r1, band_name = args
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
    elif mode in (4, 5):
        blk = np.ascontiguousarray(u16[_V_PERM])
    elif mode == 6:
        blk = np.ascontiguousarray(u16[:, _V_PERM])
    else:
        blk = np.ascontiguousarray(u16)
    w = _bf16_f32(blk)
    band = _imat(band_name) if band_name is not None else None
    if band is not None:
        _check_shape(f"band {band_name}", band.shape, (w.shape[1],))
    codes, scales = enc.encode_row_split(w, band, gs=32, qmax=127)
    return codes.tobytes(), scales.tobytes(), int(w.shape[0])


def _task_vision_part(args: tuple[str, str]) -> tuple[bytes, bytes, int]:
    """A no-band quantized vision part (Q4/Q5/Q6/W8) over full HF rows.

    The patch-embed conv3d kernel arrives 5-D [1152, 3, 2, 16, 16] and is
    flattened C-contiguous to [1152, 1536] (the 27B artifact layout).
    """
    hf_name, code = args
    u16 = G_HF.rows_u16(hf_name)
    w = _bf16_f32(u16.reshape(u16.shape[0], -1))
    gs = GROUPS[code]
    if w.shape[1] % gs:
        # k not a multiple of the group size (vision mlp/fc2, k=4304): zero-pad
        # the input columns to the layout's k_pad before encoding; the padded
        # columns decode to exactly zero.  The group axis then already matches
        # the row-split-k128 layout (no further pad needed).
        k_pad = layouts.row_split_geometry(code, w.shape).k_pad
        w = np.pad(w, ((0, 0), (0, k_pad - w.shape[1])))
    codes, scales = enc.encode_row_split(w, None, gs=gs, qmax=QMAX[code], qmin=QMIN[code])
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


def _ple_table_direct(out_path: Path, abs_offset: int, file_size: int,
                      done: dict, stop: threading.Event) -> None:
    """Single-thread PLE table copy, concurrent with the main encode loop.

    Preallocates the whole output file (one ftruncate to the final size, so
    the table's region at the front of the file is reserved), then copies the
    BF16 rows directly from the 128 shards via pread/pwrite — no decode, no
    numpy, no writer involvement.  The main writer ``skip()``s the object and
    joins this thread before finish().
    """
    fd = os.open(str(out_path), os.O_WRONLY)
    try:
        os.ftruncate(fd, file_size)
        written = 0
        for i in range(PLE_SHARDS):
            if stop.is_set():
                done["error"] = RuntimeError("stopped before the PLE copy finished")
                return
            nm = f"{PLE_SHARD_P}{i}.weight"
            shard, off, ent = G_HF._entry(nm)
            n = int(np.prod(ent["shape"])) * 2
            src = os.open(str(G_HF.model_dir / shard), os.O_RDONLY)
            try:
                pos = 0
                while pos < n:
                    c = min(PLE_COPY_BYTES, n - pos)
                    buf = os.pread(src, c, off + pos)
                    if len(buf) != c:
                        raise IOError(f"short pread on {shard} for {nm}")
                    if os.pwrite(fd, buf, abs_offset + written + pos) != c:
                        raise IOError(f"short pwrite on {out_path} for {nm}")
                    pos += c
            finally:
                os.close(src)
            written += n
        done["bytes"] = written
    except BaseException as exc:
        done["error"] = exc
    finally:
        os.close(fd)


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


def _mtp_tensor_shapes() -> list[tuple[str, tuple[int, ...]]]:
    """The 31 MTP source tensors (all BF16) with exact shapes."""
    T = MTP_P
    out = [
        (f"{MTP_HF}fc_embedding.weight", (HIDDEN, HIDDEN)),
        (f"{MTP_HF}fc_hidden.weight", (HIDDEN, HIDDEN)),
        (f"{MTP_HF}pre_fc_norm_embedding.weight", (HIDDEN,)),
        (f"{MTP_HF}pre_fc_norm_hidden.weight", (HC_DIM,)),
        (f"{MTP_HF}hyper_connection_mixer.hc_norm.weight", (HC_DIM,)),
        (f"{MTP_HF}hyper_connection_mixer.input_mix_weight_down.weight", (HC_LR, HC_DIM)),
        (f"{MTP_HF}hyper_connection_mixer.input_mix_weight_up.weight", (HC_DIM, HC_LR)),
    ]
    for sec in ("attn", "mlp"):
        P = f"{T}{sec}_hyper_connection."
        out += [
            (P + "input_mix_weight_up.weight", (HC_DIM, HC_LR)),
            (P + "input_mix_weight_down.weight", (HC_LR, HC_DIM)),
            (P + "block_inject_weight.weight", (HC_DIM // HIDDEN, HC_DIM)),
            (P + "hc_norm.weight", (HC_DIM,)),
        ]
    out += [
        (f"{T}self_attn.q_proj.weight", (ATTN_Q_DIM, HIDDEN)),
        (f"{T}self_attn.k_proj.weight", (KV_DIM, HIDDEN)),
        (f"{T}self_attn.v_proj.weight", (KV_DIM, HIDDEN)),
        (f"{T}self_attn.o_proj.weight", (HIDDEN, V_DIM)),
        (f"{T}self_attn.q_norm.weight", (HEAD_DIM,)),
        (f"{T}self_attn.k_norm.weight", (HEAD_DIM,)),
        (f"{T}self_attn.indexer.index_qk_proj.weight", (KV_DIM + IDX_K_DIM, HIDDEN)),
        (f"{T}self_attn.indexer.q_layernorm.weight", (IDX_K_DIM,)),
        (f"{T}self_attn.indexer.k_layernorm.weight", (IDX_K_DIM,)),
        (f"{T}mlp.gate.weight", (E, HIDDEN)),
        (f"{T}mlp.shared_expert_gate.weight", (1, HIDDEN)),
        (f"{T}mlp.experts.gate_up_proj", (E, 2 * I, HIDDEN)),
        (f"{T}mlp.experts.down_proj", (E, HIDDEN, I)),
        (f"{T}mlp.shared_expert.gate_proj.weight", (I, HIDDEN)),
        (f"{T}mlp.shared_expert.up_proj.weight", (I, HIDDEN)),
        (f"{T}mlp.shared_expert.down_proj.weight", (HIDDEN, I)),
    ]
    return out


def _vision_source_shapes() -> list[tuple[str, tuple[int, ...]]]:
    """The 333 vision source tensors (all BF16) with exact shapes."""
    VH = VISION_HIDDEN
    out = [
        ("model.visual.patch_embed.proj.weight", (1152, 3, 2, 16, 16)),
        ("model.visual.patch_embed.proj.bias", (VH,)),
        ("model.visual.pos_embed.weight", (2304, VH)),
    ]
    for i in range(VISION_BLOCKS):
        P = f"model.visual.blocks.{i}."
        out += [
            (P + "attn.qkv.weight", (3 * VH, VH)),
            (P + "attn.qkv.bias", (3 * VH,)),
            (P + "attn.proj.weight", (VH, VH)),
            (P + "attn.proj.bias", (VH,)),
            (P + "mlp.linear_fc1.weight", (4304, VH)),
            (P + "mlp.linear_fc1.bias", (4304,)),
            (P + "mlp.linear_fc2.weight", (VH, 4304)),
            (P + "mlp.linear_fc2.bias", (VH,)),
            (P + "norm1.weight", (VH,)),
            (P + "norm1.bias", (VH,)),
            (P + "norm2.weight", (VH,)),
            (P + "norm2.bias", (VH,)),
        ]
    M = "model.visual.merger."
    out += [
        (M + "linear_fc1.weight", (4 * VH, 4 * VH)),
        (M + "linear_fc1.bias", (4 * VH,)),
        (M + "linear_fc2.weight", (HIDDEN, 4 * VH)),
        (M + "linear_fc2.bias", (HIDDEN,)),
        (M + "norm.weight", (VH,)),
        (M + "norm.bias", (VH,)),
    ]
    return out


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
    # MTP module: 31 source tensors (all BF16), exact shapes
    for name, shape in _mtp_tensor_shapes():
        if name not in G_HF.wm:
            raise ValueError(f"missing MTP tensor {name}")
        s, dt = G_HF.shape(name)
        if s != shape:
            raise ValueError(f"MTP shape {name}: {s}, want {shape}")
        if dt != "BF16":
            raise ValueError(f"MTP dtype {name}: {dt}, want BF16")
    # Vision: 333 source tensors (all BF16), exact shapes
    for name, shape in _vision_source_shapes():
        if name not in G_HF.wm:
            raise ValueError(f"missing vision tensor {name}")
        s, dt = G_HF.shape(name)
        if s != shape:
            raise ValueError(f"vision shape {name}: {s}, want {shape}")
        if dt != "BF16":
            raise ValueError(f"vision dtype {name}: {dt}, want BF16")
    # D10: the imatrix must carry no MTP bands (text 48 layers only)
    if any(n.startswith("blk.48.") for n in IMAT_IDS):
        raise ValueError("imatrix carries MTP (layer 48) bands; D10 requires the no-band Q4 path")
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

    ensure() submits the four encode maps (routed gate_up Q4/Q8 per expert,
    routed down Q4/Q8 per expert, the three shared-expert int parts, W8
    parts).  Producers pull from the maps in task order; the W8 part order
    matches the object row order:

      GDN:    0 qk | 1 v | 2 z | 3 ssm_out
      full:   0 q | 1 k | 2 gate | 3 v | 4 attn_out
    """

    def __init__(self, layer: int, pool: ProcessPoolExecutor) -> None:
        self.layer = layer
        self.pool = pool
        self.src = f"{LM}{layer}."
        self.it_gate_up = None
        self.it_down = None
        self.it_int = None
        self.it_w8 = None
        self.ready = False
        self.report: dict[str, object] = {"moe_code": _moe_code(self.src)}

    def ensure(self) -> None:
        if self.ready:
            return
        layer = self.layer
        src = self.src
        self.it_gate_up = self.pool.map(
            _task_gu, [(src, e) for e in range(E)], chunksize=8)
        self.it_down = self.pool.map(
            _task_dn, [(src, e) for e in range(E)], chunksize=8)
        m = f"{LM}{layer}.mlp"
        self.it_int = self.pool.map(_task_int_part, [
            (src, f"{m}.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
            (src, f"{m}.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
            (src, f"{m}.shared_expert.down_proj.weight", "ffn_down_shexp.weight"),
        ], chunksize=1)
        if layer % 4 != 3:
            qkv = f"{LM}{layer}.linear_attn.in_proj_qkv.weight"
            w8_tasks = [
                (qkv, 1, 0, QK_DIM, f"blk.{layer}.attn_qkv.weight.in_sum2"),
                (qkv, 5, QK_DIM, HC_DIM, f"blk.{layer}.attn_qkv.weight.in_sum2"),
                (f"{LM}{layer}.linear_attn.in_proj_z.weight", 4, 0, 0,
                 f"blk.{layer}.attn_gate.weight.in_sum2"),
                (f"{LM}{layer}.linear_attn.out_proj.weight", 6, 0, 0,
                 f"blk.{layer}.ssm_out.weight.in_sum2"),
            ]
        else:
            q = f"{LM}{layer}.self_attn.q_proj.weight"
            w8_tasks = [
                (q, 2, 0, 0, f"blk.{layer}.attn_q.weight.in_sum2"),
                (f"{LM}{layer}.self_attn.k_proj.weight", 0, 0, 0,
                 f"blk.{layer}.attn_k.weight.in_sum2"),
                (q, 3, 0, 0, f"blk.{layer}.attn_q.weight.in_sum2"),
                (f"{LM}{layer}.self_attn.v_proj.weight", 0, 0, 0,
                 f"blk.{layer}.attn_v.weight.in_sum2"),
                (f"{LM}{layer}.self_attn.o_proj.weight", 0, 0, 0,
                 f"blk.{layer}.attn_output.weight.in_sum2"),
            ]
        self.it_w8 = self.pool.map(_task_w8_part, w8_tasks, chunksize=1)
        self.ready = True

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
    return _rowsplit_payload(codes, scales, W8, shape)


def _rowsplit_payload(codes: np.ndarray, scales: np.ndarray, code: str, shape: tuple[int, int]) -> bytes:
    """Zero-pad the group axis to k_pad (a multiple of 128) and encode.

    The encoder emits k // gs groups; the row-split-k128 layout requires
    k_pad // gs.  Padding groups are all-zero codes with zero scales, which
    decode to exactly zero (a no-op column block).
    """
    n, k = shape
    gs = GROUPS[code]
    g_pad = layouts.row_split_geometry(code, shape).groups_per_row
    g = codes.shape[1]
    if g < g_pad:
        codes = np.pad(codes, ((0, 0), (0, g_pad - g), (0, 0)))
        scales = np.pad(scales, ((0, 0), (0, g_pad - g)))
    elif g != g_pad:
        raise RuntimeError(f"group count {g} != layout {g_pad} for {code} {shape}")
    return layouts.encode_row_split(
        torch.from_numpy(np.ascontiguousarray(codes)),
        torch.from_numpy(np.ascontiguousarray(scales)), code, shape)


def _routed_payload(st: "_LayerState | _MTPState", which: str, rows: int, k: int) -> bytes:
    """Assemble a routed bank [rows, k] from the per-expert encode iterator."""
    st.ensure()
    code = _moe_code(st.src)
    gs = GROUPS[code]
    it = st.it_gate_up if which == "gu" else st.it_down
    per = rows // E
    codes = np.empty((E, per, k // gs, gs), np.int8)
    scales = np.empty((E, per, k // gs), np.float16)
    for e in range(E):
        cb, sb = next(it)
        codes[e] = np.frombuffer(cb, np.int8).reshape(per, k // gs, gs)
        scales[e] = np.frombuffer(sb, np.float16).reshape(per, k // gs)
    codes = codes.reshape(rows, k // gs, gs)
    scales = scales.reshape(rows, k // gs)
    _parity_expert0(st, which, codes, scales)
    return _rowsplit_payload(codes, scales, code, (rows, k))


def _shared_payload(st: "_LayerState | _MTPState", which: str) -> bytes:
    """Assemble the layer's shared expert [rows, k] from the int-part iterator."""
    st.ensure()
    code = _moe_code(st.src)
    gs = GROUPS[code]
    if which == "gu":
        parts = [next(st.it_int) for _ in range(2)]
        rows, k = 2 * I, HIDDEN
    else:
        parts = [next(st.it_int)]
        rows, k = HIDDEN, I
    codes = np.concatenate([np.frombuffer(p[0], np.int8).reshape(p[2], k // gs, gs) for p in parts], axis=0)
    scales = np.concatenate([np.frombuffer(p[1], np.float16).reshape(p[2], k // gs) for p in parts], axis=0)
    if codes.shape[0] != rows:
        raise RuntimeError(f"shared {which} short: {codes.shape[0]}/{rows} rows")
    return _rowsplit_payload(codes, scales, code, (rows, k))


def _parity_expert0(st: "_LayerState | _MTPState", which: str, codes: np.ndarray, scales: np.ndarray) -> None:
    """Re-encode expert 0 in the parent; require exact byte parity."""
    src = st.src
    code = _moe_code(src)
    if which == "gu":
        ref = _encode_gu_expert(src, 0)
        rows = 2 * I
    else:
        ref = _encode_dn_expert(src, 0)
        rows = HIDDEN
    if not (np.array_equal(codes[:rows], ref[0]) and np.array_equal(scales[:rows], ref[1])):
        raise RuntimeError(f"{code} expert-0 byte parity failed ({src}, {which})")


def _router_gate(layer: int) -> bytes:
    r = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.mlp.gate.weight"))
    s = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.mlp.shared_expert_gate.weight"))
    _check_shape(f"router layer {layer}", r.shape, (E, HIDDEN))
    _check_shape(f"router shared layer {layer}", s.shape, (1, HIDDEN))
    return layouts.encode_direct(
        torch.from_numpy(np.ascontiguousarray(np.concatenate([r, s], axis=0), dtype=np.float32)), FP32)


def _mtp_router_gate() -> bytes:
    r = _bf16_f32(G_HF.rows_u16(f"{MTP_P}mlp.gate.weight"))
    s = _bf16_f32(G_HF.rows_u16(f"{MTP_P}mlp.shared_expert_gate.weight"))
    _check_shape("mtp router", r.shape, (E, HIDDEN))
    _check_shape("mtp router shared", s.shape, (1, HIDDEN))
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
    moe_code = _moe_code(f"{LM}{layer}.")
    units += [
        (_spec(f"{L}/moe/routed_gate_up", (NV_ROWS, NV_K), moe_code),
         lambda s=st: _routed_payload(s, "gu", NV_ROWS, NV_K)),
        (_spec(f"{L}/moe/routed_down", (Q6_ROWS, Q6_K), moe_code),
         lambda s=st: _routed_payload(s, "dn", Q6_ROWS, Q6_K)),
        (_spec(f"{L}/moe/router_gate", (E + 1, HIDDEN), FP32),
         lambda l=layer: _router_gate(l)),
        (_spec(f"{L}/moe/shared_gate_up", (2 * I, HIDDEN), moe_code),
         lambda s=st: _shared_payload(s, "gu")),
        (_spec(f"{L}/moe/shared_down", (HIDDEN, I), moe_code),
         lambda s=st: _shared_payload(s, "dn")),
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


# ---------------------------------------------------------------------------
# MTP module (28 objects: 5 Q8 MoE + 23 dense; D13/D14, no-band D10)
# ---------------------------------------------------------------------------

class _MTPState:
    """MTP pool state (lazy): Q8 routed experts + shared parts + W8 dense."""

    def __init__(self, pool: ProcessPoolExecutor) -> None:
        self.pool = pool
        self.src = MTP_P
        self.it_gate_up = None
        self.it_down = None
        self.it_int = None
        self.it_w8 = None
        self.ready = False

    def ensure(self) -> None:
        if self.ready:
            return
        src = self.src
        self.it_gate_up = self.pool.map(
            _task_gu, [(src, e) for e in range(E)], chunksize=8)
        self.it_down = self.pool.map(
            _task_dn, [(src, e) for e in range(E)], chunksize=8)
        self.it_int = self.pool.map(_task_int_part, [
            (src, f"{MTP_P}mlp.shared_expert.gate_proj.weight", None),
            (src, f"{MTP_P}mlp.shared_expert.up_proj.weight", None),
            (src, f"{MTP_P}mlp.shared_expert.down_proj.weight", None),
        ], chunksize=1)
        sa = f"{MTP_P}self_attn"
        self.it_w8 = self.pool.map(_task_w8_part, [
            (f"{MTP_HF}fc_embedding.weight", 0, 0, 0, None),
            (f"{MTP_HF}fc_hidden.weight", 0, 0, 0, None),
            (f"{MTP_HF}hyper_connection_mixer.input_mix_weight_down.weight", 0, 0, 0, None),
            (f"{MTP_HF}hyper_connection_mixer.input_mix_weight_up.weight", 0, 0, 0, None),
            (f"{MTP_P}attn_hyper_connection.block_inject_weight.weight", 0, 0, 0, None),
            (f"{MTP_P}attn_hyper_connection.input_mix_weight_up.weight", 0, 0, 0, None),
            (f"{MTP_P}attn_hyper_connection.input_mix_weight_down.weight", 0, 0, 0, None),
            (f"{MTP_P}mlp_hyper_connection.block_inject_weight.weight", 0, 0, 0, None),
            (f"{MTP_P}mlp_hyper_connection.input_mix_weight_up.weight", 0, 0, 0, None),
            (f"{MTP_P}mlp_hyper_connection.input_mix_weight_down.weight", 0, 0, 0, None),
            (f"{sa}.q_proj.weight", 2, 0, 0, None),
            (f"{sa}.k_proj.weight", 0, 0, 0, None),
            (f"{sa}.q_proj.weight", 3, 0, 0, None),
            (f"{sa}.v_proj.weight", 0, 0, 0, None),
            (f"{sa}.o_proj.weight", 0, 0, 0, None),
            (f"{sa}.indexer.index_qk_proj.weight", 1, 0, KV_DIM, None),
            (f"{sa}.indexer.index_qk_proj.weight", 1, KV_DIM, KV_DIM + IDX_K_DIM, None),
        ], chunksize=1)
        self.ready = True

    def w8(self, k: int):
        self.ensure()
        return _iter_take(self.it_w8, k)


def _mtp_units(mst: _MTPState) -> list[tuple[TensorSpec, object]]:
    qkqv = f"{MTP_P}self_attn"
    units: list[tuple[TensorSpec, object]] = [
        (_spec("mtp/fc_embedding", (HIDDEN, HIDDEN), W8),
         lambda s=mst: _w8_payload((HIDDEN, HIDDEN), s.w8(1))),
        (_spec("mtp/fc_hidden", (HIDDEN, HIDDEN), W8),
         lambda s=mst: _w8_payload((HIDDEN, HIDDEN), s.w8(1))),
        (_spec("mtp/pre_fc_norm_embedding", (HIDDEN,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16(f"{MTP_HF}pre_fc_norm_embedding.weight"))),
        (_spec("mtp/pre_fc_norm_hidden", (HC_DIM,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16(f"{MTP_HF}pre_fc_norm_hidden.weight"))),
        (_spec("mtp/hc/norm", (HC_DIM,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16(f"{MTP_HF}hyper_connection_mixer.hc_norm.weight"))),
        (_spec("mtp/hc/input_mix_weight_down", (HC_LR, HC_DIM), W8),
         lambda s=mst: _w8_payload((HC_LR, HC_DIM), s.w8(1))),
        (_spec("mtp/hc/input_mix_weight_up", (HC_DIM, HC_LR), W8),
         lambda s=mst: _w8_payload((HC_DIM, HC_LR), s.w8(1))),
    ]
    for section, hfsec in (("attn", "attn"), ("mlp", "mlp")):
        P = f"{MTP_P}{hfsec}_hyper_connection"
        units += [
            (_spec(f"mtp/layer/{section}_hc/block_inject", (HC_DIM // HIDDEN, HC_DIM), W8),
             lambda s=mst: _w8_payload((HC_DIM // HIDDEN, HC_DIM), s.w8(1))),
            (_spec(f"mtp/layer/{section}_hc/input_mix_weight_up", (HC_DIM, HC_LR), W8),
             lambda s=mst: _w8_payload((HC_DIM, HC_LR), s.w8(1))),
            (_spec(f"mtp/layer/{section}_hc/input_mix_weight_down", (HC_LR, HC_DIM), W8),
             lambda s=mst: _w8_payload((HC_LR, HC_DIM), s.w8(1))),
            (_spec(f"mtp/layer/{section}_hc/hc_norm", (HC_DIM,), BF16),
             lambda P=P: _bf16_bytes(G_HF.rows_u16(f"{P}.hc_norm.weight"))),
        ]
    units += [
        (_spec("mtp/layer/attention/query_key_gate_value", (QKQV_ROWS, HIDDEN), W8),
         lambda s=mst: _w8_payload((QKQV_ROWS, HIDDEN), s.w8(4))),
        (_spec("mtp/layer/attention/query_norm", (HEAD_DIM,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16(f"{qkqv}.q_norm.weight"))),
        (_spec("mtp/layer/attention/key_norm", (HEAD_DIM,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16(f"{qkqv}.k_norm.weight"))),
        (_spec("mtp/layer/attention/output", (HIDDEN, V_DIM), W8),
         lambda s=mst: _w8_payload((HIDDEN, V_DIM), s.w8(1))),
        (_spec("mtp/layer/indexer/query_proj", (KV_DIM, HIDDEN), W8),
         lambda s=mst: _w8_payload((KV_DIM, HIDDEN), s.w8(1))),
        (_spec("mtp/layer/indexer/key_proj", (IDX_K_DIM, HIDDEN), W8),
         lambda s=mst: _w8_payload((IDX_K_DIM, HIDDEN), s.w8(1))),
        (_spec("mtp/layer/indexer/query_norm", (IDX_K_DIM,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16(f"{qkqv}.indexer.q_layernorm.weight"))),
        (_spec("mtp/layer/indexer/key_norm", (IDX_K_DIM,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16(f"{qkqv}.indexer.k_layernorm.weight"))),
        (_spec("mtp/moe/routed_gate_up", (NV_ROWS, NV_K), Q8),
         lambda s=mst: _routed_payload(s, "gu", NV_ROWS, NV_K)),
        (_spec("mtp/moe/routed_down", (Q6_ROWS, Q6_K), Q8),
         lambda s=mst: _routed_payload(s, "dn", Q6_ROWS, Q6_K)),
        (_spec("mtp/moe/router_gate", (E + 1, HIDDEN), FP32),
         _mtp_router_gate),
        (_spec("mtp/moe/shared_gate_up", (2 * I, HIDDEN), Q8),
         lambda s=mst: _shared_payload(s, "gu")),
        (_spec("mtp/moe/shared_down", (HIDDEN, I), Q8),
         lambda s=mst: _shared_payload(s, "dn")),
    ]
    return units


# ---------------------------------------------------------------------------
# vision (333 objects; the 27B quantized map, no-band D10)
# ---------------------------------------------------------------------------

class _VisionState:
    """Vision pool state: one no-band encode task per quantized tensor."""

    def __init__(self, pool: ProcessPoolExecutor) -> None:
        self.pool = pool
        self.it = None

    def ensure(self):
        if self.it is None:
            tasks = [
                ("model.visual.patch_embed.proj.weight", Q6),
            ]
            for i in range(VISION_BLOCKS):
                P = f"model.visual.blocks.{i}."
                tasks += [
                    (P + "attn.qkv.weight", Q4),
                    (P + "attn.proj.weight", Q5),
                    (P + "mlp.linear_fc1.weight", Q4),
                    (P + "mlp.linear_fc2.weight", Q5),
                ]
            tasks += [
                ("model.visual.merger.linear_fc1.weight", W8),
                ("model.visual.merger.linear_fc2.weight", W8),
            ]
            self.it = self.pool.map(_task_vision_part, tasks, chunksize=1)
        return self.it


def _vision_payload(it, code: str, shape: tuple[int, int]) -> bytes:
    cb, sb, nr = next(it)
    n, k = shape
    g = layouts.row_split_geometry(code, shape).groups_per_row
    if nr != n:
        raise RuntimeError(f"vision assembly short: {nr}/{n} rows")
    codes = np.frombuffer(cb, np.int8).reshape(n, g, GROUPS[code])
    scales = np.frombuffer(sb, np.float16).reshape(n, g)
    return _rowsplit_payload(codes, scales, code, shape)


def _vision_units(vst: _VisionState) -> list[tuple[TensorSpec, object]]:
    VH = VISION_HIDDEN
    it = vst.ensure()
    units: list[tuple[TensorSpec, object]] = [
        (_spec("vision/patch_embedding", (VH, 3 * 2 * 16 * 16), Q6),
         lambda: _vision_payload(it, Q6, (VH, 3 * 2 * 16 * 16))),
        (_spec("vision/patch_embedding_bias", (VH,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16("model.visual.patch_embed.proj.bias"))),
        (_spec("vision/position_embedding", (2304, VH), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16("model.visual.pos_embed.weight"))),
    ]
    for i in range(VISION_BLOCKS):
        P = f"model.visual.blocks.{i}."
        L = f"vision/layers/{i}"
        units += [
            (_spec(f"{L}/attention/qkv", (3 * VH, VH), Q4),
             lambda: _vision_payload(it, Q4, (3 * VH, VH))),
            (_spec(f"{L}/attention/qkv_bias", (3 * VH,), BF16),
             lambda P=P: _bf16_bytes(G_HF.rows_u16(P + "attn.qkv.bias"))),
            (_spec(f"{L}/attention/output", (VH, VH), Q5),
             lambda: _vision_payload(it, Q5, (VH, VH))),
            (_spec(f"{L}/attention/output_bias", (VH,), BF16),
             lambda P=P: _bf16_bytes(G_HF.rows_u16(P + "attn.proj.bias"))),
            (_spec(f"{L}/mlp/fc1", (4304, VH), Q4),
             lambda: _vision_payload(it, Q4, (4304, VH))),
            (_spec(f"{L}/mlp/fc1_bias", (4304,), BF16),
             lambda P=P: _bf16_bytes(G_HF.rows_u16(P + "mlp.linear_fc1.bias"))),
            (_spec(f"{L}/mlp/fc2", (VH, 4304), Q5),
             lambda: _vision_payload(it, Q5, (VH, 4304))),
            (_spec(f"{L}/mlp/fc2_bias", (VH,), BF16),
             lambda P=P: _bf16_bytes(G_HF.rows_u16(P + "mlp.linear_fc2.bias"))),
            (_spec(f"{L}/norm1/weight", (VH,), BF16),
             lambda P=P: _bf16_bytes(G_HF.rows_u16(P + "norm1.weight"))),
            (_spec(f"{L}/norm1/bias", (VH,), BF16),
             lambda P=P: _bf16_bytes(G_HF.rows_u16(P + "norm1.bias"))),
            (_spec(f"{L}/norm2/weight", (VH,), BF16),
             lambda P=P: _bf16_bytes(G_HF.rows_u16(P + "norm2.weight"))),
            (_spec(f"{L}/norm2/bias", (VH,), BF16),
             lambda P=P: _bf16_bytes(G_HF.rows_u16(P + "norm2.bias"))),
        ]
    units += [
        (_spec("vision/merger/fc1", (4 * VH, 4 * VH), W8),
         lambda: _vision_payload(it, W8, (4 * VH, 4 * VH))),
        (_spec("vision/merger/fc1_bias", (4 * VH,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16("model.visual.merger.linear_fc1.bias"))),
        (_spec("vision/merger/fc2", (HIDDEN, 4 * VH), W8),
         lambda: _vision_payload(it, W8, (HIDDEN, 4 * VH))),
        (_spec("vision/merger/fc2_bias", (HIDDEN,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16("model.visual.merger.linear_fc2.bias"))),
        (_spec("vision/merger/norm/weight", (VH,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16("model.visual.merger.norm.weight"))),
        (_spec("vision/merger/norm/bias", (VH,), BF16),
         lambda: _bf16_bytes(G_HF.rows_u16("model.visual.merger.norm.bias"))),
    ]
    return units


def build_plan(layers: tuple[int, ...], mst: _MTPState, vst: _VisionState):
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
    # the PLE table is always planned; its payload is copied by the single
    # background thread (_ple_table_direct), not by the writer loop
    add(_spec(PLE_TABLE_NAME, (PLE_ROWS, PLE_DIM), BF16), None)
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
    for spec, producer in _mtp_units(mst):
        add(spec, producer)
    for spec, producer in _vision_units(vst):
        add(spec, producer)
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
    print(f"[jbq4] opening {args.model}", flush=True)
    G_HF = _HFSource(Path(args.model))
    G_IMAT = GGUFReader(args.imatrix)
    IMAT_IDS = {t.name: i for i, t in enumerate(G_IMAT.tensors)}
    layers = _parse_layers(args.layers)
    if args.smoke and not args.layers:
        layers = (0, 1, 3)
    print(f"[jbq4] layers={layers} workers={args.workers} smoke={args.smoke}", flush=True)
    checks = _preflight(layers, Path(args.resources))
    print(f"[jbq4] preflight OK ({len(G_HF.wm)} checkpoint tensors, "
          f"{len(IMAT_IDS)} imatrix tensors)", flush=True)
    for k, v in checks.items():
        print(f"[jbq4]   {k} = {v}", flush=True)

    resource_map = {}
    for name in RESOURCE_NAMES:
        data = Path(args.resources, name.removeprefix("frontend/")).read_bytes()
        if not data:
            raise ValueError(f"empty frontend resource {name}")
        resource_map[name] = data

    pool = ProcessPoolExecutor(max_workers=args.workers)
    try:
        LAYER_STATES = {l: _LayerState(l, pool) for l in layers}
        specs, producers = build_plan(layers, _MTPState(pool), _VisionState(pool))
        plan = family_conversion.build_object_plan(specs, resource_map)
        total_bytes = plan.payload_span_bytes
        by_format: dict[str, int] = {}
        for obj in plan.objects:
            fmt = getattr(obj, "format", "resource")
            by_format[fmt] = by_format.get(fmt, 0) + 1
        print(f"[jbq4] plan: {len(specs)} objects, {total_bytes / 1e9:.3f} GB payload; "
              + " ".join(f"{k}:{v}" for k, v in sorted(by_format.items())), flush=True)
        if args.plan_only:
            print("[jbq4] --plan-only: no artifact written", flush=True)
            return 0

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = ArtifactWriter(out_path, ArtifactIdentity(MODEL_ID, WEIGHTS_ID), plan.specs)
        if writer.objects != plan.objects:
            writer.close()
            raise RuntimeError("writer object plan differs from build plan")
        ple_obj = next(o for o in writer.objects if o.name == PLE_TABLE_NAME)
        ple_abs = writer.payload_offset + ple_obj.offset
        file_size = writer.payload_offset + writer.objects[-1].offset + writer.objects[-1].bytes
        ple_done: dict[str, object] = {}
        ple_stop = threading.Event()
        ple_thread = threading.Thread(
            target=_ple_table_direct,
            args=(out_path, ple_abs, file_size, ple_done, ple_stop),
            name="ple-table-direct")
        ple_thread.start()
        try:
            for i, spec in enumerate(specs, start=1):
                name = spec.name
                if name == PLE_TABLE_NAME:
                    writer.skip(name)
                    print(f"[{i}/{len(specs)}] {name}  (background direct copy)  t={time.time() - t0:7.1f}s", flush=True)
                    continue
                payload = resource_map[name] if isinstance(spec, ResourceSpec) else producers[name]()
                writer.write(name, payload)
                print(f"[{i}/{len(specs)}] {name}  t={time.time() - t0:7.1f}s", flush=True)
            ple_thread.join()
            if "error" in ple_done:
                raise RuntimeError(f"PLE table direct copy failed: {ple_done['error']}")
            writer.finish()
        finally:
            ple_stop.set()
            writer.close()
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
            "mtp": {"moe_code": Q8, "dense": W8, "bands": "none (D10)"},
            "objects": len(specs),
            "payload_bytes": total_bytes,
            "file_bytes": file_bytes,
            "workers": args.workers,
            "smoke": bool(args.smoke),
            "ple_table": {
                "rows": PLE_ROWS,
                "bytes": ple_done.get("bytes"),
                "mode": "single background thread, direct shard->file pread/pwrite",
            },
            "elapsed_s": round(elapsed, 1),
            "encoder": "tools/convert/common/encoder_imatrix.py (pure numpy, imatrix-aware)",
            "conventions": {
                "hf_layout": "source (out,in); artifact (out,in)",
                "gdn_v_tiling": "V heads grouped->tiled (r=3) on qkv v-rows, z, a/b, A_log/dt, conv v-channels, out cols",
                "gdn_a_log": "-exp(reordered(A_log)) via torch FP32 exp (linear decay rate, sign-flipped)",
                "norms_folded": "f32(1.0) + f32(bf16(w)) for HC/mixer/q/k/indexer/PLE norms; GDN 128-dim norm not folded",
                "moe_code_map": "D14: text layers {0,1,47} + MTP = Q8G64_F16S; other 45 text layers = Q4G64_F16S",
                "experts": "[E,1280,2560] per-expert [gate 640 | up 640], K=2560; row-split-k128 group-64; Q4 [-8,7] / Q8 [-128,127]",
                "routed_down": "e-major, [2560,640], same codec as the layer routed_gate_up",
                "shared_experts": "[1280,2560]/[2560,640], layer codec, 1-D imatrix bands (text) / no-band (MTP)",
                "w8": "W8G32_F16S (int8, group-32); text dense with 1-D per-source imatrix bands; MTP dense no-band",
                "w8_padding": "k not a multiple of 128 (MTP input_mix_weight_up k=320): zero codes + zero scales pad groups to k_pad",
                "mtp_dense": "all 2-D linears W8G32_F16S no-band; 1-D norms BF16 passthrough",
                "mtp_qkqv": "fuses q|k|gate|v with the text full-attention de-interleave (q/gate from q_proj, r=24 head interleave)",
                "vision": "27B quantized map, no-band: patch Q6G64; block qkv/fc1 Q4G64; block output/fc2 Q5G64; merger W8G32; vectors BF16",
                "vision_padding": "mlp/fc2 k=4304 is not a multiple of the Q5 group size 64: 48 zero input columns pad k to k_pad=4352 before encoding (padded columns decode to exactly zero)",
                "ple_u64": "checkpoint I64 tensors, I32 lo/hi pairs, lo first (no I64 in the artifact)",
                "ple_table": "BF16 [320001536,160] = shards 0..127 concatenated in index order",
                "full_attn_deinterleave": "head h: q=rows h*512..h*512+255, gate=h*512+256..(h+1)*512-1",
                "hc_head": "no inject tensor at the head (graph passes nullptr)",
            },
        }
        report_path = out_path.with_suffix(out_path.suffix + ".conversion.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[jbq4] done in {elapsed:.1f}s: {file_bytes / 1e9:.3f} GB -> {out_path}", flush=True)
        print(f"[jbq4] report -> {report_path}", flush=True)
        return 0
    finally:
        pool.shutdown(wait=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Build the jbq4 .ninfer artifact (all-MoE Q4/Q8, D14) for "
                    "Qwen3.8-Flash-Next (qwen4exp) from the HF safetensors "
                    "checkpoint + unsloth imatrix.")
    ap.add_argument("--model", required=True, help="HF safetensors checkpoint dir (131 shards)")
    ap.add_argument("--imatrix", required=True, help="unsloth imatrix GGUF")
    ap.add_argument("--resources", required=True,
                    help="checkpoint dir with the six frontend resource files")
    ap.add_argument("--out", required=True, help="output .ninfer path")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--layers", default=None,
                    help="layer subset, e.g. '0-3' or '0,1,3' (default: all 48)")
    ap.add_argument("--smoke", action="store_true",
                    help="small subset run (text layers 0,1,3 + full MTP + full "
                         "vision; the full PLE table is copied by a single "
                         "background thread)")
    ap.add_argument("--plan-only", action="store_true",
                    help="build and print the object plan; write nothing")
    args = ap.parse_args(argv)
    sys.exit(convert(args))


if __name__ == "__main__":
    main()
