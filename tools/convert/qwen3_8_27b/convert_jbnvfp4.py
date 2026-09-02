"""JB-NVFP4 ``.ninfer`` artifact builder for Qwen3.8-27B (dense qwen35),
from the HF safetensors checkpoint.

Source of truth
---------------
* HF safetensors checkpoint (``Qwen3.8-27B-BF16``, 18 shards, ~55.5 GB).
  Text weights live under ``model.language_model.*``; the multimodal
  sections are NOT read from the checkpoint.
* unsloth imatrix file (``imatrix_unsloth_qwen3.8-27B.gguf_file``) — per-tensor
  input second-moment bands (``in_sum2``) and per-tensor token counts.
* the existing ``qwen3.8-27b`` artifact (``qwen3_8_27b_nvfp4.ninfer``): source
  for the imatrix-free sections — the vision tower (333 objects), the MTP
  module (12 objects), and the draft head (2 objects), which are copied
  byte-for-byte.
* the six frontend resources: read from the checkpoint directory and
  sha256-verified against the pinned official hashes.

Numerics
--------
The encoder is the shared pure-numpy imatrix-aware selection from
``tools/convert/common/encoder_imatrix.py``: the per-object objective is the
importance-weighted MSE of the quantization error.  No llama-quantize / C++
on this path.

Recipe (matches the existing artifact's format allocation 100%)
----------------------------------------------------------------
* MLP layers 0-55: ``mlp/gate_up`` NVFP4 + ``mlp/down`` NVFP4 + per-object
  weight divisor (FP32 scalar) and input scale divisor (FP32 scalar);
  layers 56-63: ``mlp/gate_up`` / ``mlp/down`` FP8.
* GDN layers (l % 4 != 3): ``gdn/query_key_value_z`` (q|k|v|z) and
  ``gdn/output`` FP8; projections stored direct (norms BF16, a_log/dt_bias
  FP32, convolution BF16, a_b_projection BF16).
* Full-attention layers (l % 4 == 3): ``attention/query_key_gate_value``
  (de-interleaved q|k|gate|v) and ``attention/output`` FP8; q/k norms BF16.
* ``text/token_embedding`` and ``text/output_head``: FP8 reference profile
  (no imatrix entry for either tensor).
* Vision tower / MTP / draft head: byte copies from the base artifact
  (same source checkpoint, reference quantization, no imatrix).

HF -> artifact transforms (verified byte-exact against the canonical GGUF,
``/tmp/27b_probe.py``, 31/31 PASS — see stage1.md)
---------------------------------------------------
* GDN V-head tiling (grouped -> tiled; NK=16, R=3, DV=128): the qkv v rows
  (tensor rows 4096:10240), the z rows, and the conv v channels are row-
  permuted by ``VPERM`` (6144); the a/b rows and the A_log/dt_bias elements
  by ``AP48`` (48); ``out_proj`` is column-permuted by ``VPERM``.
* ``a_log`` = ``-exp(reordered(A_log))`` via torch f32 exp (sign-flipped
  decay); ``dt_bias`` = f32(bf16(reordered(dt))); ``conv`` = transposed
  [4, 10240] (bf16-representable f32) RNE-cast to BF16; ``a|b`` = vstack.
* Norms: the GGUF stores the pre-folded ``1 + w``; the HF checkpoint does
  not, so the artifact's BF16 norm objects are the weights directly.
* Full-attention ``q_proj`` interleaves q|gate per head (head h: q = rows
  ``h*512 .. h*512+255``, gate = ``h*512+256 .. h*512+511``); the artifact
  ships ``q | k | gate | v`` de-interleaved.
* Bands (``in_sum2``) are 1-D per input column, already in artifact (tiled)
  order — used as-is; only weight rows/columns are permuted.

Canonical invocation
--------------------
::

    /home/robot/workplace/unsloth_env/bin/python -m \
        tools.convert.qwen3_8_27b.convert_jbnvfp4 \
        --model /llm/models/Qwen3.8-27B-BF16 \
        --imatrix /llm/models/imatrix_unsloth_qwen3.8-27B.gguf_file \
        --base-artifact /llm/models/qwen3_8_27b_nvfp4.ninfer \
        --out /llm/models/Qwen3.8-27B-JB-NVFP4.ninfer \
        --workers 32

Execution model
---------------
Parent process: safetensors index, imatrix reader, base artifact reader,
object plan, per-layer futures.  A ``ProcessPoolExecutor`` (fork) does the
heavy per-row-chunk encodes; workers read safetensors shards with
``os.pread`` (position-independent, fork-safe) and the imatrix GGUF via
memmap.
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

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.artifact import layouts
from tools.convert.common import encoder_imatrix as enc
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_8_27b import inventory_nvfp4 as inv
from tools.convert.qwen3_8_27b.convert import OFFICIAL_RESOURCE_SHA256

MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "jbnvfp4"

# --- model geometry (shape-asserted at preflight) ---------------------------
HIDDEN = 5120
FFN = 17408
LAYERS = 64
HEADS = 24           # full-attention q heads
HEAD_DIM = 256
KV_HEADS = HEADS // 6   # full-attention k/v heads (4)
GDN_V_HEADS = 48       # GDN value heads
GDN_HEAD_V = 128       # GDN head_v dim
VOCAB = 248320
# GDN projection output rows.  v = 48*128 = 6144 = z; q|k = 10240 - 6144.
GDN_QKV_OUT = 10240            # GDN q|k|v projection output (in_proj_qkv rows)
GDN_QK_OUT = GDN_QKV_OUT - GDN_V_HEADS * GDN_HEAD_V   # 4096: q|k rows
GDN_Z_OUT = GDN_V_HEADS * GDN_HEAD_V   # 6144: z (in_proj_z rows)
GDN_QKVZ_OUT = GDN_QKV_OUT + GDN_Z_OUT # 16384
FULL_QG_OUT = HEADS * HEAD_DIM         # 6144 rows of q, or of gate
FULL_KV_OUT = KV_HEADS * HEAD_DIM      # 1024
FULL_QKVZ_OUT = 2 * FULL_QG_OUT + 2 * FULL_KV_OUT  # 14336
PROJ_OUT = HEADS * HEAD_DIM            # 6144: gdn output / attn output in-dim
GATE_UP_OUT = 2 * FFN                  # 34816

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
FP8 = "FP8_E4M3FN_ROW_BF16S"

LM = "model.language_model.layers."

# q|gate de-interleave row indexes on the q_proj output axis.
_Q_IDX = np.repeat(np.arange(HEADS) * 2 * HEAD_DIM, HEAD_DIM).astype(np.intp)
_G_IDX = _Q_IDX + HEAD_DIM

# GDN V-head tiling permutations (grouped -> tiled; NK=16, R=3, DV=128).
_V_PERM: np.ndarray = (
    np.arange(16 * 3 * 128).reshape(16, 3, 128).transpose(1, 0, 2).reshape(-1)
)                                            # 6144 (v rows/cols/conv ch)
_AP48: np.ndarray = (
    np.arange(16 * 3).reshape(16, 3).transpose(1, 0).reshape(-1)
)                                            # 48 (a/b/A_log/dt elements)

NV_CHUNK = 256          # encoder band-search row chunk (worker RAM bound)
ROW_TARGET = 2048       # target rows per pool task
DOWN_ROW_TARGET = 1024  # wider K (17408) -> smaller tasks

G_HF: "_HFSource"
G_IMAT: GGUFReader
G_OLD: Artifact
G_POOL: ProcessPoolExecutor | None = None
G_RES_MAP: dict[str, bytes] = {}
IMAT_IDS: dict[str, int]
LAYER_STATES: dict[int, "_LayerState"]

RESOURCE_NAMES = tuple(s.name for s in inv.OBJECT_SPECS
                       if s.name.startswith("frontend/"))
COPIED_NAMES = tuple(s.name for s in inv.OBJECT_SPECS
                     if s.name.startswith(("mtp/", "vision/"))
                     or s.name in ("text/draft_head",
                                   "text/draft_head_token_ids"))


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def _bf16_f32(raw_u16: np.ndarray) -> np.ndarray:
    """Bit-exact BF16 -> FP32 (bf16 word << 16 as f32 bits)."""
    return (raw_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _bf16_bytes(blk_u16: np.ndarray) -> bytes:
    """BF16 rows (u16) -> direct-format BF16 object bytes."""
    return layouts.encode_direct(
        torch.from_numpy(np.ascontiguousarray(blk_u16)).view(torch.bfloat16),
        BF16)


def _f32_bytes(b: bytes) -> bytes:
    return layouts.encode_direct(
        torch.from_numpy(np.frombuffer(b, np.float32)), FP32)


def _imat(name: str) -> np.ndarray:
    return G_IMAT.get_tensor(IMAT_IDS[name]).data


def _band_name(layer: int, role: str) -> str:
    return f"blk.{layer}.{role}.weight.in_sum2"


def _band(layer: int, role: str) -> np.ndarray:
    return np.asarray(_imat(_band_name(layer, role)),
                      dtype=np.float32).ravel()


def _dw_bytes(amax: float) -> bytes:
    """d_w = f32(2688 / amax) little-endian (4 bytes)."""
    if amax <= 0.0:
        return struct.pack("<f", np.float32(2688.0))
    return struct.pack("<f", np.float32(np.float32(2688.0) / np.float32(amax)))


def _row_chunks(n: int, target: int) -> list[tuple[int, int]]:
    """Split n rows into ~n/target chunks, each a multiple of 128."""
    if n % 128:
        raise ValueError(f"row count {n} is not a multiple of 128")
    n_parts = max(1, -(-n // target))
    base = (n // 128) // n_parts * 128
    extra = n - base * n_parts          # multiple of 128, folded into part 0
    out: list[tuple[int, int]] = []
    r = 0
    for i in range(n_parts):
        take = base + (extra if i == 0 else 0)
        out.append((r, r + take))
        r += take
    return out


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
        ent = self._entry(name)[2]
        return tuple(ent["shape"]), ent["dtype"]

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


# ---------------------------------------------------------------------------
# worker tasks (module level so they fork-inherit the readers)
# ---------------------------------------------------------------------------

def _task_amax(args: tuple[str,]) -> float:
    """FP32 absolute max over a full BF16 matrix (NVFP4 d_w input)."""
    (name,) = args
    blk = np.ascontiguousarray(G_HF.rows_u16(name))
    return float(np.abs(_bf16_f32(blk)).max())


def _task_nvfp4(args: tuple[str, int, int, bytes, np.ndarray]) -> tuple[bytes, bytes]:
    """One row range of one HF matrix, NVFP4 t-space encode."""
    name, r0, r1, dw, band = args
    w = _bf16_f32(np.ascontiguousarray(G_HF.rows_u16(name, r0, r1)))
    t = w * np.frombuffer(dw, np.float32)
    packed, words = enc.encode_nvfp4(t, band, row_chunk=NV_CHUNK)
    return packed.tobytes(), words.tobytes()


def _task_fp8(args: tuple[str, int, int, int, np.ndarray]) -> tuple[bytes, bytes]:
    """One row range of one HF matrix, imatrix-aware FP8 encode.

    mode 0: rows [r0:r1) of the source; mode 1: de-interleaved q rows of a
    full-attention q_proj (band = the attn_q band); mode 2: the gate rows;
    mode 3: the GDN v block (in_proj_qkv rows GDN_QK_OUT:), VPERM rows;
    mode 4: the whole matrix with VPERM rows (in_proj_z); mode 5: rows
    [r0:r1) with VPERM columns (out_proj).
    """
    name, mode, r0, r1, band = args
    if mode == 0:
        blk = np.ascontiguousarray(G_HF.rows_u16(name, r0, r1))
    elif mode == 1:
        blk = np.ascontiguousarray(G_HF.rows_u16(name)[_Q_IDX])
    elif mode == 2:
        blk = np.ascontiguousarray(G_HF.rows_u16(name)[_G_IDX])
    elif mode == 3:
        blk = np.ascontiguousarray(
            G_HF.rows_u16(name, GDN_QK_OUT, GDN_QKV_OUT)[_V_PERM])
    elif mode == 4:
        blk = np.ascontiguousarray(G_HF.rows_u16(name)[_V_PERM])
    elif mode == 5:
        blk = np.ascontiguousarray(G_HF.rows_u16(name, r0, r1)[:, _V_PERM])
    else:
        raise ValueError(f"unknown mode {mode}")
    w = _bf16_f32(blk)
    codes, scales = enc.quantize_fp8_row_scaled(w, band, row_chunk=NV_CHUNK)
    return codes.tobytes(), scales.view(np.uint16).tobytes()


# ---------------------------------------------------------------------------
# parent-side assembly
# ---------------------------------------------------------------------------

def _parts(futs: list[tuple]) -> list[tuple[bytes, bytes, int]]:
    """(future, n_rows) list -> (payload1, payload2, n_rows) in order."""
    return [(f.result()[0], f.result()[1], nr) for f, nr in futs]


def _assemble_fp8(shape: tuple[int, int],
                  parts: list[tuple[bytes, bytes, int]]) -> bytes:
    n, k = shape
    codes = np.zeros((n, k), dtype=np.uint8)
    scales = np.zeros(n, dtype=np.uint16)
    r = 0
    for cb, sb, nr in parts:
        codes[r:r + nr] = np.frombuffer(cb, np.uint8).reshape(nr, k)
        scales[r:r + nr] = np.frombuffer(sb, np.uint16).reshape(nr)
        r += nr
    if r != n:
        raise RuntimeError(f"fp8 assembly for {shape}: {r} != {n} rows")
    return layouts.encode_fp8_row_scaled(
        torch.from_numpy(codes),
        torch.from_numpy(scales).view(torch.bfloat16), shape)


def _assemble_nvfp4(shape: tuple[int, int],
                    parts: list[tuple[bytes, bytes, int]],
                    divisor: bytes) -> bytes:
    n, k = shape
    packed = np.zeros((n, k // 2), dtype=np.uint8)
    words = np.zeros((n, k // 16), dtype=np.uint8)
    r = 0
    for pb, wb, nr in parts:
        packed[r:r + nr] = np.frombuffer(pb, np.uint8).reshape(nr, k // 2)
        words[r:r + nr] = np.frombuffer(wb, np.uint8).reshape(nr, k // 16)
        r += nr
    if r != n:
        raise RuntimeError(f"nvfp4 assembly for {shape}: {r} != {n} rows")
    return layouts.encode_nvfp4(torch.from_numpy(packed),
                                torch.from_numpy(words), divisor, shape)


def _norm_bf16(hf_name: str) -> bytes:
    """Norm gamma: the artifact stores the unfolded weight directly (the
    1+w fold is applied by the runtime)."""
    return _bf16_bytes(G_HF.rows_u16(hf_name))


def _a_log(layer: int) -> bytes:
    """GDN a_log = -exp(reordered(A_log)) via torch f32 exp (sign-flipped
    decay)."""
    al = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.linear_attn.A_log"))[_AP48]
    t = -torch.exp(torch.from_numpy(np.ascontiguousarray(al)))
    return layouts.encode_direct(t, FP32)


def _dt_bias(layer: int) -> bytes:
    """GDN dt_bias in reordered (V-tiled) row order, f32."""
    d = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.linear_attn.dt_bias"))[_AP48]
    return layouts.encode_direct(torch.from_numpy(np.ascontiguousarray(d)),
                                 FP32)


def _convolution(layer: int) -> bytes:
    """GDN conv kernel in artifact tap-major [4, C] order.

    HF stores [C, 1, 4] (channel-major, 4 taps per channel); the V channel
    block (channels GDN_QK_OUT..) is V-head-tiled.  The artifact stores the
    transposed [K, C] form, RNE-cast to BF16 (the f32 values are
    bf16-representable, so the cast is exact).
    """
    c = _bf16_f32(G_HF.rows_u16(f"{LM}{layer}.linear_attn.conv1d.weight")
                  .reshape(GDN_QKV_OUT, 4))
    c = np.concatenate([c[:GDN_QK_OUT], c[GDN_QK_OUT:][_V_PERM]], axis=0)
    t = torch.from_numpy(np.ascontiguousarray(c.T)).to(torch.bfloat16)
    return layouts.encode_direct(t, BF16)


def _a_b_projection(layer: int) -> bytes:
    """gdn/a_b_projection = [a (48 rows) | b (48 rows)], V-tiled row order."""
    a = G_HF.rows_u16(f"{LM}{layer}.linear_attn.in_proj_a.weight")[_AP48]
    b = G_HF.rows_u16(f"{LM}{layer}.linear_attn.in_proj_b.weight")[_AP48]
    return _bf16_bytes(np.concatenate([a, b], axis=0))


def _fp8_whole(name: str, shape: tuple[int, int],
               band: np.ndarray | None) -> bytes:
    """One whole HF matrix as FP8, row-chunked tasks."""
    n, _ = shape
    futs: list[tuple] = []
    for r0, r1 in _row_chunks(n, ROW_TARGET):
        futs.append((G_POOL.submit(_task_fp8, (name, 0, r0, r1, band)), r1 - r0))
    return _assemble_fp8(shape, _parts(futs))


# ---------------------------------------------------------------------------
# per-layer state (lazy: submit on first object, block on its producer)
# ---------------------------------------------------------------------------

class _LayerState:
    def __init__(self, layer: int):
        self.layer = layer
        self.ready = False
        self.report: dict[str, object] = {
            "is_full_attention": layer in inv.FULL_ATTENTION_LAYERS,
        }

    def _band_roles(self) -> list[str]:
        roles = ["ffn_gate", "ffn_up", "ffn_down"]
        if self.layer in inv.GDN_LAYERS:
            roles += ["attn_qkv", "attn_gate", "ssm_out"]
        else:
            roles += ["attn_q", "attn_k", "attn_v", "attn_output"]
        return roles

    def _hf_names(self) -> dict[str, str]:
        l = self.layer
        if l in inv.GDN_LAYERS:
            return {
                "qkv": f"{LM}{l}.linear_attn.in_proj_qkv.weight",
                "z": f"{LM}{l}.linear_attn.in_proj_z.weight",
                "out": f"{LM}{l}.linear_attn.out_proj.weight",
            }
        return {
            "q": f"{LM}{l}.self_attn.q_proj.weight",
            "k": f"{LM}{l}.self_attn.k_proj.weight",
            "v": f"{LM}{l}.self_attn.v_proj.weight",
            "out": f"{LM}{l}.self_attn.o_proj.weight",
        }

    def ensure(self) -> None:
        if self.ready:
            return
        l = self.layer
        pool = G_POOL
        names = self._hf_names()
        bands = {role: _band(l, role) for role in self._band_roles()}

        if l in inv.NVFP4_MLP_LAYERS:
            ag = pool.submit(_task_amax, (f"{LM}{l}.mlp.gate_proj.weight",)).result()
            au = pool.submit(_task_amax, (f"{LM}{l}.mlp.up_proj.weight",)).result()
            ad = pool.submit(_task_amax, (f"{LM}{l}.mlp.down_proj.weight",)).result()
            self.dw_gu = _dw_bytes(max(ag, au))
            self.dw_dn = _dw_bytes(ad)
            in_div_gu = enc.input_divisor(
                np.concatenate([bands["ffn_gate"], bands["ffn_up"]]))
            in_div_dn = enc.input_divisor(bands["ffn_down"])
            gu: list[tuple] = []
            for r0, r1 in _row_chunks(FFN, ROW_TARGET):
                gu.append((pool.submit(_task_nvfp4,
                                       (f"{LM}{l}.mlp.gate_proj.weight", r0, r1,
                                        self.dw_gu, bands["ffn_gate"])), r1 - r0))
            for r0, r1 in _row_chunks(FFN, ROW_TARGET):
                gu.append((pool.submit(_task_nvfp4,
                                       (f"{LM}{l}.mlp.up_proj.weight", r0, r1,
                                        self.dw_gu, bands["ffn_up"])), r1 - r0))
            dn: list[tuple] = []
            for r0, r1 in _row_chunks(HIDDEN, DOWN_ROW_TARGET):
                dn.append((pool.submit(_task_nvfp4,
                                       (f"{LM}{l}.mlp.down_proj.weight", r0, r1,
                                        self.dw_dn, bands["ffn_down"])), r1 - r0))
            self.gu_futs, self.dn_futs = gu, dn
            self.in_div_gu, self.in_div_dn = in_div_gu, in_div_dn
            self.report.update({
                "mlp": "NVFP4",
                "amax_gate": round(ag, 6), "amax_up": round(au, 6),
                "amax_down": round(ad, 6),
                "d_w_gate_up": struct.unpack("<f", self.dw_gu)[0],
                "d_w_down": struct.unpack("<f", self.dw_dn)[0],
                "input_div_gate_up": struct.unpack("<f", in_div_gu)[0],
                "input_div_down": struct.unpack("<f", in_div_dn)[0],
            })
        else:
            gu = []
            for r0, r1 in _row_chunks(FFN, ROW_TARGET):
                gu.append((pool.submit(_task_fp8,
                                       (f"{LM}{l}.mlp.gate_proj.weight", 0, r0, r1,
                                        bands["ffn_gate"])), r1 - r0))
            for r0, r1 in _row_chunks(FFN, ROW_TARGET):
                gu.append((pool.submit(_task_fp8,
                                       (f"{LM}{l}.mlp.up_proj.weight", 0, r0, r1,
                                        bands["ffn_up"])), r1 - r0))
            dn = []
            for r0, r1 in _row_chunks(HIDDEN, DOWN_ROW_TARGET):
                dn.append((pool.submit(_task_fp8,
                                       (f"{LM}{l}.mlp.down_proj.weight", 0, r0, r1,
                                        bands["ffn_down"])), r1 - r0))
            self.gu_futs, self.dn_futs = gu, dn
            self.report["mlp"] = "FP8"

        if l in inv.GDN_LAYERS:
            qkvz = []
            for r0, r1 in _row_chunks(GDN_QK_OUT, ROW_TARGET):
                qkvz.append((pool.submit(_task_fp8,
                                         (names["qkv"], 0, r0, r1,
                                          bands["attn_qkv"])), r1 - r0))
            qkvz.append((pool.submit(_task_fp8,
                                     (names["qkv"], 3, 0, 0,
                                      bands["attn_qkv"])), GDN_Z_OUT))
            qkvz.append((pool.submit(_task_fp8,
                                     (names["z"], 4, 0, 0,
                                      bands["attn_gate"])), GDN_Z_OUT))
            out = []
            for r0, r1 in _row_chunks(HIDDEN, ROW_TARGET):
                out.append((pool.submit(_task_fp8,
                                        (names["out"], 5, r0, r1,
                                         bands["ssm_out"])), r1 - r0))
        else:
            qkvz = [(pool.submit(_task_fp8,
                                 (names["q"], 1, 0, 0,
                                  bands["attn_q"])), FULL_QG_OUT)]
            for r0, r1 in _row_chunks(FULL_KV_OUT, ROW_TARGET):
                qkvz.append((pool.submit(_task_fp8,
                                         (names["k"], 0, r0, r1,
                                          bands["attn_k"])), r1 - r0))
            qkvz.append((pool.submit(_task_fp8,
                                     (names["q"], 2, 0, 0,
                                      bands["attn_q"])), FULL_QG_OUT))
            for r0, r1 in _row_chunks(FULL_KV_OUT, ROW_TARGET):
                qkvz.append((pool.submit(_task_fp8,
                                         (names["v"], 0, r0, r1,
                                          bands["attn_v"])), r1 - r0))
            out = []
            for r0, r1 in _row_chunks(HIDDEN, ROW_TARGET):
                out.append((pool.submit(_task_fp8,
                                        (names["out"], 0, r0, r1,
                                         bands["attn_output"])), r1 - r0))
        self.qkvz_futs, self.out_futs = qkvz, out
        try:
            self.report["imatrix_counts_ffn_gate"] = _imat(
                f"blk.{l}.ffn_gate.weight.counts").item()
        except KeyError:
            pass
        self.ready = True


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def build_plan(layers: tuple[int, ...], smoke: bool) -> list:
    """Specs in exact write order (a subset of the inventory object plan)."""
    if not smoke:
        return list(inv.OBJECT_SPECS)
    names = set(RESOURCE_NAMES)
    for l in layers:
        for s in inv.OBJECT_SPECS:
            if s.name.startswith(f"text/layers/{l}/"):
                names.add(s.name)
    names.add("text/final_norm")
    return [s for s in inv.OBJECT_SPECS if s.name in names]


def _producer_for(name: str):
    """Producer callable for one object name (module-level: pool is global)."""
    if name in RESOURCE_NAMES:
        return lambda name=name: G_RES_MAP[name]
    if name == "text/token_embedding":
        return lambda: _fp8_whole(
            "model.language_model.embed_tokens.weight", (VOCAB, HIDDEN), None)
    if name == "text/final_norm":
        return lambda: _norm_bf16("model.language_model.norm.weight")
    if name == "text/output_head":
        return lambda: _fp8_whole("lm_head.weight", (VOCAB, HIDDEN), None)
    if name.startswith("text/layers/"):
        parts = name.split("/")
        l = int(parts[2])
        role = "/".join(parts[3:])
        st = LAYER_STATES[l]
        if role == "input_norm":
            return lambda l=l: _norm_bf16(f"{LM}{l}.input_layernorm.weight")
        if role == "post_attention_norm":
            return lambda l=l: _norm_bf16(
                f"{LM}{l}.post_attention_layernorm.weight")
        if role == "gdn/a_log":
            return lambda l=l: _a_log(l)
        if role == "gdn/dt_bias":
            return lambda l=l: _dt_bias(l)
        if role == "gdn/convolution":
            return lambda l=l: _convolution(l)
        if role == "gdn/a_b_projection":
            return lambda l=l: _a_b_projection(l)
        if role == "gdn/norm":
            return lambda l=l: _norm_bf16(f"{LM}{l}.linear_attn.norm.weight")
        if role == "gdn/query_key_value_z":
            return lambda st=st: _assemble_fp8(
                (GDN_QKVZ_OUT, HIDDEN), _parts(st.qkvz_futs))
        if role == "gdn/output":
            return lambda st=st: _assemble_fp8(
                (HIDDEN, PROJ_OUT), _parts(st.out_futs))
        if role == "attention/query_key_gate_value":
            return lambda st=st: _assemble_fp8(
                (FULL_QKVZ_OUT, HIDDEN), _parts(st.qkvz_futs))
        if role == "attention/query_norm":
            return lambda l=l: _norm_bf16(f"{LM}{l}.self_attn.q_norm.weight")
        if role == "attention/key_norm":
            return lambda l=l: _norm_bf16(f"{LM}{l}.self_attn.k_norm.weight")
        if role == "attention/output":
            return lambda st=st: _assemble_fp8(
                (HIDDEN, PROJ_OUT), _parts(st.out_futs))
        if l in inv.NVFP4_MLP_LAYERS:
            if role == "mlp/gate_up":
                return lambda st=st: _assemble_nvfp4(
                    (GATE_UP_OUT, HIDDEN), _parts(st.gu_futs), st.dw_gu)
            if role == "mlp/down":
                return lambda st=st: _assemble_nvfp4(
                    (HIDDEN, FFN), _parts(st.dn_futs), st.dw_dn)
            if role == "mlp/gate_up_projection/input_scale_divisor":
                return lambda st=st: _f32_bytes(st.in_div_gu)
            if role == "mlp/down_projection/input_scale_divisor":
                return lambda st=st: _f32_bytes(st.in_div_dn)
        else:
            if role == "mlp/gate_up":
                return lambda st=st: _assemble_fp8(
                    (GATE_UP_OUT, HIDDEN), _parts(st.gu_futs))
            if role == "mlp/down":
                return lambda st=st: _assemble_fp8(
                    (HIDDEN, FFN), _parts(st.dn_futs))
        raise KeyError(f"no producer for object {name!r}")
    # everything else (vision / mtp / draft head) is copied from the base
    # artifact byte-for-byte
    return lambda name=name: bytes(G_OLD.payload(name))


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def _layer_tensor_names(layer: int) -> list[str]:
    base = [f"{LM}{layer}.input_layernorm.weight",
            f"{LM}{layer}.post_attention_layernorm.weight",
            f"{LM}{layer}.mlp.gate_proj.weight",
            f"{LM}{layer}.mlp.up_proj.weight",
            f"{LM}{layer}.mlp.down_proj.weight"]
    if layer in inv.GDN_LAYERS:
        base += [f"{LM}{layer}.linear_attn.in_proj_qkv.weight",
                 f"{LM}{layer}.linear_attn.in_proj_z.weight",
                 f"{LM}{layer}.linear_attn.in_proj_a.weight",
                 f"{LM}{layer}.linear_attn.in_proj_b.weight",
                 f"{LM}{layer}.linear_attn.A_log",
                 f"{LM}{layer}.linear_attn.dt_bias",
                 f"{LM}{layer}.linear_attn.conv1d.weight",
                 f"{LM}{layer}.linear_attn.norm.weight",
                 f"{LM}{layer}.linear_attn.out_proj.weight"]
    else:
        base += [f"{LM}{layer}.self_attn.q_proj.weight",
                 f"{LM}{layer}.self_attn.k_proj.weight",
                 f"{LM}{layer}.self_attn.v_proj.weight",
                 f"{LM}{layer}.self_attn.o_proj.weight",
                 f"{LM}{layer}.self_attn.q_norm.weight",
                 f"{LM}{layer}.self_attn.k_norm.weight"]
    return base


def _preflight(layers: tuple[int, ...], resources_dir: Path) -> dict:
    """Verify the checkpoint config, tensor presence/shapes, imatrix bands,
    base-artifact copy availability, and the pinned resource hashes."""
    # checkpoint config (multimodal: text fields live under text_config, with
    # a top-level fallback for a language-only checkpoint)
    raw = json.loads((resources_dir / "config.json").read_text())
    cfg = raw["text_config"] if isinstance(raw.get("text_config"), dict) else raw
    cfg_checks: dict[str, object] = {
        "config.num_hidden_layers": cfg.get("num_hidden_layers"),
        "config.hidden_size": cfg.get("hidden_size"),
        "config.intermediate_size": cfg.get("intermediate_size"),
        "config.num_attention_heads": cfg.get("num_attention_heads"),
        "config.num_key_value_heads": cfg.get("num_key_value_heads"),
        "config.head_dim": cfg.get("head_dim"),
        "config.full_attention_interval": cfg.get("full_attention_interval"),
        "config.linear_num_key_heads": cfg.get("linear_num_key_heads"),
        "config.linear_num_value_heads": cfg.get("linear_num_value_heads"),
        "config.linear_key_head_dim": cfg.get("linear_key_head_dim"),
        "config.linear_value_head_dim": cfg.get("linear_value_head_dim"),
        "config.mamba_ssm_dtype": cfg.get("mamba_ssm_dtype"),
    }
    want = {
        "config.num_hidden_layers": LAYERS,
        "config.hidden_size": HIDDEN,
        "config.intermediate_size": FFN,
        "config.num_attention_heads": HEADS,
        "config.num_key_value_heads": KV_HEADS,
        "config.head_dim": HEAD_DIM,
        "config.full_attention_interval": 4,
        "config.linear_num_key_heads": 16,
        "config.linear_num_value_heads": GDN_V_HEADS,
        "config.linear_key_head_dim": GDN_HEAD_V,
        "config.linear_value_head_dim": GDN_HEAD_V,
        "config.mamba_ssm_dtype": "float32",
    }
    for k, w in want.items():
        if cfg_checks[k] != w:
            raise ValueError(f"config mismatch: {k} = {cfg_checks[k]!r}, "
                             f"want {w!r}")

    # HF tensor + shape existence for every planned layer
    for l in layers:
        for name in _layer_tensor_names(l):
            if name not in G_HF.wm:
                raise KeyError(f"checkpoint is missing {name!r}")
        for role in _LayerState(l)._band_roles():
            if _band_name(l, role) not in IMAT_IDS:
                raise KeyError(f"imatrix is missing {_band_name(l, role)!r}")
    for name, want_shape in (
        (f"{LM}0.linear_attn.in_proj_qkv.weight", (GDN_QKV_OUT, HIDDEN)),
        (f"{LM}0.linear_attn.in_proj_z.weight", (GDN_Z_OUT, HIDDEN)),
        (f"{LM}0.linear_attn.in_proj_a.weight", (16 * 3, HIDDEN)),
        (f"{LM}0.linear_attn.conv1d.weight", (GDN_QKV_OUT, 1, 4)),
        (f"{LM}0.linear_attn.A_log", (GDN_V_HEADS,)),
        (f"{LM}0.linear_attn.norm.weight", (GDN_HEAD_V,)),
        (f"{LM}0.linear_attn.out_proj.weight", (HIDDEN, PROJ_OUT)),
        (f"{LM}3.self_attn.q_proj.weight", (2 * FULL_QG_OUT, HIDDEN)),
        (f"{LM}3.self_attn.k_proj.weight", (FULL_KV_OUT, HIDDEN)),
        (f"{LM}3.self_attn.v_proj.weight", (FULL_KV_OUT, HIDDEN)),
        (f"{LM}3.self_attn.o_proj.weight", (HIDDEN, PROJ_OUT)),
        (f"{LM}3.self_attn.q_norm.weight", (HEAD_DIM,)),
        (f"{LM}0.mlp.gate_proj.weight", (FFN, HIDDEN)),
        (f"{LM}0.mlp.down_proj.weight", (HIDDEN, FFN)),
        ("model.language_model.embed_tokens.weight", (VOCAB, HIDDEN)),
        ("lm_head.weight", (VOCAB, HIDDEN)),
        ("model.language_model.norm.weight", (HIDDEN,)),
    ):
        if name not in G_HF.wm:
            raise KeyError(f"checkpoint is missing {name!r}")
        got_shape, got_dtype = G_HF.shape(name)
        if got_shape != tuple(want_shape) or got_dtype != "BF16":
            raise ValueError(f"checkpoint tensor {name}: {got_shape} "
                             f"{got_dtype}, want {tuple(want_shape)} BF16")

    # base artifact: every copied object must exist
    for name in COPIED_NAMES:
        G_OLD.find(name)          # KeyError if absent

    # resources: read from the checkpoint directory, sha256-verified against
    # the pinned official hashes
    G_RES_MAP.clear()
    for r in RESOURCE_NAMES:
        data = (resources_dir / r.removeprefix("frontend/")).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        pinned = OFFICIAL_RESOURCE_SHA256[r]
        if sha != pinned:
            raise ValueError(f"resource {r} sha256 {sha[:16]}... != pinned "
                             f"{pinned[:16]}... (not the official bytes)")
        G_RES_MAP[r] = data
    return cfg_checks


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
    global G_HF, G_IMAT, G_OLD, G_POOL, G_RES_MAP, IMAT_IDS, LAYER_STATES
    t0 = time.time()
    resources_dir = Path(args.resources or args.model)
    print(f"[jbnvfp4-27b] opening {args.model} (safetensors) + {args.imatrix} "
          f"(imatrix)", flush=True)
    G_HF = _HFSource(Path(args.model))
    G_IMAT = GGUFReader(args.imatrix)
    G_OLD = Artifact(args.base_artifact)
    IMAT_IDS = {t.name: i for i, t in enumerate(G_IMAT.tensors)}
    layers = _parse_layers(args.layers)
    if args.smoke and not args.layers:
        layers = (0, 3, 55, 56, 63)
    print(f"[jbnvfp4-27b] layers={layers} workers={args.workers} "
          f"smoke={args.smoke}", flush=True)
    checks = _preflight(layers, resources_dir)
    print(f"[jbnvfp4-27b] preflight OK ({len(G_HF.wm)} checkpoint tensors, "
          f"{len(IMAT_IDS)} imatrix tensors)", flush=True)

    pool = ProcessPoolExecutor(max_workers=args.workers)
    G_POOL = pool
    try:
        LAYER_STATES = {l: _LayerState(l) for l in layers}
        specs = build_plan(layers, args.smoke)
        plan = family_conversion.build_object_plan(specs, G_RES_MAP)
        total_bytes = plan.payload_span_bytes
        by_format: dict[str, int] = {}
        for obj in plan.objects:
            fmt = getattr(obj, "format", "resource")
            by_format[fmt] = by_format.get(fmt, 0) + 1
        print(f"[jbnvfp4-27b] plan: {len(specs)} objects, "
              f"{total_bytes / 1e9:.3f} GB payload; "
              + " ".join(f"{k}:{v}" for k, v in sorted(by_format.items())),
              flush=True)
        if args.plan_only:
            print("[jbnvfp4-27b] --plan-only: no artifact written", flush=True)
            return 0

        parity = None
        if args.smoke:
            parity = _smoke_parity(layers)
            if not all(bool(v["match"]) for v in parity.values()):
                raise RuntimeError(f"smoke parity mismatch: {parity}")
            print(f"[jbnvfp4-27b] parity: {parity}", flush=True)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(out_path,
                            ArtifactIdentity(MODEL_ID, WEIGHTS_ID),
                            plan.specs) as writer:
            if writer.objects != plan.objects:
                raise RuntimeError("writer object plan differs from build plan")
            for i, spec in enumerate(specs, start=1):
                name = spec.name
                if name.startswith("text/layers/"):
                    LAYER_STATES[int(name.split("/")[2])].ensure()
                payload = _producer_for(name)()
                writer.write(name, payload)
                print(f"[{i}/{len(specs)}] {name}  t={time.time() - t0:7.1f}s",
                      flush=True)
        elapsed = time.time() - t0
        file_bytes = out_path.stat().st_size
        report = {
            "identity": {"model_id": MODEL_ID, "weights_id": WEIGHTS_ID},
            "inputs": {
                "model": str(args.model),
                "imatrix": str(args.imatrix),
                "base_artifact": str(args.base_artifact),
                "resources": str(resources_dir),
                "out": str(out_path),
            },
            "config_checks": {k: str(v) for k, v in checks.items()},
            "copied_from_base_artifact": {
                "objects": len(COPIED_NAMES),
                "note": "vision tower, MTP module, draft head — outside "
                        "imatrix coverage; byte copies (same source, "
                        "reference quantization)",
            },
            "layers": {str(l): LAYER_STATES[l].report for l in layers},
            "parity": parity,
            "objects": len(specs),
            "payload_bytes": total_bytes,
            "file_bytes": file_bytes,
            "workers": args.workers,
            "smoke": bool(args.smoke),
            "elapsed_s": round(elapsed, 1),
            "encoder": "tools/convert/common/encoder_imatrix.py (pure numpy, "
                       "imatrix-aware)",
            "conventions": {
                "hf_layout": "HF safetensors: linear weights [out, in] C-order "
                             "BF16; GDN in_proj_qkv is q|k|v row order; "
                             "conv1d [C, 1, 4]",
                "gdn_vreorder": "V-head tiling (grouped -> tiled): qkv v rows / "
                                "z rows / conv v channels by VPERM (6144); "
                                "a/b/A_log/dt_bias by AP48 (48); out_proj "
                                "columns by VPERM",
                "gdn_transforms": "a_log = -exp(reordered(A_log)) via torch "
                                  "f32 exp; dt_bias = f32(bf16(reordered(dt))); "
                                  "conv = transposed [4, 10240] (bf16-"
                                  "representable f32) RNE-cast to BF16; a|b = "
                                  "vstack BF16",
                "norms": "layer input/post norms, attn q/k norms, gdn norm, "
                         "final_norm: artifact BF16 = bf16(HF weight) "
                         "directly (no 1+w fold on the HF source)",
                "mlp_nvfp4_layers_0_55": "gate+up share d_w = f32(2688/amax"
                                         "(gate,up)); down its own; "
                                         "input_scale_divisor = f32(2688/max_j "
                                         "sqrt(band[j])) over gate+up bands",
                "mlp_fp8_layers_56_63": "FP8_E4M3FN_ROW_BF16S, per-row BF16 "
                                        "scale, 17-candidate imatrix search",
                "gdn_qkvz": "rows [in_proj_qkv 10240 (q|k|v-tiled) | "
                            "in_proj_z 6144 (tiled)], per-source 1-D bands",
                "full_attn_qkvz": "rows [q 6144 | k 1024 | gate 6144 | v "
                                  "1024]; q/gate de-interleaved per head from "
                                  "q_proj (head h: q = h*512..h*512+255, "
                                  "gate = h*512+256..h*512+511)",
                "token_embedding_output_head": "FP8 reference profile (band "
                                               "= None); no imatrix entry",
                "copied_sections": "vision/MTP/draft-head byte-copied from "
                                   "--base-artifact; resources from "
                                   "--resources sha256-verified against the "
                                   "pinned official hashes",
            },
        }
        report_path = out_path.with_suffix(out_path.suffix + ".conversion.json")
        with open(report_path, "w") as rf:
            rf.write(json.dumps(report, indent=2) + "\n")
            rf.flush()
            os.fsync(rf.fileno())
        print(f"[jbnvfp4-27b] done in {elapsed:.1f}s: "
              f"{file_bytes / 1e9:.3f} GB -> {out_path}", flush=True)
        print(f"[jbnvfp4-27b] report -> {report_path}", flush=True)
        return 0
    finally:
        pool.shutdown(wait=True)
        G_OLD.close()


def _smoke_parity(layers: tuple[int, ...]) -> dict:
    """Parent re-encodes the first task rows of two objects; the worker
    bytes must match exactly (forked readers see identical data)."""
    out: dict[str, object] = {}
    for l in layers:
        LAYER_STATES[l].ensure()
    l_nv = next((l for l in layers if l in inv.NVFP4_MLP_LAYERS), None)
    l_fp = next((l for l in layers if l not in inv.NVFP4_MLP_LAYERS), None)

    if l_nv is not None:
        # NVFP4 gate rows [0:first chunk] of layer l_nv (first task part)
        st = LAYER_STATES[l_nv]
        f, nr = st.gu_futs[0]
        w = _bf16_f32(np.ascontiguousarray(
            G_HF.rows_u16(f"{LM}{l_nv}.mlp.gate_proj.weight", 0, nr)))
        packed, words = enc.encode_nvfp4(
            w * np.frombuffer(st.dw_gu, np.float32),
            _band(l_nv, "ffn_gate"), row_chunk=NV_CHUNK)
        out[f"nvfp4_gate_l{l_nv}"] = {
            "rows": nr,
            "packed_sha256": hashlib.sha256(packed.tobytes()).hexdigest()[:16],
            "worker_packed_sha256": hashlib.sha256(f.result()[0]).hexdigest()[:16],
            "match": (packed.tobytes() == f.result()[0]
                      and words.tobytes() == f.result()[1]),
        }

    if l_fp is not None:
        # FP8 gate rows [0:first chunk] of layer l_fp (first task part)
        st = LAYER_STATES[l_fp]
        f, nr = st.gu_futs[0]
        w = _bf16_f32(np.ascontiguousarray(
            G_HF.rows_u16(f"{LM}{l_fp}.mlp.gate_proj.weight", 0, nr)))
        codes, scales = enc.quantize_fp8_row_scaled(
            w, _band(l_fp, "ffn_gate"), row_chunk=NV_CHUNK)
        out[f"fp8_gate_l{l_fp}"] = {
            "rows": nr,
            "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest()[:16],
            "worker_codes_sha256": hashlib.sha256(f.result()[0]).hexdigest()[:16],
            "match": (codes.tobytes() == f.result()[0]
                      and scales.view(np.uint16).tobytes() == f.result()[1]),
        }
    if not out:
        raise ValueError("smoke parity needs at least one NVFP4 and one FP8 "
                         "mlp layer")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Build the JB-NVFP4 .ninfer artifact for Qwen3.8-27B "
                    "from the HF safetensors checkpoint + unsloth imatrix; "
                    "vision/MTP/draft-head sections are byte-copied from the "
                    "base artifact (outside imatrix coverage).")
    ap.add_argument("--model", required=True,
                    help="HF safetensors checkpoint directory "
                         "(Qwen3.8-27B-BF16)")
    ap.add_argument("--imatrix", required=True,
                    help="unsloth imatrix GGUF")
    ap.add_argument("--base-artifact", required=True,
                    help="existing .ninfer (vision/MTP/draft source)")
    ap.add_argument("--resources", default=None,
                    help="frontend resources directory (default: --model)")
    ap.add_argument("--out", required=True, help="output .ninfer path")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--layers", default=None,
                    help="layer subset, e.g. '0-3' or '0,3,56' (default: all 64)")
    ap.add_argument("--smoke", action="store_true",
                    help="subset run (layers 0,3,55,56,63 + final_norm; parity "
                         "checks)")
    ap.add_argument("--plan-only", action="store_true",
                    help="build and print the object plan; write nothing")
    args = ap.parse_args(argv)
    sys.exit(convert(args))


if __name__ == "__main__":
    main()
