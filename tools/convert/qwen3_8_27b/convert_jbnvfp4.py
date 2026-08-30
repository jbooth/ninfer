"""JB-NVFP4 ``.ninfer`` artifact builder for Qwen3.8-27B (dense qwen35).

Source of truth
---------------
* canonical BF16 GGUF (``Qwen3.8-27B-BF16.gguf``, 54.6 GB, 65 blocks);
* unsloth imatrix file (``imatrix_unsloth_qwen3.8-27B.gguf_file``) — per-tensor
  input second-moment bands (``in_sum2``) and per-tensor token counts;
* the existing ``qwen3.8-27b`` artifact (``qwen3_8_27b_nvfp4.ninfer``): source
  for the imatrix-free sections — the six frontend resources, the vision tower
  (333 objects), the MTP module (12 objects), and the draft head (2 objects),
  which are copied byte-for-byte.

Numerics
--------
The encoder is the shared pure-numpy imatrix-aware selection from
``tools/convert/common/encoder_imatrix.py``: the per-object objective is the
importance-weighted MSE of the quantization error (plan §2.4 — "the MSE of
importance × error").  No llama-quantize / C++ on this path.

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
* Vision tower / MTP / draft head / resources: byte copies from the base
  artifact (same source checkpoint, reference quantization, no imatrix).

Canonical invocation
--------------------
::

    /home/robot/workplace/unsloth_env/bin/python -m \
        tools.convert.qwen3_8_27b.convert_jbnvfp4 \
        --model /llm/models/Qwen3.8-27B-BF16.gguf \
        --imatrix /llm/models/imatrix_unsloth_qwen3.8-27B.gguf_file \
        --base-artifact /llm/models/qwen3_8_27b_nvfp4.ninfer \
        --out /home/robot/jb/Qwen3.8-27B-JB-NVFP4.ninfer \
        --workers 32

Layout conventions
------------------
* GGUF metadata stores linears as ``(in, out)``; the reader returns C-order
  data with the metadata dims reversed, so 2-D linears arrive as
  ``(out, in)`` — already the artifact orientation.  The 2-D conv kernels
  (metadata ``(K=4, C=10240)``) arrive as ``(C, K)`` and are transposed.
* The GGUF stores 1-D norms and the conv kernel as FP32; the artifact's BF16
  objects are the RNE BF16 cast of those values.
* Full-attention source ``attn_q`` interleaves q|gate per head (head h: q =
  rows ``h*512 .. h*512+255``, gate = ``h*512+256 .. h*512+511``); the
  parent ships ``q | k | gate | v`` de-interleaved.
* Per-layer object write order: ``input_norm`` -> attention/GDN ->
  ``post_attention_norm`` -> MLP.
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

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.artifact import layouts
from tools.convert.common import encoder_imatrix as enc
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_8_27b import inventory_nvfp4 as inv
from tools.convert.qwen3_8_27b.convert import OFFICIAL_RESOURCE_SHA256

MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "jbnvfp4"

# --- model geometry (GGUF qwen35 KV; asserted at preflight) ----------------
HIDDEN = 5120
FFN = 17408
LAYERS = 64
HEADS = 24
HEAD_DIM = 256
VOCAB = 248320
GDN_QKV_OUT = HEADS * 4 * HEAD_DIM        # 10240: q|k|v
GDN_Z_OUT = HEADS * 2 * HEAD_DIM          # 6144:  z
GDN_QKVZ_OUT = GDN_QKV_OUT + GDN_Z_OUT    # 16384
FULL_QG_OUT = HEADS * HEAD_DIM            # 6144 rows of q or of gate
FULL_KV_OUT = HEADS * HEAD_DIM            # 1024
FULL_QKVZ_OUT = 2 * FULL_QG_OUT + 2 * FULL_KV_OUT  # 14336
PROJ_OUT = HEADS * HEAD_DIM               # 6144: gdn output / attn output in-dim
GATE_UP_OUT = 2 * FFN                     # 34816

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
FP8 = "FP8_E4M3FN_ROW_BF16S"

# q|gate de-interleave row indexes on the source attn_q output axis.
_Q_IDX = np.repeat(np.arange(HEADS) * 2 * HEAD_DIM, HEAD_DIM).astype(np.intp)
_G_IDX = _Q_IDX + HEAD_DIM

NV_CHUNK = 256          # encoder band-search row chunk (worker RAM bound)
ROW_TARGET = 2048       # target rows per pool task
DOWN_ROW_TARGET = 1024  # wider K (17408) -> smaller tasks

G_SRC: GGUFReader
G_IMAT: GGUFReader
G_OLD: Artifact
G_POOL: ProcessPoolExecutor | None = None
SRC_IDS: dict[str, int]
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

def _bf16_f32(raw_u8: np.ndarray) -> np.ndarray:
    """Bit-exact BF16 -> FP32 (bf16 word << 16 as f32 bits)."""
    u16 = raw_u8.view(np.uint16)
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _src(name: str) -> np.ndarray:
    """Source GGUF tensor as a raw memmap view (uint8 for BF16, f32 as-is)."""
    return G_SRC.get_tensor(SRC_IDS[name]).data


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


def _orient(where: str, data: np.ndarray, want: tuple) -> np.ndarray:
    """Return data in artifact orientation (see module docstring)."""
    if tuple(data.shape) == tuple(want):
        return data
    if data.ndim >= 2 and tuple(data.shape[::-1]) == tuple(want):
        return np.ascontiguousarray(data.T)
    raise ValueError(f"cannot orient {where}: {tuple(data.shape)}, want {tuple(want)}")


# ---------------------------------------------------------------------------
# worker tasks (module level so they fork-inherit the readers)
# ---------------------------------------------------------------------------

def _task_amax(args: tuple[str,]) -> float:
    name, = args
    raw = np.ascontiguousarray(_src(name))
    return float(np.abs(_bf16_f32(raw)).max())


def _task_nvfp4(args: tuple[str, int, int, bytes, np.ndarray]) -> tuple[bytes, bytes]:
    """One row range of one source matrix, NVFP4 t-space encode."""
    name, r0, r1, dw, band = args
    w = _bf16_f32(np.ascontiguousarray(_src(name).view(np.uint16)[r0:r1]))
    t = w * np.frombuffer(dw, np.float32)
    packed, words = enc.encode_nvfp4(t, band, row_chunk=NV_CHUNK)
    return packed.tobytes(), words.tobytes()


def _task_fp8(args: tuple[str, int, int, int, np.ndarray]) -> tuple[bytes, bytes]:
    """One row range of one source matrix, imatrix-aware FP8 encode.

    mode 0: rows [r0:r1] of the source; mode 1: de-interleaved q rows of a
    full-attention attn_q (band = the attn_q band); mode 2: the gate rows.
    """
    name, mode, r0, r1, band = args
    u16 = _src(name).view(np.uint16)
    if mode == 0:
        blk = np.ascontiguousarray(u16[r0:r1])
    elif mode == 1:
        blk = np.ascontiguousarray(u16[_Q_IDX])
    elif mode == 2:
        blk = np.ascontiguousarray(u16[_G_IDX])
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


def _f32_bytes(b: bytes) -> bytes:
    return layouts.encode_direct(
        torch.from_numpy(np.frombuffer(b, np.float32)), FP32)


def _direct(name: str, want: tuple, fmt: str) -> bytes:
    """Small GGUF tensor -> direct-format artifact object.

    FP32 sources become the RNE BF16 cast for BF16 objects (the registered
    transform for the direct BF16 format).  ``_orient`` handles the conv
    kernel's transposed arrival; everything else arrives in artifact
    orientation.
    """
    d = np.asarray(_src(name), dtype=np.float32)
    d = np.ascontiguousarray(_orient(name, d, want))
    t = torch.from_numpy(d)
    if fmt == BF16:
        t = t.to(torch.bfloat16)
    return layouts.encode_direct(t, fmt)


def _a_b_projection(layer: int) -> bytes:
    """gdn/a_b_projection = [ssm_alpha (48 rows) | ssm_beta (48 rows)]."""
    a = _bf16_f32(np.ascontiguousarray(_src(f"blk.{layer}.ssm_alpha.weight").view(np.uint16)))
    b = _bf16_f32(np.ascontiguousarray(_src(f"blk.{layer}.ssm_beta.weight").view(np.uint16)))
    t = torch.from_numpy(np.ascontiguousarray(np.vstack((a, b)))).to(torch.bfloat16)
    return layouts.encode_direct(t, BF16)


def _fp8_whole(name: str, shape: tuple[int, int],
               band: np.ndarray | None) -> bytes:
    """One whole (out, in) source matrix as FP8, row-chunked tasks."""
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

    def ensure(self) -> None:
        if self.ready:
            return
        l = self.layer
        pool = G_POOL
        bands = {role: _band(l, role) for role in self._band_roles()}

        if l in inv.NVFP4_MLP_LAYERS:
            ag = pool.submit(_task_amax, (f"blk.{l}.ffn_gate.weight",)).result()
            au = pool.submit(_task_amax, (f"blk.{l}.ffn_up.weight",)).result()
            ad = pool.submit(_task_amax, (f"blk.{l}.ffn_down.weight",)).result()
            self.dw_gu = _dw_bytes(max(ag, au))
            self.dw_dn = _dw_bytes(ad)
            in_div_gu = enc.input_divisor(
                np.concatenate([bands["ffn_gate"], bands["ffn_up"]]))
            in_div_dn = enc.input_divisor(bands["ffn_down"])
            gu: list[tuple] = []
            for r0, r1 in _row_chunks(FFN, ROW_TARGET):
                gu.append((pool.submit(_task_nvfp4,
                                       (f"blk.{l}.ffn_gate.weight", r0, r1,
                                        self.dw_gu, bands["ffn_gate"])), r1 - r0))
            for r0, r1 in _row_chunks(FFN, ROW_TARGET):
                gu.append((pool.submit(_task_nvfp4,
                                       (f"blk.{l}.ffn_up.weight", r0, r1,
                                        self.dw_gu, bands["ffn_up"])), r1 - r0))
            dn: list[tuple] = []
            for r0, r1 in _row_chunks(FFN, DOWN_ROW_TARGET):
                dn.append((pool.submit(_task_nvfp4,
                                       (f"blk.{l}.ffn_down.weight", r0, r1,
                                        self.dw_dn, bands["ffn_down"])), r1 - r0))
            self.gu_futs, self.dn_futs = gu, dn
            self.in_div_gu, self.in_div_dn = in_div_gu, in_div_dn
            self.report.update({
                "mlp": "NVFP4",
                "amax_gate": round(ag, 6), "amax_up": round(au, 6),
                "amax_down": round(ad, 6),
                "d_w_gate_up": float(np.frombuffer(self.dw_gu, np.float32)),
                "d_w_down": float(np.frombuffer(self.dw_dn, np.float32)),
                "input_div_gate_up": float(np.frombuffer(in_div_gu, np.float32)),
                "input_div_down": float(np.frombuffer(in_div_dn, np.float32)),
            })
        else:
            gu = []
            for r0, r1 in _row_chunks(FFN, ROW_TARGET):
                gu.append((pool.submit(_task_fp8,
                                       (f"blk.{l}.ffn_gate.weight", 0, r0, r1,
                                        bands["ffn_gate"])), r1 - r0))
            for r0, r1 in _row_chunks(FFN, ROW_TARGET):
                gu.append((pool.submit(_task_fp8,
                                       (f"blk.{l}.ffn_up.weight", 0, r0, r1,
                                        bands["ffn_up"])), r1 - r0))
            dn = []
            for r0, r1 in _row_chunks(FFN, DOWN_ROW_TARGET):
                dn.append((pool.submit(_task_fp8,
                                       (f"blk.{l}.ffn_down.weight", 0, r0, r1,
                                        bands["ffn_down"])), r1 - r0))
            self.gu_futs, self.dn_futs = gu, dn
            self.report["mlp"] = "FP8"

        if l in inv.GDN_LAYERS:
            qkvz = []
            for r0, r1 in _row_chunks(GDN_QKV_OUT, ROW_TARGET):
                qkvz.append((pool.submit(_task_fp8,
                                         (f"blk.{l}.attn_qkv.weight", 0, r0, r1,
                                          bands["attn_qkv"])), r1 - r0))
            for r0, r1 in _row_chunks(GDN_Z_OUT, ROW_TARGET):
                qkvz.append((pool.submit(_task_fp8,
                                         (f"blk.{l}.attn_gate.weight", 0, r0, r1,
                                          bands["attn_gate"])), r1 - r0))
            out = []
            for r0, r1 in _row_chunks(HIDDEN, ROW_TARGET):
                out.append((pool.submit(_task_fp8,
                                        (f"blk.{l}.ssm_out.weight", 0, r0, r1,
                                         bands["ssm_out"])), r1 - r0))
        else:
            qkvz = [(pool.submit(_task_fp8,
                                 (f"blk.{l}.attn_q.weight", 1, 0, 0,
                                  bands["attn_q"])), FULL_QG_OUT)]
            for r0, r1 in _row_chunks(FULL_KV_OUT, ROW_TARGET):
                qkvz.append((pool.submit(_task_fp8,
                                         (f"blk.{l}.attn_k.weight", 0, r0, r1,
                                          bands["attn_k"])), r1 - r0))
            qkvz.append((pool.submit(_task_fp8,
                                     (f"blk.{l}.attn_q.weight", 2, 0, 0,
                                      bands["attn_q"])), FULL_QG_OUT))
            for r0, r1 in _row_chunks(FULL_KV_OUT, ROW_TARGET):
                qkvz.append((pool.submit(_task_fp8,
                                         (f"blk.{l}.attn_v.weight", 0, r0, r1,
                                          bands["attn_v"])), r1 - r0))
            out = []
            for r0, r1 in _row_chunks(HIDDEN, ROW_TARGET):
                out.append((pool.submit(_task_fp8,
                                        (f"blk.{l}.attn_output.weight", 0, r0, r1,
                                         bands["attn_output"])), r1 - r0))
        self.qkvz_futs, self.out_futs = qkvz, out
        try:
            self.report["imatrix_counts_ffn_gate"] = float(
                _imat(f"blk.{l}.ffn_gate.weight.counts"))
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
        return lambda name=name: bytes(G_OLD.payload(name))
    if name == "text/token_embedding":
        return lambda: _fp8_whole("token_embd.weight", (VOCAB, HIDDEN), None)
    if name == "text/final_norm":
        return lambda: _direct("output_norm.weight", (HIDDEN,), BF16)
    if name == "text/output_head":
        return lambda: _fp8_whole("output.weight", (VOCAB, HIDDEN), None)
    if name.startswith("text/layers/"):
        parts = name.split("/")
        l = int(parts[2])
        role = "/".join(parts[3:])
        st = LAYER_STATES[l]
        if role == "input_norm":
            return lambda l=l: _direct(f"blk.{l}.attn_norm.weight", (HIDDEN,), BF16)
        if role == "post_attention_norm":
            return lambda l=l: _direct(f"blk.{l}.post_attention_norm.weight",
                                       (HIDDEN,), BF16)
        if role == "gdn/a_log":
            return lambda l=l: _direct(f"blk.{l}.ssm_a", (48,), FP32)
        if role == "gdn/dt_bias":
            return lambda l=l: _direct(f"blk.{l}.ssm_dt.bias", (48,), FP32)
        if role == "gdn/convolution":
            return lambda l=l: _direct(f"blk.{l}.ssm_conv1d.weight",
                                       (4, GDN_QKV_OUT), BF16)
        if role == "gdn/a_b_projection":
            return lambda l=l: _a_b_projection(l)
        if role == "gdn/norm":
            return lambda l=l: _direct(f"blk.{l}.ssm_norm.weight", (128,), BF16)
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
            return lambda l=l: _direct(f"blk.{l}.attn_q_norm.weight",
                                       (HEAD_DIM,), BF16)
        if role == "attention/key_norm":
            return lambda l=l: _direct(f"blk.{l}.attn_k_norm.weight",
                                       (HEAD_DIM,), BF16)
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

KV_CHECKS = (
    ("general.architecture", "qwen35"),
    ("qwen35.block_count", 65),
    ("qwen35.embedding_length", HIDDEN),
    ("qwen35.feed_forward_length", FFN),
    ("qwen35.attention.head_count", HEADS),
    ("qwen35.attention.head_count_kv", HEADS // 6),
    ("qwen35.nextn_predict_layers", 1),
)


def _preflight(layers: tuple[int, ...]) -> dict:
    kv: dict = {}
    for name, want in KV_CHECKS:
        got = G_SRC.get_field(name).contents()
        kv[name] = got
        if got != want:
            raise ValueError(f"GGUF KV {name} = {got!r}, want {want!r}")

    for name in ("token_embd.weight", "output.weight", "output_norm.weight"):
        if name not in SRC_IDS:
            raise KeyError(f"source GGUF is missing {name!r}")
    for l in layers:
        need_src = [f"blk.{l}.attn_norm.weight", f"blk.{l}.post_attention_norm.weight",
                    f"blk.{l}.ffn_gate.weight", f"blk.{l}.ffn_up.weight",
                    f"blk.{l}.ffn_down.weight"]
        if l in inv.GDN_LAYERS:
            need_src += [f"blk.{l}.attn_qkv.weight", f"blk.{l}.attn_gate.weight",
                         f"blk.{l}.ssm_out.weight", f"blk.{l}.ssm_a",
                         f"blk.{l}.ssm_dt.bias", f"blk.{l}.ssm_conv1d.weight",
                         f"blk.{l}.ssm_norm.weight", f"blk.{l}.ssm_alpha.weight",
                         f"blk.{l}.ssm_beta.weight"]
        else:
            need_src += [f"blk.{l}.attn_q.weight", f"blk.{l}.attn_k.weight",
                         f"blk.{l}.attn_v.weight", f"blk.{l}.attn_q_norm.weight",
                         f"blk.{l}.attn_k_norm.weight", f"blk.{l}.attn_output.weight"]
        for name in need_src:
            if name not in SRC_IDS:
                raise KeyError(f"source GGUF is missing {name!r}")
        for role in _LayerState(l)._band_roles():
            if _band_name(l, role) not in IMAT_IDS:
                raise KeyError(f"imatrix is missing {_band_name(l, role)!r}")

    for name in COPIED_NAMES:
        G_OLD.find(name)          # KeyError if absent
    for name in RESOURCE_NAMES:
        data = bytes(G_OLD.payload(name))
        sha = hashlib.sha256(data).hexdigest()
        want = OFFICIAL_RESOURCE_SHA256[name.removeprefix("frontend/")]
        if sha != want:
            raise ValueError(f"base artifact resource {name} sha256 {sha[:16]}... "
                             f"!= pinned {want[:16]}...")
    return kv


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
    global G_SRC, G_IMAT, G_OLD, G_POOL, SRC_IDS, IMAT_IDS, LAYER_STATES
    t0 = time.time()
    print(f"[jbnvfp4-27b] opening {args.model}", flush=True)
    G_SRC = GGUFReader(args.model)
    G_IMAT = GGUFReader(args.imatrix)
    G_OLD = Artifact(args.base_artifact)
    SRC_IDS = {t.name: i for i, t in enumerate(G_SRC.tensors)}
    IMAT_IDS = {t.name: i for i, t in enumerate(G_IMAT.tensors)}
    layers = _parse_layers(args.layers)
    if args.smoke and not args.layers:
        layers = (0, 3, 55, 56, 63)
    print(f"[jbnvfp4-27b] layers={layers} workers={args.workers} smoke={args.smoke}",
          flush=True)
    kv = _preflight(layers)
    print(f"[jbnvfp4-27b] preflight OK ({len(SRC_IDS)} source tensors, "
          f"{len(IMAT_IDS)} imatrix tensors)", flush=True)

    pool = ProcessPoolExecutor(max_workers=args.workers)
    G_POOL = pool
    try:
        LAYER_STATES = {l: _LayerState(l) for l in layers}
        specs = build_plan(layers, args.smoke)
        resource_map = {name: bytes(G_OLD.payload(name)) for name in RESOURCE_NAMES}
        plan = family_conversion.build_object_plan(specs, resource_map)
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
                "out": str(out_path),
            },
            "kv_checks": {k: str(v) for k, v in kv.items()},
            "copied_from_base_artifact": {
                "objects": len(COPIED_NAMES),
                "note": "vision tower, MTP module, draft head — outside imatrix "
                        "coverage; byte copies (same source, reference "
                        "quantization)",
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
                "gguf_layout": "source metadata (in,out); reader data (out,in) "
                               "C-order; 1-D norms/conv are FP32 in the source, "
                               "BF16 objects are the RNE cast",
                "mlp_nvfp4_layers_0_55": "gate+up share d_w = f32(2688/amax"
                                         "(gate,up)); down its own; "
                                         "input_scale_divisor = f32(2688/max_j "
                                         "sqrt(band[j])) over gate+up bands",
                "mlp_fp8_layers_56_63": "FP8_E4M3FN_ROW_BF16S, per-row BF16 "
                                        "scale, 17-candidate imatrix search",
                "gdn_qkvz": "rows [attn_qkv 10240 (q|k|v) | attn_gate 6144 "
                            "(z)], per-source 1-D bands",
                "full_attn_qkvz": "rows [q 6144 | k 1024 | gate 6144 | v 1024]; "
                                  "q/gate de-interleaved per head from attn_q "
                                  "(head h: q = h*512..h*512+255, gate = "
                                  "h*512+256..h*512+511)",
                "token_embedding_output_head": "FP8 reference profile (band "
                                               "= None); no imatrix entry",
                "copied_sections": "resources (6) sha256-verified against the "
                                   "pinned official hashes",
            },
        }
        report_path = out_path.with_suffix(out_path.suffix + ".conversion.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[jbnvfp4-27b] done in {elapsed:.1f}s: "
              f"{file_bytes / 1e9:.3f} GB -> {out_path}", flush=True)
        print(f"[jbnvfp4-27b] report -> {report_path}", flush=True)
        return 0
    finally:
        pool.shutdown(wait=True)
        G_OLD.close()


def _smoke_parity(layers: tuple[int, ...]) -> dict:
    """Parent re-encodes the first task rows of three objects; the worker
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
            _src(f"blk.{l_nv}.ffn_gate.weight").view(np.uint16)[:nr]))
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
            _src(f"blk.{l_fp}.ffn_gate.weight").view(np.uint16)[:nr]))
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
        raise ValueError("smoke parity needs at least one NVFP4 and one FP8 mlp layer")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Build the JB-NVFP4 .ninfer artifact for Qwen3.8-27B from "
                    "the canonical BF16 GGUF + unsloth imatrix; vision/MTP/"
                    "draft-head sections are byte-copied from the base "
                    "artifact (outside imatrix coverage).")
    ap.add_argument("--model", required=True,
                    help="BF16 GGUF (54.6 GB)")
    ap.add_argument("--imatrix", required=True,
                    help="unsloth imatrix GGUF")
    ap.add_argument("--base-artifact", required=True,
                    help="existing .ninfer (resource/vision/MTP/draft source)")
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
