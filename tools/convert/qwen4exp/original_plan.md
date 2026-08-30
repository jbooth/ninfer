# stage1 — Qwen3.8-Flash-Next (`qwen4exp`) JB-NVFP4 `.ninfer` build

Working scratch doc for the v1 deliverable: an imatrix-aware encoder +
converter that produces the `.ninfer` artifact for Qwen3.8-Flash-Next.
Governing spec: `nvfp.md` (§2 encoder, §3 builder, §4.2/§4.6 reference case).
Declared in 2026-08-30 session; owner-directed.

## 1. Inputs / outputs

| item | path |
|---|---|
| BF16 source GGUF (canonical input, 329.7 GiB, 1224 tensors, GGUFv3) | `/llm/models/Qwen3.8-Flash-Next-BF16.gguf` |
| imatrix GGUF (580 MB, 1852 tensors: 926 `.in_sum2` F32 + 926 `.counts` F32) | `/llm/models/imatrix_unsloth_qwen38next.gguf_file` |
| UD reference (quality cross-check only) | `/llm/models/Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf` |
| HF checkpoint (6 frontend resources + config; NOT the weight input) | `/llm/models/Qwen3.8-Flash-Next/master/` |
| 27B NVFP4 artifact (convention oracle) | `/llm/models/qwen3_8_27b_nvfp4.ninfer` |
| Output artifact | `/llm/models/Qwen3.8-Flash-Next-JB-NVFP4.ninfer` (~186.5 GB) |
| Python (numpy 2.5.2, torch 2.11+cu130, gguf 0.19.0) | `/home/robot/workplace/unsloth_env/bin/python` |
| Converter home (new package) | `tools/convert/qwen4exp/` in `/home/jay/workplace/ninfer-dev` |

Hardware: 32 CPUs, 121 GB RAM (103 avail), 697 GB free on `/`. No GPU used.

## 2. Source-inventory facts (measured)

GGUF type 30 = BF16. All 3D expert tensors: shape `[K, N, E]` (ne0=K=input
dim, ne1=N=output dim, ne2=E=512 experts). Per-expert matrix `W[n,k] =
tensor[k, n, e]` (row-major `[N,K]`), e-major flattening for banks.
**Reader data orientation (measured, session 5):** `reader
.get_tensor(i).data` is C-order with metadata dims reversed — linears
`(in, out)` → data `(out, in)`; experts `(in, out, E)` → data
`(E, out, in)`; conv kernels `(K, C)` → data `(C, K)`. The driver's
`_orient()` handles the only case that needs a transpose (conv
kernels); see §8 session-5 entry.

| GGUF tensor (count) | GGUF shape `[K,N(,E)]` | .ninfer object `[N,K]` | format |
|---|---|---|---|
| `ffn_gate_exps`+`ffn_up_exps` (48 each) | `[2560,640,512]` | `text/layers/{l}/moe/routed_gate_up` `[655360,2560]` (e-major: gate rows then up rows per expert) | NVFP4 |
| `ffn_down_exps` (48) | `[640,2560,512]` | `moe/routed_down` `[1310720,640]` | Q6G64_F16S |
| `ffn_gate_inp`+`ffn_gate_inp_shexp` (48) | `[2560,512]`+`[2560,1]` | `moe/router_gate` `[513,2560]` (router 512 rows, shared-gate row 512) | FP32 |
| `ffn_gate_shexp`+`ffn_up_shexp` (48) | `[2560,640]` | `moe/shared_gate_up` `[1280,2560]` | W8G32_F16S |
| `ffn_down_shexp` (48) | `[640,2560]` | `moe/shared_down` `[2560,640]` | W8G32_F16S |
| `attn_qkv` (36 GDN) | `[2560,10240]` = q 2048\|k 2048\|v 6144 | `gdn/query_key` `[4096,2560]` + `gdn/value_z` v-part | W8G32_F16S |
| `attn_gate` (36 GDN) | `[2560,6144]` (z) | `gdn/value_z` z-part `[6144,2560]` (fused v\|z = `[12288,2560]`) | W8G32_F16S |
| `attn_q` (12 full) | `[2560,12288]` = per head q 256 \| gate 256 (24 heads, interleaved) | `attention/query_key_gate_value` q+gate parts | W8G32_F16S |
| `attn_k`,`attn_v` (12 each) | `[2560,512]` | same parent k/v parts | W8G32_F16S |
| | parent = `[13312,2560]` = q 6144 \| k 512 \| gate 6144 \| v 512 | | |
| `ssm_out` (36 GDN) | `[6144,2560]` | `gdn/output` `[2560,6144]` | W8G32_F16S |
| `attn_output` (12 full) | `[6144,2560]` | `attention/output` `[2560,6144]` | W8G32_F16S |
| `ssm_alpha`,`ssm_beta` (36) | `[2560,48]` | `gdn/a_b_projection` `[96,2560]` (alpha rows 0..47, beta 48..95) | BF16 |
| `ssm_a` (36) | `[48]` F32 | `gdn/a_log` `[48]` | FP32 |
| `ssm_dt.bias` (36) | `[48]` F32 | `gdn/dt_bias` `[48]` | FP32 |
| `ssm_conv1d` (36) | `[4,10240]` F32 | `gdn/convolution` `[4,10240]` (tap-major, 35b precedent) | FP32 |
| `ssm_norm` (36) | `[128]` F32 | `gdn/norm` | FP32 |
| `hc_attn_{up,down,inject}` (48) | `[320,10240]`,`[10240,320]`,`[10240,4]` | `text/layers/{l}/hc_attn_{up:[10240,320],down:[320,10240],inject:[10240,4]}` | BF16 |
| `hc_ffn_{up,down,inject}` (48) | same | `hc_ffn_*` same shapes | BF16 |
| `hc_{attn,ffn}_norm` (48) | `[10240]` F32 | `hc_{attn,ffn}_norm` | FP32 |
| `output_hc_{up,down,inject,norm}` (1) | as above | `text/output_hc_*` | BF16/FP32 |
| `indexer.q_proj` (12) | `[2560,512]` | `text/layers/{l}/indexer/query_proj` `[512,2560]` | BF16 |
| `indexer.k_proj` (12) | `[2560,128]` | `indexer/key_proj` `[128,2560]` | BF16 |
| `indexer.{q,k}_norm` (12) | `[128]` F32 | `indexer/{query,key}_norm` | FP32 |
| `output` (1) | `[2560,248320]` | `text/output_head` `[248320,2560]` | BF16 |
| `token_embd` (1) | `[2560,248320]` | `text/token_embedding` `[248320,2560]` | BF16 |
| `per_layer_token_embd` (1) | `[160,320001536]` | `text/per_layer_token_embedding` `[320001536,160]` | BF16 (102.4 GB; chunked copy) |
| `ple_key` (1) | `[2560,10240]` | `text/ple/key` `[10240,2560]` | BF16 |
| `ple_value` (1) | `[2560,2560]` | `text/ple/value` | BF16 |
| `ple_conv1d` (1) | `[4,10240]` | `text/ple/convolution` `[4,10240]` (tap-major) | BF16 |
| `ple_norm_{query,key,conv}` (1) | `[10240]` F32 | `text/ple/norm_{query,key,conv}` | FP32 |

Full-attn layers = `l % 4 == 3` → {3,7,…,47} (12); GDN = the other 36.
MoE (48×): 512 experts, top-10, inter 640, hidden 2560.
No `input_norm`/`post_attention_norm` objects: HC mixer replaces them
(`hc_attn_norm` before attention/GDN sublayer, `hc_ffn_norm` before FFN;
no final norm — HC head mixer is the output norm).

Object totals: 48×(8 MoE + 8 HC + 6 GDN or full-attn set) + globals
≈ 950 tensors + 48 input-divisor FP32[1] objects + 6 resources.
Size target ≈ 186.5 GB (plan §4.3): NVFP4 45.3 + Q6 31.5 + W8 2.84
+ BF16 103.7+1.27+1.28+0.03 + F32 0.25+rest.

## 3. Imatrix layout (measured)

Per weight tensor `<name>.weight`: `<name>.weight.in_sum2` + `.counts`,
F32. Bands keyed by **GGUF tensor name**.
- 2D weights: band `[K]` (Σ over tokens of x_t[j]², j = input column).
- 3D `*_exps` weights: band `[K, E]` — per expert e: column e; `counts [1,E]`.
  gate/up bands `[2560,512]`, down band `[640,512]` ✓ matches §K dims.
- Roles covered: every `attn_*`, `ffn_*_exps`, `ffn_*_shexp`, `hc_*`,
  `indexer.*_proj`, `ssm_out`, `ssm_alpha/beta`, `ple_key/value`, `output`.
  No bands for norms/`ssm_a`/`ssm_dt`/`ssm_conv1d`/routers/embeddings —
  those are T3/BF16 copy rows anyway.
- All 926 bands fit in RAM (~0.5 GB) → load once at start.

## 4. Numeric contracts (final, implemented in encoder)

### 4.1 NVFP4 (routed gate/up) — `W[n,k] = e2m1(c)·e4m3fn(s)/d_w`

**Convention settled (27B artifact saturation argument — IMPLEMENTED & TESTED):**
- Stored scale *word* max = **126 (0x7E)** = value **448** (NOT 448 as word —
  0x7F is NaN/invalid; 448 is the max finite E4M3FN *value*). Max E2M1
  magnitude = 6 ⇒ max block value in t-space = 6·448 = **2688**.
- **`d_w = f32(2688.0/A)`**, `A = max|W|` over the whole object matrix (f32),
  little-endian 4 B after the scale plane (`blockscale-k16-m128x4-v1` via
  `encode_nvfp4`). Proven by saturation: the 27B artifact's stored scale
  values reach 448 (word 126); the reference scale of a saturated block is
  RNE_e4m3(a16/6), so a stored value 448 requires a16 = 2688, i.e.
  A·d_w = 2688. (The raw d_w values O(10³) alone do NOT discriminate 448/A
  from 2688/A — both give plausible A; the saturation observation is the
  discriminator. The earlier "sane amax magnitudes" argument was
  non-discriminating and is retracted.)
- Per 16-sub-block `a16 = amax|t|`, t = W·d_w:
  - **Reference** (band=None, byte-exact vs ggml `quantize_row_nvfp4_ref`;
    SELF-TEST PASSING): `s_ref_word = ggml_fp32_to_ue4m3(f32(a16/6.0f))`
    (exact bit-algorithm in §4.4; ggml divides in f32 — single rounding);
    codes = `best_index_mxfp4(x, d)` — kvalues table, **strict-< ⇒ ties to
    lowest index = lowest |value|**; negative values rounding to zero emit
    code **0** (indices 0 and 8 both hold 0; strict-< keeps index 0).
    a16==0 → word 0, all codes 0.
  - **Imatrix search** (IMPLEMENTED): candidates = ascending E4M3FN words
    from the reference word up to the largest word with value ≤ a16
    (≈16–24 grid points; fixed bound K=24 suffices: the value range spans
    a factor of 6 ≈ 2.6 octaves × 8 values); pick `argmin Σ_j
    band[j]·(t_j − snap_j(s))²` with snap = e2m1(t/s)·s; strict-< (numpy
    argmin first-wins) ⇒ reference wins ties. Only s ≥ s_ref is searched
    (smaller s overflows the E2M1 grid at the block max).
  - **Fallback** (band missing/all-zero): reference, bit-exact.
- Nibble packing (ggml and ninfer agree): `qs[j] = code[2j] |
  (code[2j+1]<<4)` (low nibble = even index; `_pack_low_nibbles`
  layouts.py:594; ggml ref:375-379).
- **Input divisor** (per projection site, FP32[1] object): engine computes
  `alpha = 1/(input_scale_divisor·weight_scale_divisor)`
  (`src/ops/linear/nvfp4/nvfp4_w4a4.cu:31`); 27B stores true divisors
  (values O(10²)). qwen4exp (DECIDED, implemented in the encoder): the
  imatrix band `band[j] = Σ_t x[t,j]²` ⇒ `R = max_j sqrt(band[j])` is a
  STRICT upper bound on |x[t,j]| for every calibration token and column.
  **`in_div = f32(2688.0/R)`** — mirrors the weight divisor; conservative
  (never saturates on calibration data). Object name: `text/layers/{l}/moe/
  gate_up_input_divisor` (plan §4.6 wording).
- E4M3FN grid: exp bias 7; word w<0x7F finite; value = 2^(e-7)·(1+m/8)
  (e=0 → 2^-9·m); 0x7E=448; 0x7F=NaN. `gguf.NVFP4` (quants.py:707) is the
  vectorized oracle (×2 kvalues `0,1,2,3,4,6,8,12` + `ue4m3_to_fp32 =
  raw*0.5` — net = standard E2M1×E4M3, same as the ninfer runtime codec);
  `fp32_to_ue4m3` verified bit-for-bit vs ggml in self-test 1 (200k random
  + 17 edges + scalar port).

### 4.2 Int-group Q6/W8 (routed down / projections / shared) — row-split

Reference = `tools/convert/common/quantize.py` MAXABS_F16_RECIP_RNE_V1,
ported to pure numpy (byte-exact):

```
grouped [N, G, gs] f32 (gs=64 Q6 / gs=32 W8; K pads 0 to %gs)
max_abs = amax per group (f32)
raw   = f32(f64(max_abs) / f64(qmax))          # qmax 31 (Q6) / 127 (W8)
scale = f16(raw)                               # RNE binary16
if scale==0 and max_abs>0: scale = 2**-24      # min subnormal f16
recip = f32(f64(1.0)/f64(scale)) if scale>0 else 0
codes = clamp(rne(round(grouped * recip)), qmin, qmax)   # qmin -32/-127
```

Imatrix-aware search: candidates `s_i = RNE_f16(s_ref · 2**(-i/4))`,
i = 0..16 (i=0 = reference; smaller = finer scale with peak clipping);
per candidate recompute `recip` exactly as above; error
`Σ_j band[j]·(x_j − c_j·s_i)²`; strict-< argmin → reference on ties.
Same fallback (no band → reference bytes exactly).

### 4.3 Direct formats

BF16: `encode_direct` pass-through (bf16→bf16 byte copy; GGUF bf16 words
are the source of truth). F32: same. No search. `per_layer_token_embd`
(102.4 GB) streamed in chunks (e.g. 4 M rows × 160 cols = 1.28 GB).

### 4.4 Reference bit-exact algorithms (captured from llama.cpp)

`ggml_fp32_to_ue4m3(x)` (ggml-impl.h:517): x≤0→0; x>448→clamp 448;
bits = f32 bits; e32 = exp−127, m3 = top 3 mantissa bits ((bits>>20)&7);
ue = e32+7; if ue≤0: man = (int)(x·512+0.5) clamped ≤7, man<1→0 else man;
if ue≥15 → 0x7E; else round_bit=(bits>>19)&1, man = m3 + round_bit,
if >7 {man=0; ue++}; if ue≥15 → 0x7E; return **(ue<<3)|man** (3-bit
mantissa — confirmed ggml-impl.h:556). The normal path is "top 3 bits +
4th bit as round bit" (NOT full RNE over discarded bits); the numpy port
keeps this quirk. Self-test 1 verifies bit parity vs the `gguf` package
oracle (200k random + 17 edges) and vs the scalar port.

`best_index_mxfp4(x, e)` (ggml-quants.c:337): table
kvalues_mxfp4 = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12} (×2 units);
err_i = |kvalues[i]·e − x|; strict-< ⇒ ties → lowest index = lowest |value|.
`e` = the DECODED scale value (ue4m3_to_fp32(s)), not the word.

`quantize_row_nvfp4_ref` (ggml-quants.c:384): per 16-sub-block:
amax over f32; s = ggml_fp32_to_ue4m3(f32(amax/6.0f)); d = ue4m3→f32;
codes via best_index_mxfp4(x_j, d).

Int-group reference: `tools/convert/common/quantize.py`
`_canonical_scale_words` + `quantize_matrix` (read fully, captured in §4.2):
scale = f16(f32(f64(max_abs)/f64(qmax))); underflow (scale==0, max_abs>0)
→ 2^-24; recip = f32(f64(1)/f64(scale)); codes = clamp(round(x·recip),
qmin, qmax) — torch.round = RNE. Pad K to %gs with zeros (does not affect
scale since zero ≤ amax). qmax: Q6 31, W8 127; qmin −32/−127.

## 5. Encoder / converter design

### 5.1 Files (new, single repo — ninfer-dev only)

- `tools/convert/common/encoder_imatrix.py` — the §4 encoders, pure
  numpy, source-agnostic. **IMPLEMENTED (session 3); public API (final):**
  - `encode_nvfp4(t f32 [n,k], band f32 [k] |None) -> (packed u8
    [n,k//2], scale words u8 [n,k//16])` — input is already t-space
    (t = w·d_w); band=None → reference (byte-exact vs ggml ref).
  - `encode_row_split(w f32 [n,k], band f32 [k] |None, gs, qmax) ->
    (codes i8 [n,k//gs,gs], scales f16 [n,k//gs])` — gs 64/qmax 31 (Q6)
    or 32/127 (W8); band=None → MAXABS reference (byte-exact vs
    quantize.py recipe).
  - `weight_divisor(w) -> 4B LE` = f32(2688/A); `input_divisor(band) ->
    4B LE` = f32(2688/R), R = max_j sqrt(band[j]).
  - `decode_nvfp4(packed, words, d_w)` test oracle; grid tables
    `E2M1_MAG/E2M1_MID/E4M3_WORD_VALUES`; scalar oracles
    `_oracle_fp32_to_ue4m3`, `_oracle_ue4m3_to_fp32`,
    `_oracle_best_index_mxfp4`, `oracle_quantize_row_nvfp4_ref`.
  - Chunked internally (NVFP4 512 rows, row-split 256 rows) to bound
    worker memory at 32-way parallelism.
  - `Nvfp4Result(packed_codes u8 [N,K//2], scales u8 [N,K//16],
    d_w_bytes 4)` — `encode_nvfp4_imatrix(w f32 [N,K], band f32 [K]
    |None, d_w f32)` (band None → reference only)
  - `RowSplitResult(codes i8 [N,Kpad], scales f16 [N,G])` —
    `encode_row_split_imatrix(w f32 [N,K], band f32 [K] |None, fmt)`
  - reference variants `encode_nvfp4_reference`, `encode_row_split_reference`
  - oracle helpers: `e4m3fn_decode/encode`, `e2m1_nearest` (ties→low |k|),
    `ue4m3_grid_points(lo, hi) -> words`
- `tools/convert/qwen4exp/__init__.py`
- `tools/convert/qwen4exp/convert_jbnvfp4.py` — driver:
  1. open source GGUF (`gguf.GGUFReader`, memmap — 354 GB streams fine),
     open imatrix GGUF, load all bands+counts.
  2. build the object plan (this doc §2 table) → `plan_objects` specs
     (`tools/artifact/container.py`: `TensorSpec(name, shape, format,
     layout)`, `ResourceSpec`, `ArtifactIdentity`).
  3. stream source tensors per layer: read GGUF bytes at
     `t.data_offset` (bf16), convert to f32, apply §4 encoder with the
     role's band (expert-banded rows use per-expert band columns),
     encode payload via `tools/artifact/layouts.py`
     (`encode_nvfp4` / `encode_row_split` / `encode_direct`),
     `ArtifactWriter.write(name, payload)`.
  4. resources: 6 frontend files from
     `/llm/models/Qwen3.8-Flash-Next/master/` (all present):
     tokenizer.json, tokenizer_config.json, chat_template.jinja,
     generation_config.json, preprocessor_config.json,
     video_preprocessor_config.json — same 6 canonical names as
     `tools/convert/qwen3_6/common/official_resources.py`.
  5. `finish()` → report JSON (sizes, per-tier bytes, per-object
     band-weighted RMSE vs reference RMSE, promotion candidates).

### 5.2 Layout encoders (existing, reused as-is)

`tools/artifact/layouts.py`:
- `block_scale_geometry("NVFP4", shape)` → n, k (K%16==0), groups_per_row
  (K/16), code_plane_bytes = N·K/2, scale plane (swizzled
  128×4 blocks, `swizzle_nvfp4_scales`), trailing 4-B d_w.
- `encode_nvfp4(packed u8 [N,K//2], scales u8 [N,K//16], d_w 4B, shape)`.
  Scale validity: no 0x80 sign, no 0x7F.
- `row_split_geometry(spec, shape)` → k_pad, groups_per_row;
  `encode_row_split(codes i8 [N,Kpad], scales f16 [N,G], spec, shape)`
  layout `row-split-k128-v1`.
- `encode_direct(x, "BF16"|"FP32")`.
- `decode_nvfp4_words(payload, shape)` → (codes, scales, d_w) for
  round-trip self-test.
- formats registry `tools/artifact/numeric.py`: `Q6G64_F16S` (bits 6,
  gs 64, qmin -32 qmax 31), `W8G32_F16S` (8, 32, -127/127), `NVFP4`,
  `BF16`, `FP32`; scales binary16.

Container `tools/artifact/container.py`:
- `ArtifactIdentity(model_id, weights_id)` — model_id
  `qwen3.8-flash-next/jbnvfp4` (owner may rename), weights_id = the
  source checkpoint hash or a descriptive string (check 27b driver for
  the exact pattern — `tools/convert/qwen3_8_27b/convert_nvfp4.py`).
- `ArtifactWriter` / `write_artifact` (L445): constructor + `write(name,
  payload)` + `finish()` — re-read exact signature before driver.
- Inspect: `python -m tools.artifact.inspect --objects --json file`.

### 5.3 Parallelism / memory

- Per layer: NVFP4 gate+up bf16 = 3.35 GB; Q6 down 1.68 GB. Encoder
  temps bounded by internal row chunking (~0.4–1.5 GB per expert task).
- Design: **phase 1** = 32-worker pool of per-object encode tasks
  (parent amax pre-pass per NVFP4 object is sequential in-task: read the
  object slice once for amax, then encode — the GGUF memmap is shared
  read-only). Workers write per-object temp payload files under
  `out/.jb_tmp/` (186.5 GB; `/` has 697 GB free).
- **Phase 2** = sequential parent loop: stream each temp payload through
  `ArtifactWriter.write(name, payload)` in strict plan order, delete
  temps as consumed. `per_layer_token_embedding` (102.4 GB pass-through)
  is streamed chunked directly from the source GGUF in phase 2 (no temp).
- Wall estimate: encode 30–60 min @32 workers + IO ~15 min + write
  186 GB ~2 min. Total < 1.5 h.

### 5.4 Self-tests (gate before full run)

1. Byte parity no-imatrix: `encode_*_imatrix(band=None)` ≡
   `encode_*_reference` bit-for-bit (random f32 + real expert slice).
2. gguf-package oracle: NVFP4 words ≡ `gguf.NVFP4` dequantize/quantize
   path on the same inputs (reference mode).
3. Search never worsens: band-weighted RMSE(search) ≤ RMSE(reference)
   on real tensors (holds by construction; assert per tensor).
4. Decode round-trip: `decode_nvfp4_words` + grid decode reproduces
   encoder intent exactly (e4m3/e2m1 exact).
5. Shape/contract: every object shape satisfies NVFP4 %16, row-split
   K%gs; K values: 2560/640/6144/512/128 all fine.
6. After build: `tools.artifact.inspect --json` (object count, format
   histogram, byte total ≈ 186.5 GB).

## 6. Open items / decisions

- [ ] `ArtifactWriter` constructor: `ArtifactWriter(path,
      ArtifactIdentity(model_id, weights_id), specs)`; `.write(name,
      payload)` STRICT plan order (raises on mismatch); payload may be a
      generator of chunks; `.finish()` truncates+flushes.
      `write_artifact(path, identity, entries)` is a convenience wrapper.
      ⇒ workers emit per-layer temp payloads; final sequential merge
      streams them in object order. 27b pattern:
      `ArtifactIdentity("qwen3.8-27b", "nvfp4")` — qwen4exp:
      `("qwen3.8-flash-next", "jbnvfp4")` (confirm with owner).
- [x] PLE prime data (RESOLVED, session 3 KV type check):
      `ple.layer_multipliers` = [23703573157769, 20109073645365,
      8052911324071] — **u64[3]** (≈45-bit; the earlier "u32" label was
      wrong: the values exceed 2³², and the KV field type is
      ARRAY(UINT64)), `ple.heads_per_ngram` = 8, `ple.ngram_size` 3,
      `ple.layers` [1], `ple.head_offsets` **u64[16]** (starts
      0,20000003,40000026,…), `ple.head_vocab_sizes` **u64[16]** primes
      (20000003, 20000023, 20000033, 20000047, …), `ple.eos_token_id`
      248044, `ple.image_token_id` 248056, `ple.conv_kernel` 4.
      Artifact has no I64 direct format (registry: BF16/FP32/I32 only)
      ⇒ store as I32 lo/hi little-endian pairs: `text/ple/
      layer_multipliers` I32[6], `text/ple/head_offsets` I32[32],
      `text/ple/head_vocab_sizes` I32[32] (names TBD vs the future
      qwen4exp C++ target binding — flag in report).
- [x] `moe/routed_gate_up` input-divisor naming (RESOLVED):
      `text/layers/{l}/moe/gate_up_input_divisor` FP32[1] (plan §4.6
      wording). Recorded in the conversion report.
- [x] **Input-divisor value (RESOLVED, session 3)**: .ninfer stores a
      true DIVISOR (engine `alpha = 1/(in_div·w_div)`, 27B values O(10²)
      confirm divisor orientation). Imatrix gives Σ_t x² per column, not
      amax ⇒ `R = max_j sqrt(band[j])` is a strict upper bound on
      |x[t,j]| over all calibration tokens; **`in_div = f32(2688.0/R)`**
      (mirrors the weight divisor 2688/A). Implemented in
      `encoder_imatrix.input_divisor`; self-tested (round-trip + zero
      guard). No 4× Gaussian factor needed (R is exact, not statistical).
- [ ] Plan §4.2 "Layout" paragraph shape strings are swapped
      (`[524288,640]`/`[327680,2560]`); §4.6 table + this doc have the
      correct `[655360,2560]`/`[1310720,640]`. Fix the §4.2 paragraph
      when next editing nvfp.md (byte totals in §4.3 already match the
      correct shapes).
- [ ] Plan §4.6 table row "ssm_out ×36, attn_output ×12 [6144,2560]"
      is in [K,N] convention — object shape is [2560,6144] per the op
      map; noted, fix with the §4.2 edit.
- [ ] "JB" expansion: owner naming, not yet specified.
- [ ] Optional HC flip BF16→W8 (−0.6 GB): deferred.
- [ ] **Encoder file name**: plan §2.4 says
      `tools/convert/common/encoder_imatrix_nvfp4.py` — but it will
      hold Q6/W8 search too; keep the plan's name or rename to
      `encoder_imatrix.py`. Use `encoder_imatrix.py` (clearer); update
      plan §2.4 line in the same edit as the shape fix.

## 7. Emerged facts (session 2)

- **GGUF KV fields (source file, all 63)**: arch `qwen4exp`, block_count
  48, embedding_length 2560, embedding_length_per_layer_input 160,
  expert_count 512, expert_feed_forward_length 640,
  expert_shared_feed_forward_length 640, expert_used_count 10,
  full_attention_interval 4, attention head_count 24 / head_count_kv 2
  (key_length 256, value_length 256), indexer.head_count 4 /
  key_length 128 / top_k 2048, compress_ratios [0,0,0,4]×12,
  hyper_connection.count 4 / low_rank 320, context_length 262144,
  ssm.group_count 16 / inner_size 6144 / state_size 128 /
  time_step_rank 48 / conv_kernel 4, rope dimension_count 64 /
  sections [11,11,10,0] / freq_base 1e7, rms eps 1e-6, ple.* (see open
  item), sampling temp 1.0 top_k 20 top_p 0.95, bos 248044 eos 248046
  (tokenizer.ggml) / image 248056, pre `qwen35`. general.file_type 32
  = this fork's BF16 enum (≠ stock 22). GGUFv3, quantization_version 2.
- **GGUF tensor naming (source, 1224 tensors)**: `blk.{0..47}.*` (NOT
  `layer.`), globals `token_embd.weight`, `per_layer_token_embd.weight`,
  `output.weight`, `output_hc_{up,down,norm}.weight`. PLE only in blk.1
  (`ple_conv1d/ple_key/ple_norm_{conv,key,query}/ple_value`). Full-attn
  layers 3,7,…,47 have `attn_{q,k,v,output}*.weight` +
  `indexer.{q,k}_{proj,norm}.weight`; GDN layers have `attn_qkv`,
  `attn_gate`, `ssm_*`.
- **Imatrix file**: 1852 tensors, names `blk.{l}.<role>.weight.in_sum2`
  + `.counts` — SAME naming as source. No bands for: `output`,
  `token_embd`, `per_layer_token_embd`, `ssm_a`, `ssm_dt.bias`,
  `ssm_conv1d`, `ssm_norm`, `hc_*_norm` (all pass-through roles — fine).
  HAS bands for `output`?? NO — `output.weight.in_sum2` absent (LM head
  is BF16 anyway). PLE key/value DO have bands (BF16 pass-through —
  unused, fine).
- **Reader API**: `gguf.GGUFReader(path)` (memmap, lazy);
  `r.tensors` → list of `ReaderTensor(name, tensor_type, shape,
  n_elements, n_bytes, data_offset, data, field)`; `r.get_tensor(i)`;
  `r.fields` dict → `r.get_field(key)` → `ReaderField` with `.contents()`
  (decodes KV). Tensor bytes: `t.data` memmap slice at data_offset.
- **Encoder speed estimate revised**: NVFP4 search over 80.5B elems:
  16-sub-blocks, ~24 candidates × 16 elems → ~30 GFLOP-equivalent
  vector ops; vectorized numpy at ~5 GB/s effective ≈ 30–60 min for
  gate/up alone; Q6/W8 40.3B + 2.8B similar → total encode ~1.5–2.5 h
  single-process; ÷8 workers ≈ 15–25 min. IO: 354 GB read + 186 GB
  write ≈ 10 min. Total < 1 h at N=8.
- **d_w verdict (FINAL, session 3)**: `d_w = f32(2688.0/A)` — proven by
  the saturation argument (27B artifact scale values reach 448 = word
  126; a reference-scaled saturated block has value RNE_e4m3(a16/6), so
  448 requires a16 = 2688 ⇒ A·d_w = 2688). The O(10³) d_w values alone
  do not discriminate 448/A vs 2688/A. Input divisors likewise:
  `in_div = f32(2688.0/R)`, R = max_j sqrt(band[j]) = strict activation
  upper bound (imatrix band is Σ_t x², so sqrt(band[j]) ≥ |x[t,j]| for
  every calibration token). The session-2 "448/amax settled" line was an
  error and is retracted here.
- **Nibble order**: low nibble = even element index (layouts
  `_pack_low_nibbles`: `out = even | (odd<<4)`; ggml `qs = x0 | (x1<<4)`
  with x0 = even j).
- **encode_direct** requires a torch tensor of exact dtype
  (BF16→torch.bfloat16, FP32→torch.float32, I32→torch.int32),
  contiguous, LE bytes. For pass-through we can convert numpy→torch
  per tensor (fine) or hand the raw GGUF bytes directly (bf16 words are
  LE in this fork — verify one tensor against torch round-trip in
  self-test 0).
- **encode_row_split** wants `codes` int8 [N, G, gs] (K padded to %gs
  with zero groups — caller fills padding), `scales` f16 [N, G],
  layout `row-split-k128-v1`, format string `"Q6G64_F16S"`/`"W8G32_F16S"`.
  **encode_nvfp4** wants packed u8 [N, K//2], scales u8 [N, K//16]
  (natural order; it swizzles internally), divisor 4 B.
- **TensorSpec(name, shape, format, layout)** for NVFP4: layout
  `"blockscale-k16-m128x4-v1"`, requires N%128==0, K%64==0 ✓
  ([655360,2560]: 655360%128=0 ✓, 2560%64=0 ✓). Row-split:
  `"row-split-k128-v1"`. Direct: `"contiguous-le-v1"`.
- **Artifact identity**: `ArtifactIdentity("qwen3.8-flash-next",
  "jbnvfp4")` (pending owner confirmation of model_id string).


## 8. Progress

- [x] 2026-08-30: recon — gguf 0.19 API (memmap reader, `NVFP4` oracle
      class), imatrix layout (per-expert bands + counts), full source
      inventory (1224 tensors, shapes locked), 27b d_w convention
      verified empirically (d_w = 2688/amax, s=448 saturated),
      reference int encoder contract captured, object naming from 35b
      doc, plan §2/§4.2/§4.6/§4.7 re-read.
- [x] 2026-08-30 (session 2): plan §2.3–§2.5 re-read verbatim;
      llama.cpp reference algorithms captured bit-exact
      (`ggml_fp32_to_ue4m3`, `best_index_mxfp4`, `quantize_row_nvfp4_ref`);
      27B artifact NVFP4 objects decoded — **d_w = 448/amax settled
      empirically** (word max 126/0x7E=448 value; d_w ∈ {6400, 11584, …}
      ⇒ amax O(0.03–0.19)); input divisors are true divisors
      (448/amax_x, engine `alpha = 1/(in_div·w_div)`); nibble order
      confirmed (even = low); PLE KV data resolved (u32 multipliers /
      head offsets / vocab sizes — NOT 45-bit); all 63 source KV fields
      read; imatrix coverage re-verified (no bands only for pass-through
      roles); encode_direct/encode_row_split/encode_nvfp4 signatures
      and TensorSpec/layout strings locked; writer = strict plan-order
      (per-layer temp + sequential merge design); speed revised < 1 h
      @N=8.
- [x] 2026-08-30 (session 3): **Phase B COMPLETE** —
      `tools/convert/common/encoder_imatrix.py` +
      `tools/convert/common/test_encoder_imatrix.py` written; all 8
      self-tests PASS: (1) e4m3 words bit-parity vs gguf-package oracle
      (200k random + 17 edges) + scalar ggml port; (2) e4m3 decode 127
      words; (3) e2m1 snapping 1004 scalar-oracle checks + 14 tie points
      (incl. −0.25 → code 0: indices 0 and 8 both hold 0, strict-< keeps
      0); (4) **NVFP4 reference byte-parity vs ggml
      `quantize_row_nvfp4_ref` port (96 rows × 16 sub-blocks, packed +
      words)**; (5) NVFP4 imatrix search: weighted err 2.79e10 →
      8.37e9 (3.3×), never worse, words stay in [ref, 126]; (6) Q6/W8
      reference byte-parity vs scalar MAXABS oracle (96 rows, gs 64/32);
      (7) decode error bound; (8) divisor round-trips.
      Bugs found & fixed this session: e4m3 word bit layout is
      (ue<<3)|man (3-bit mantissa, ggml-impl.h:556) — not <<4; e2m1
      negative-zero → code 0 not 8; `ndarray.take` 2-D index broadcast
      (replaced by `take_along_axis`); band-path broadcast axes
      (`s[..., None]` not `s[..., None, :]`); zero-scale recip guard in
      the int-group reference path (mirrors quantize.py recip=0 when
      scale=0); row chunking added to both band searches (worker RAM);
      oracle a16/6 division must be f32 (ggml `amax/6.0f`), not f64
      double-rounding. d_w convention FINAL = 2688/A (saturation
      argument); stage1 §4.1/§7 "448/amax" text retracted. PLE KV types
      verified = ARRAY(UINT64) ⇒ I32 lo/hi pairs in the artifact. `gguf`
      package NVFP4 oracle = bit-exact ggml convention (×2 kvalues +
      ×0.5 ue4m3 decode).
- [x] Phase C: **COMPLETE (session 4/5)** —
      `tools/convert/qwen4exp/convert_jbnvfp4.py` (851 lines, pure
      numpy + ProcessPoolExecutor, 32 workers). Object plan: 1077
      objects / 186.468 GB payload (BF16:356 FP32:388 I32:3 NVFP4:48
      Q6:48 W8:228 resource:6), `--plan-only` validated.
      **CRITICAL orientation discovery (session 5):** the gguf 0.19
      reader returns C-order data with the metadata dims **reversed**
      — 2-D linears arrive as `(out, in)` (already artifact
      orientation, no transpose), 3-D experts arrive as `(E, out, in)`
      (expert e = `data[e]`, NOT `data[..., e]`), 1-D unchanged. Only
      the 2-D conv1d kernels (metadata `(K=4, C)`) arrive as `(C, K)`
      and need a transpose. Driver rule implemented as `_orient()`
      (passthrough if `data.shape == want`, transpose iff the reversed
      shape matches). The PLE table arrives directly as `(rows, 160)`
      token-major — plain chunked copy, no transpose.
      Other session-5/6 fixes: W8 band lookup missing `.in_sum2`
      suffix; W8 band shape check must be `w.shape[1]` not HIDDEN
      (ssm_out band is 6144); expert slicing `[e]` not `[:, :, e]`;
      **encoder bug (encoder_imatrix.py `encode_row_split`)**: the 1-D
      (broadcast) band was sliced `bandg[r0:r1]` — zero rows when
      `n > 256` (the chunk size); fixed by selecting the whole 1-row
      tensor when `bandg.shape[0] == 1` (`encode_nvfp4` already had the
      guard). Verified: 1-D band == per-row broadcast, all 9
      self-tests still pass.
- [x] Phase D smoke: `--smoke --workers 32` (layers 0,1,3) — **PASS
      in 584 s**: 86 objects, 7.723 GB, all expert-0 byte-parity
      checks passed (NVFP4 + Q6, per layer), PLE 1M-row chunk + SHA256
      recorded, KV checks, per-layer amax/d_w/in_div in report
      (layer 0: amax 0.6875, d_w 3909.8, in_div 5.126). Inspect
      validates identity `qwen3.8-flash-next/jbnvfp4`, formats,
      layouts, 16 KiB header, object table.
- [ ] Phase D full: 48-layer run **in progress**
      (session 6). First attempt to
      `/llm/models/Qwen3.8-Flash-Next-JB-NVFP4.ninfer` failed at
      startup: `/llm/models` is `jay:jay`-owned and not writable as
      `robot` (the executing user). Output path is now
      `/home/robot/jb/Qwen3.8-Flash-Next-JB-NVFP4.ninfer` (645 GB
      free); move as `jay` afterward. expected ~2.5–3 h (MoE layers dominate:
      ~195 s/layer at 32 workers; PLE 102.4 GB stream adds ~10 min).
      Post-run: `tools.artifact.inspect` + size check (~186.5 GB) +
      report. Per-layer log lines `[i/1077] name t=...s`.
      Timing note: layer-3 full-attn W8 took 185 s of its 577 s —
      de-interleaved qkqv (fancy-index on 12288×2560) is the
      full-attn cost; MoE NVFP4/Q6 maps are ~40 s each.
- [x] 2026-08-30 (session 7): **FP8 quantizer added to
      `encoder_imatrix.py`** (now 701 lines). `quantize_fp8_row_scaled(w,
      band, row_chunk=256) -> (codes u8 [n,k], scale_words u16 [n])`
      matches the existing 27B `.ninfer` FP8 convention 100%: reference
      path (band=None) = `s = RNE_bf16(amax/448)`;
      `code = RNE_e4m3fn(f32(f64(x) * f64(f32recip(s))))`; zero rows →
      all-zero codes + zero scale; subnormal-scale guard (min BF16 word
      0x0001). Imatrix path = 17 candidates
      `s_i = RNE_bf16(s_ref·2^(-i/4))`, i=0..16, candidate 0 = reference
      (never-worse guarantee). Verified: E4M3 RNE table bit-exact vs
      torch `float8_e4m3fn` (127 values); reference path bit-exact vs
      `tools/.../fp8_embedding.quantize_bf16_rows` (4 cases incl. zeros /
      −0.0 / large amax); never-worse on 16 random matrices; designed
      case (B=5000 column) scale changes 2.875→2.855; zero rows →
      codes 0 / scales 0; round-trip mean rel 0.0225, max 0.9628, no NaN
      words; per-row band == 1-D broadcast; chunk independence
      (rc=1,7,256). **Performance (not benchmarked further — CPU
      reserved for the flash-next run):** imatrix path 16 small
      matrices (≤512×5120) = 99.7 s; 16384×5120 imatrix path timed out
      > 400 s ⇒ the 27B dense FP8 objects (up to 17408×5120 banded)
      are the slow part; per-layer task chunking (2048 rows) keeps
      worker RAM bounded; est. full 27B run 1–2 h.
- [x] 2026-08-30 (session 8): **27B JB-NVFP4 converter driver written —
      NOT RUN (owner directive).**
      `tools/convert/qwen3_8_27b/convert_jbnvfp4.py` (798 lines),
      mirrors the flash-next driver.
      **Sources (all already on disk):** canonical
      `/llm/models/Qwen3.8-27B-BF16.gguf` (54.6 GB, arch `qwen35`, 866
      tensors, 65 blocks: 48 GDN + 16 full-attn at l%4==3 + nextn
      block 64); imatrix
      `/llm/models/imatrix_unsloth_qwen3.8-27B.gguf_file` (13.6 MB,
      992 tensors = 496 `in_sum2` 1-D bands + 496 `counts`); base
      artifact `/llm/models/qwen3_8_27b_nvfp4.ninfer` (21.5 GB, 1124
      objects) for the imatrix-free sections.
      **Plan:** 1124 objects (identical to the existing artifact's
      object plan, reused from `inventory_nvfp4`): 771 text-core
      generated from GGUF+imatrix; 353 byte-copied from the base
      artifact (6 resources sha256-verified against
      `OFFICIAL_RESOURCE_SHA256`, 333 vision, 12 MTP, 2 draft head —
      no imatrix coverage for any of them). Identity
      `("qwen3.8-27b", "jbnvfp4")`.
      **Recipe (format allocation = existing artifact 100%):**
      MLP 0–55 NVFP4 (gate+up share `d_w = f32(2688/amax(gate,up))`;
      down own `d_w`; `input_scale_divisor = f32(2688/max_j
      sqrt(band[j]))` over gate+up bands — same `np.concatenate`
      convention as flash-next); MLP 56–63 FP8 (imatrix 17-candidate
      search). GDN qkvz rows `[attn_qkv 10240 (q|k|v) | attn_gate 6144
      (z)]` FP8; GDN output `(5120, 6144)` FP8 (ssm_out band).
      Full-attn qkvz rows `[q 6144 | k 1024 | gate 6144 | v 1024]`
      FP8 with the same per-head de-interleave as flash-next
      (`_Q_IDX`/`_G_IDX`, head h: q = h·512..h·512+255, gate =
      h·512+256..h·512+511); full-attn output FP8 (attn_output band).
      token_embedding + output_head FP8 **reference profile**
      (band=None — no imatrix entry).
      **GGUF dtype facts (measured):** linears BF16 (type 30, data
      (out,in) u8 = (out,in) u16 words); 1-D norms, ssm_a, ssm_dt.bias,
      ssm_conv1d are **FP32** (type 0) in the source and are NOT exact
      BF16 upcasts ⇒ artifact BF16 objects (input/post/ssm/q/k norms,
      conv kernel) = RNE BF16 cast of the FP32 source values (the
      registered transform for the direct BF16 format); a_log/dt_bias
      stay FP32 verbatim; conv data (10240, 4) → transpose to
      (4, 10240) via `_orient`.
      **Imatrix band names:** `blk.{l}.{role}.weight.in_sum2`,
      role ∈ ffn_gate/ffn_up/ffn_down (64×, [5120]/[5120]/[17408]),
      attn_qkv/attn_gate (48×, [5120]), ssm_out (48×, [6144]),
      attn_q/attn_k/attn_v (16×, [5120]), attn_output (16×, [6144]).
      No bands for token_embd/output/nextn/conv/norms ⇒ embedding &
      output_head use band=None; copied sections untouched.
      **Parallelism:** fork-based `ProcessPoolExecutor(32)`,
      module-level reader globals; per-layer `_LayerState.ensure()`
      (idempotent, triggered from the write loop) submits: 3 amax
      tasks (NVFP4 layers) → d_w, then ~17 NVFP4 tasks (gate 9 + up 9 +
      down 5 row-chunks) or ~21 FP8 tasks; GDN/attn FP8 tasks 8–11;
      row chunks 2048 (1024 for down K=17408), each a multiple of 128.
      Parent assembles rows in order → `layouts.encode_fp8_row_scaled`
      / `encode_nvfp4` (torch tensors) / `encode_direct`.
      **CLI:** `--model --imatrix --base-artifact --out --workers 32
      [--layers '0-3,56'] [--smoke] [--plan-only]`. Smoke = layers
      (0,3,55,56,63) + final_norm, plus parent byte-parity re-encodes of
      the first task chunk of an NVFP4 gate and an FP8 gate (must match
      worker bytes exactly).
      **Validation done (static only, no run):** `py_compile` +
      `--help` OK; dispatch coverage audit: all 771 generated names hit
      an intended producer branch, 353 copies, 0 uncovered; layer
      partition 48 GDN ∪ 16 full-attn = 0..63; NVFP4 mlp layers
      0–55. Bugs found & fixed before first run: `_direct` double
      transpose (conv); `dw_gu` report NameError; smoke parity needed
      `ensure()` before future access; dead code removed.
      **To run (when the flash-next run is done / CPU is free):**
      plan-only first, then smoke, then full:
      `/home/robot/workplace/unsloth_env/bin/python -m
      tools.convert.qwen3_8_27b.convert_jbnvfp4 --model
      /llm/models/Qwen3.8-27B-BF16.gguf --imatrix
      /llm/models/imatrix_unsloth_qwen3.8-27B.gguf_file
      --base-artifact /llm/models/qwen3_8_27b_nvfp4.ninfer --out
      /home/robot/jb/Qwen3.8-27B-JB-NVFP4.ninfer --workers 32`
- [x] 2026-08-30 (session 8b): **object-identity confirmation vs the
      existing `/llm/models/qwen3_8_27b_nvfp4.ninfer`** — driver full
      plan (`build_plan` → `build_object_plan`) compared object-for-
      object: 1124 = 1124; (name, offset, bytes) 1124/1124 identical;
      (kind, shape, format, layout) 1124/1124 identical; write order
      identical; payload span identical (21,492,514,816 B); per-format
      counts identical (BF16 534 / FP32 208 / FP8 146 / NVFP4 112 /
      Q4 55 / Q5 54 / Q6 1 / W8 7 / I32 1 / resources 6); producer
      coverage 771 generated + 353 copied, 0 unmapped. Only intended
      non-identities: identity weights_id `nvfp4` → `jbnvfp4`, and
      payload contents of the 771 generated objects (re-quantized from
      BF16 GGUF + imatrix; same shapes/formats/layouts/offsets).
- [x] 2026-08-30 (session 8b): **flash-next FULL conversion run
      COMPLETE.** PID 896119 + 32 workers finished at t≈160 min
      (9612.4 s): 1077/1077 objects, **186.468 GB** →
      `/home/robot/jb/Qwen3.8-Flash-Next-JB-NVFP4.ninfer` (+
      `.conversion.json`). Log `/tmp/jbnvfp4_full.log` ends clean
      ("done in 9612.4s"). Phase 1 (M0 converter) is DONE. Post-run
      validation next: `tools.artifact.inspect --objects`, verify
      ~186.5 GB + object census vs phase2.md §9.
- [x] 2026-08-30 (session 9): **`phase2.md` WRITTEN** (33 KB, 17
      sections) — the qwen4exp runtime-harness plan. Contents:
      (1) phase 2 = supported plumbing producing garbage (compiles,
      binds real 186.5 GB artifact, fits 32 GB, 1+ prefill + N decode
      to completion, no crash); NOT numerically correct (that is
      phase 3 / plan §4.8 M1–M7). (2) Architectural decisions: NEW
      target family (not a qwen3_6 Variant — HC/PLE/QSA/512×10 MoE
      don't fit the 3-leaf Variant interface); fully eager (no graph
      capture, F5→M8); text-only, no MTP/vision/DFlash; prefix reuse
      DISABLED (degenerate Program paths); reuse qwen3_6::Frontend +
      core/ + runtime/ + ops/; adds identity qwen3.8-flash-next/
      jbnvfp4. (3) File layout src/targets/qwen4exp/ mirroring 27b.
      (4) Registry integration: 4 concrete edits (registry.h Loaded/
      Instance + variant; registry.cpp construct_target branch via the
      generic construct_registered; src/CMakeLists.txt
      add_subdirectory; package CMakeLists). (5) Package surface:
      aliases + 7 statics. (6) Program interface: real vs degenerate
      method table (engine drives 12; plan/admit/prefill/decode/commit
      real; context-transaction/capture/pressure/prefix reuse = no-op
      "none"). (7-9) Binder + full object inventory table (1077
      objects; resident vs host-streamed; geometry constants).
      (10) CPU-side routines REAL in phase 2: K0 token_embd gather +
      PLE hash/gather; MoE expert-streaming orchestrator (D2H ids →
      mmap range reads → H2D scratch bank), exercising the true ~1.72
      GB/step H2D. (11) Per-component requirements table: new ops X
      (sparse_moe_512x10, qsa_indexer, qsa_softmax_attention, ple),
      new small kernels N (grouped_rmsnorm, silu_div4,
      gate_mul_mean4, hc_combine, rope sections, moe_router_topk,
      stream_dot, softsign_gate, dilated_causal_conv1d_silu), Rn
      registration rows — each with signature shape, I/O, workspace,
      reference (llama.cpp qwen4exp.cpp line / ninfer op), phase-2 stub
      body, fill-in milestone (M#). (12) Real workspace/scratch/KV
      capacity = the ≤32 GB gate (14.1 GB @262K). (13) Stub policy:
      real contract + deterministic WELL-FORMED body (in-range token/
      expert ids, finite floats); moe_router stub emits fixed ids [0..9]
      so streaming reads real spans deterministically. Not stubbed:
      binder, capacity math, eager loop, streaming, K0, sampling, out
      publication. (14) Dataflow diagram. (15) Acceptance gates (build
      green, bind 1077, fit ≤32 GB, run to completion, deterministic
      garbage). (16) Pushback items (not a Variant; eager not captured;
      stubs at op+CPU level; M0 already done; identity-list update is
      phase 3). (17) Phase 1→2→3 relationship.
      **Exploration anchors (verified):** EngineCore<Instance> fully
      generic (src/runtime/engine/engine_core.h); registry
      construct_registered<Target,Loaded,Instance> template
      (src/targets/registry.cpp); 27b package.h = alias-set over family
      qwen3_6::Xxx<Variant>; Program<Variant> full interface
      (~30 methods, runtime.h:636); 27b/35b file tree (CMakeLists +
      export/package.h + impl/{config.h,load/bindings,package.cpp,
      variant.h/cpp}); all target packages compile into the single
      ninfer_engine static lib (src/CMakeLists.txt:300-330); sparse_moe
      op is CLOSED 35b geometry (256/top-8/2048/512, Q4+Q5|Q4+Q6|W8+W8;
      include/ninfer/ops/sparse_moe.h) ⇒ flash-next needs a NEW
      sparse_moe_512x10 closed op (512/top-10/2560/640, NVFP4+Q6).
