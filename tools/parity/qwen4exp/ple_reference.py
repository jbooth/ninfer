#!/usr/bin/env python3
"""M1 oracle for the qwen4exp PLE n-gram hash gather (phase3.md §4.4).

Independent port of `llm_graph_input_ple::set_input`
(llama.cpp @ cc83d7b48, src/models/qwen4exp.cpp:990). Written directly from
the reference source; it is the acceptance oracle for
`ninfer::targets::qwen4exp::detail::compute_ple_rows`
(src/targets/qwen4exp/impl/cpu/ple_hash.h).

Geometry (GGUF KV, qwen3.8-flash-next): ngram_size = 3, heads_per_ngram = 8,
ple_n_heads = 16 (heads 0..7: 2-gram; heads 8..15: 3-gram), head_dim = 160,
eos_token_id = 248044, image_token_id = 248056 (unreachable: v1 is text-only).

Modes:
  --ids 1,2,3 --position 2 [--eos 248044] --descriptors ARTIFACT
      print the 16 row indices for position `position` of the token list,
      using the descriptor objects (text/ple/layer_multipliers [3u64],
      text/ple/head_offsets [16u64], text/ple/head_vocab_sizes [16u64],
      each stored as I32 lo/hi pairs) read from the .ninfer artifact.

  --descriptors ARTIFACT
      parse the .ninfer header and print the PLE descriptor objects and
      the text/per_layer_token_embedding table extents.

  --row-bytes ARTIFACT ROW
      pread one 320-byte BF16 row of the PLE table and print its hex.

The .ninfer artifact is the sole authority for the descriptor values (not
the GGUF source); `--descriptors` is required for the hash modes.

Exit code 0 on success; 2 on usage/artifact error.
"""

import argparse
import json
import struct
import sys

PLE_NGRAM_SIZE = 3
PLE_HEADS_PER_NGRAM = 8
PLE_N_HEADS = 16


def ple_rows(ids, i, eos, multipliers, offsets, sizes):
    """Reference loop, verbatim semantics (qwen4exp.cpp:990-1040).

    ctx[0] is the current token; its own EOS does not cut its own context.
    A missing predecessor (before sequence start) reads as EOS, and an EOS
    predecessor cuts every further context slot to EOS.
    """
    n_gram = PLE_NGRAM_SIZE
    n_prev = n_gram - 1

    def tok(k):
        return ids[k] if 0 <= k < len(ids) else None

    ctx = [tok(i)]
    cut = False
    for s in range(1, n_gram):
        t = None if cut else tok(i - s)  # oldest-first predecessor lookup
        cut = cut or (t is None) or (t == eos)
        ctx.append(eos if cut else t)

    rows = []
    for n in range(2, n_gram + 1):
        # u64 wrap: the C++ reference multiplies u64 values and the product
        # wraps modulo 2^64 (well-defined unsigned overflow).
        mixed = (ctx[0] * multipliers[0]) & 0xFFFFFFFFFFFFFFFF
        for j in range(1, n):
            mixed ^= (ctx[j] * multipliers[j]) & 0xFFFFFFFFFFFFFFFF
        base = (n - 2) * PLE_HEADS_PER_NGRAM
        for g in range(PLE_HEADS_PER_NGRAM):
            h_i = base + g
            rows.append((mixed % sizes[h_i]) + offsets[h_i])
    assert len(rows) == PLE_N_HEADS
    return rows


def read_header(path):
    with open(path, "rb") as f:
        prefix = f.read(16)
        if prefix[:8] != b"NINFER\x00\x02":
            raise SystemExit(f"not a .ninfer v2 artifact: {path!r}")
        (json_bytes,) = struct.unpack("<Q", prefix[8:])
        header = json.loads(f.read(json_bytes))
    return header


def lift_u64_pairs(data_bytes):
    vals = struct.unpack(f"<{len(data_bytes) // 4}i", data_bytes)
    out = []
    for k in range(len(vals) // 2):
        lo = vals[2 * k] & 0xFFFFFFFF
        hi = vals[2 * k + 1] & 0xFFFFFFFF
        out.append(lo + (hi << 32))
    return out


def load_descriptors(path):
    header = read_header(path)
    objects = {o["name"]: o for o in header["objects"]}
    out = {}
    with open(path, "rb") as f:
        for key in ("text/ple/layer_multipliers", "text/ple/head_offsets",
                    "text/ple/head_vocab_sizes"):
            obj = objects[key]
            f.seek(obj["offset"])
            out[key] = lift_u64_pairs(f.read(obj["bytes"]))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated token ids")
    ap.add_argument("--position", type=int, default=-1)
    ap.add_argument("--eos", type=int, default=248044)
    ap.add_argument("--descriptors", metavar="ARTIFACT")
    ap.add_argument("--row-bytes", metavar="ARTIFACT")
    ap.add_argument("--row", type=int, default=-1)
    args = ap.parse_args(argv)

    if args.ids is not None:
        if args.descriptors is None:
            raise SystemExit("--ids requires --descriptors ARTIFACT")
        ids = [int(t) for t in args.ids.split(",")]
        desc = load_descriptors(args.descriptors)
        multipliers = desc["text/ple/layer_multipliers"]
        offsets = desc["text/ple/head_offsets"]
        sizes = desc["text/ple/head_vocab_sizes"]
        rows = ple_rows(ids, args.position, args.eos, multipliers, offsets, sizes)
        for r in rows:
            print(r)
        return 0

    if args.descriptors is not None:
        for key, vals in load_descriptors(args.descriptors).items():
            print(key, vals)
        header = read_header(args.descriptors)
        objects = {o["name"]: o for o in header["objects"]}
        table = objects["text/per_layer_token_embedding"]
        print("table", table["shape"], "bytes", table["bytes"])
        return 0

    if args.row_bytes is not None:
        if args.row < 0:
            raise SystemExit("--row-bytes requires --row")
        header = read_header(args.row_bytes)
        objects = {o["name"]: o for o in header["objects"]}
        obj = objects["text/per_layer_token_embedding"]
        row_bytes = obj["bytes"] // obj["shape"][0]
        with open(args.row_bytes, "rb") as f:
            f.seek(obj["offset"] + args.row * row_bytes)
            payload = f.read(row_bytes)
        print(payload.hex())
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
