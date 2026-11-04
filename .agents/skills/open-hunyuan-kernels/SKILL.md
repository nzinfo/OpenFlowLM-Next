---
name: open-hunyuan-kernels
description: Run Tencent Hy-MT2 (hunyuan dense, Hy-MT2-1.8B-NPU2) on the open XDNA2 dense recipe. Use when rebuilding the hy-mt2:1.8b kernel set, registering the FastFlowLM Q4NX container in a local registry, debugging a hunyuan load refusal, or checking the block-attention (rope_qknorm) route.
---

# Hy-MT2-1.8B on the dense recipe

HunYuan dense is the **sixth family** on `open_kernels/recipes/dense.py`. It is
Llama's shape with two family properties:

- **q / k RMSNorm after RoPE** — `QKNORM_POST_ROPE` in `recipes/dense.py`,
  `QKNORM_POST` in the `dx` / `dxB` kernels. The block-attention host half
  (`block_host.cpp attention_prep`) takes `qknorm_after_rope = true` and the
  manifest says `attn_block.prep = "rope_qknorm"`.
- **one static NTK-alpha RoPE base** — `recipes/spec.py` folds
  `rope_theta * alpha^(hd/(hd-2))` at spec time. For 1.8B: 10000 with
  `alpha = 1000` becomes **11158839.92507748**, which is what `spec.json` and
  the manifest carry. Do not "correct" it back to 10000.

## The container — FastFlowLM's 4608-byte Q4_0 chunk

`flm/models/Hy-MT2-1.8B-NPU2/model.q4nx` uses FastFlowLM's chunk, not the
5120 / 8704 / 4736 forms `pools.cpp` used to know:

```
d[256] bf16  @ [0, 512)      index g*32 + r      (g = 32-column group, r = row)
qs[256][16]  @ [512, 4608)   byte k*16 + r/2, even row in the low nibble
value(r, k) = d[k/32][r] * int4(nib)              two's complement, -8..7, no min
```

It is the Q4_K chunk with the scale promoted to bf16 and the min dropped. The
transcode to q4_1 (`pools::q4_0_4608_to_q4_1_chunks`, `pack.q4_0_4608_to_q4_1`)
is exact: flip bit 3 of every nibble and write `min = -8 * d`. The derived role
map stays `q4_1`, so **an existing kernel set runs the container unchanged**;
the scale is not `absmax/8`, it is the MSE fit `(q·w)/(q·q)` (guessing absmax
gives corr 0.05 against bf16).

## Build the kernel set

```bash
cd works/openflowlm-next
source utilities/env.sh
python open_kernels/export_qwen36_kernels.py \
    --model-dir /home/nzinfo/src.npu/amd_npu/models/Hy-MT2-1.8B-NPU2 \
    --out src/xclbins/Hy-MT2-1.8B-NPU2/open_kernels -j 4
```

The attention tuple `(hd 128, 16 heads, 4 kv, rot 128, qk_norm, no gate,
post-rope norm)` entered `catalogue.py` on 2026-11-05 after the hardware run
below, so `OPEN_KERNELS_UNVALIDATED=1` is no longer needed. A *new* HunYuan
size (the 7B's 32/8 is already there) would need it until measured.

The `ag_s<L>` / `ag_pv<L>` builds are **content-cached and shared**: 1.8B's
`m = (16/4) * 256 = 1024`, `hd = 128` is exactly Qwen3-4B's attention
geometry, so the export copies the existing 32 xclbins instead of compiling
them. Only `dx`, `dx_fast`, `ln`, `lm_head_q4`, `gemm_n*` are rebuilt.

## Register it

OpenFlowLM's shipped `src/model_list.json` has no hunyuan entry. Generate a
local registry (the script appends `hy-mt2:1.8b` and points `model_path` at the
workspace models tree):

```bash
cd works && python3 mk_hy_registry.py
export OFLM_CONFIG_PATH=$PWD/oflm-cfg/model_list_local.json
export OFLM_XCLBIN_PATH=$PWD/openflowlm-next/src
export OFLM_MODEL_PATH=/home/nzinfo/src.npu/amd_npu/models
```

`model_families.hpp` maps `hunyuan` to the Granite adapter (both are dense and
open-kernel only); the engine log names the model's own family.

**FastFlowLM 1.0.11 renamed the tag**: it is `hy-mt2-flash:1.8b`, not
`hy-mt2:1.8b`. The model files are the same.

## Verify

```bash
# unit, no NPU
ironvenv/bin/python -m pytest specs/open-engine/tests -q          # 821 passed
src/open_qwen36/build/manifest_test specs/.../manifest_qwen36.json specs/.../manifest_qwen3_4b.json
python open_kernels/model/replica_block.py --fixture /tmp/bh && src/open_qwen36/build/block_host_test /tmp/bh

# on the NPU, one process only
python utilities/dense-decode-probe/attn_route_check.py \
    --cli src/open_qwen36/build/open_qwen36_cli \
    --model /home/nzinfo/src.npu/amd_npu/models/Hy-MT2-1.8B-NPU2 \
    --kernels src/xclbins/Hy-MT2-1.8B-NPU2/open_kernels --tokens 19,64,600,981 --seq
```

Hand check of the block route against `dxB` (2026-11-05, 93-token prompt,
`--gemm-block --prefill-logits` both ways): 93/93 positions dumped, logits
corr **>= 0.999915**, argmax **92/93**, top-5 **463/465**, same greedy first
token. That is the bf16-P distance the other dense families keep, not a bug.

End-to-end (greedy, FLM 1.0.11 as ground truth):

| prompt | output |
|---|---|
| `今天天气很好。` | `The weather is very good today.` |
| `The quick brown fox jumps over the lazy dog.` | `快速的棕色狐狸跳过了懒狗。` |
| `人工智能正在改变世界。` | `Artificial intelligence is changing the world.` |

## Speed (Strix Halo, 2026-11-05, performance pmode)

| | open engine | FastFlowLM 1.0.11 |
|---|---:|---:|
| decode 1.8B q4 | 29–37 tok/s | 42–43 tok/s |
| prefill 484 tok, dxB only | ~95 tok/s | ~1105 tok/s |
| prefill 484 tok, block attention | ~186 tok/s | — |

Block attention roughly doubles prefill; the rest of the gap is the open
engine's per-block host stages and GEMM dispatches, and Qwen3-4B (a peer set
with the same `ag_*` kernels) measures ~125 tok/s at 494 tokens on the same
machine, so it is not HunYuan-specific. Neither route is bit-identical to the
other; on a 180-token translation the greedy text can drift by a word after
~50 tokens.

## Rules

- One NPU tenant: check `ss -tln` and `pgrep -af oflm` before running, and stop
  the other engine by port. A leftover `oflm serve` makes the next load fail
  with `mmap ... Cannot allocate memory` (errno 12), not a clear refusal.
- The block-attention route needs the `ag_*` xclbins *and*
  `attn_block.prep = "rope_qknorm"`; an older set without them still runs
  correctly on the `dxB` dispatches, just slower.
- The registry's `default_context_length` is 32768 while FastFlowLM's entry
  caps at 1024; the kernel set's `max_ctx_default` is 4096. Keep the prefill
  chunk at 4096.
