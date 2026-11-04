# Traces: OPEN-QUANT-FLM-Q4-0, OPEN-PACK-PLAN (canonical spec: specs/open-engine/spec.md)
"""FastFlowLM's Q4_0 containers: the 4608-byte chunk, its transcode to the pool's q4_1
chunk, and the agreement between the converter that writes the format and the packers
that read it.

Every NPU2 `.q4nx` FastFlowLM ships stores its projections this way -- Hy-MT2-1.8B, Qwen3,
MiniCPM -- and OFLM's own containers never do, so this is the one chunk format that has to
be read from the outside. What the reverse engineering settled (FastFlowLM 1.0.11, and
checked against the bf16 originals of Hy-MT2-1.8B: correlation 0.9937-0.9966 over all 225
quantized tensors, the residual being 4-bit quantisation and nothing else): a chunk is the
Q4_K chunk with the block scale promoted to bf16 and the min dropped, which is exactly
4608 B --

    d[256] bf16 @ [0, 512)      index g*32 + r   (g = 32-column group, r = row)
    qs[256][16] @ [512, 4608)   byte k*16 + r/2, even row in the low nibble
    value(r, k) = d[k/32][r] * int4(nib), the nibble TWO'S COMPLEMENT, -8..7, no min

so the tile is the 32-row x 256-column tile every other chunk holds, the meta index is
q4_1's `g*32 + r`, and the nibble raster is Q4_K's byte for byte. Two consequences the
tests pin:

  * the transcode is EXACT -- flip bit 3 of every nibble (two's complement to offset
    binary) and write min = -8 * d, which moves a bf16 exponent and nothing else -- so
    unlike Q4_K there is no rounding at all and the two readings are bit-equal;
  * no chunk index law, plan, manifest or kernel moves, and the derived role map stays
    q4_1, so an existing kernel set runs the container unchanged.

The same synthetic chunks are built, transcoded and hashed by
src/open_qwen36/pools_test.cpp, so the two packers are held to each other byte for byte
without a model file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recipes import pack

from test_pack_plan import _fnv1a, _lcg_bytes

REPO = Path(__file__).resolve().parents[3]

Q40 = 4608
CH = pack.CH
NCH = 8                                        # 8 chunks: 32 rows x 256 K each
OUT_DIM, IN_DIM = 128, 512

Q40_NAME = "model.layers.{l}.mlp.gate_proj.weight"
Q4_NAME = "model.layers.{l}.mlp.down_proj.weight"
BAD_NAME = "model.layers.{l}.mlp.up_proj.weight"

# FNV-1a 64 of the transcoded chunks of _q4_0_vector(NCH). src/open_qwen36/pools_test.cpp
# asserts the same number on the same bytes.
Q40_TRANSCODE_FNV1A = 0x9E55D107764FBD60
# ... and of the pool std_perm puts them in, beside a q4_1 tensor.
Q40_POOL_FNV1A = 0x99FDDC17B85CD7B4


def _bf16(u16) -> np.ndarray:
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _q4_0_vector(nch: int) -> np.ndarray:
    """The vector pools_test.cpp builds, byte for byte: per chunk one LCG seeded from the
    chunk index, 256 bf16 `d` with exponent 0x76 (finite, positive, a real weight's
    magnitude) and then 4096 plain nibble bytes -- arbitrary nibbles are all legal here,
    which is not true of a q4_1 chunk's mins."""
    M32 = 0xFFFFFFFF
    out = np.zeros((nch, Q40), np.uint8)
    for c in range(nch):
        s = (0x9E3779B9 * (c + 1)) & M32
        d = np.zeros(256, np.uint16)
        for i in range(256):
            s = (s * 1664525 + 1013904223) & M32
            d[i] = 0x3B00 | (s >> 24)
        out[c, :512] = d.view(np.uint8)
        for i in range(4096):
            s = (s * 1664525 + 1013904223) & M32
            out[c, 512 + i] = s >> 24
    return out


def _q4_0_exact(chunk) -> np.ndarray:
    """[4608] chunk bytes -> [32, 8, 32] f32 (row, block, lane), read straight off the
    layout table rather than through the transcode."""
    b = np.asarray(chunk, np.uint8)
    d = _bf16(np.ascontiguousarray(b[:512]).view(np.uint16)).reshape(8, 32)      # [g, r]
    qs = b[512:4608].reshape(256, 16)                                           # [k, r // 2]
    r = np.arange(32)
    nib = np.where(r % 2 == 0, qs[:, r // 2] & 0xF, qs[:, r // 2] >> 4).astype(np.int32)
    nib = np.where(nib < 8, nib, nib - 16).astype(np.float32)                    # [k, r]
    return nib.T.reshape(32, 8, 32) * d.T[:, :, None]


class Q40Container:
    """One 4608-byte tensor, one q4_1 tensor and one with 1280-byte chunks -- the shape of
    a FastFlowLM container, which keeps everything but the embedding in Q4_0."""

    def __init__(self):
        self.data = {
            Q40_NAME.replace("{l}", "0"): _q4_0_vector(NCH).reshape(-1),
            Q4_NAME.replace("{l}", "0"): _lcg_bytes(0x1234567, NCH * CH),
            BAD_NAME.replace("{l}", "0"): _lcg_bytes(0x89ABCDE, NCH * 1280),
        }
        self.chunks = {Q40_NAME.replace("{l}", "0"): Q40,
                       Q4_NAME.replace("{l}", "0"): CH,
                       BAD_NAME.replace("{l}", "0"): 1280}

    def raw(self, name):
        return self.data[name]

    def chunk_bytes_of(self, name):
        return self.chunks[name]


def _q4_0_pool() -> np.ndarray:
    m = Q40Container()
    dst = np.zeros(2 * NCH * CH, np.uint8)
    pack.apply_op({"op": "std_perm", "tensor": Q40_NAME, "dst": 0, "nch": NCH, "in_dim": IN_DIM},
                  m, 0, dst)
    pack.apply_op({"op": "std_perm", "tensor": Q4_NAME, "dst": NCH * CH, "nch": NCH, "in_dim": IN_DIM},
                  m, 0, dst)
    return dst


# ------------------------------------------------------------------- the transcode
def test_the_transcode_reads_exactly_what_the_container_says():
    """Unlike Q4_K, this transcode moves no value: the nibble bit flip is a relabelling and
    -8 * d is an exponent change, so the pool's `nib*d + m` reproduces `d * int4(q)` bit
    for bit. That is what makes a container's own reading and the NPU's the same weights."""
    from q4nx import dq_chunks_q4_0, dq_chunks_q4_1
    src = _q4_0_vector(NCH)
    got = dq_chunks_q4_1(pack.q4_0_4608_to_q4_1(src))
    want = dq_chunks_q4_0(src)
    assert np.array_equal(got, want)
    assert np.array_equal(want, np.stack([_q4_0_exact(c) for c in src]))


def test_the_numpy_and_cpp_transcodes_agree():
    """The gate on the two implementations. src/open_qwen36/pools_test.cpp builds the same
    chunks and asserts this same FNV-1a; if either transcode changes, one of the two fails."""
    assert _fnv1a(pack.q4_0_4608_to_q4_1(_q4_0_vector(NCH)).tobytes()) == Q40_TRANSCODE_FNV1A, \
        "the FLM Q4_0 transcode changed; update src/open_qwen36/pools_test.cpp too"


# ------------------------------------------------------------------- as a pack source
def test_a_q4_0_source_packs_in_the_same_chunk_order():
    """The permutation is applied to transcoded chunks, exactly as it is to file chunks: a
    4608-byte chunk holds the same 32 x 256 tile, so std_perm does not move."""
    got = _q4_0_pool()[:NCH * CH].reshape(NCH, CH)
    want = pack.q4_0_4608_to_q4_1(_q4_0_vector(NCH))[pack.std_perm(NCH, IN_DIM)]
    assert np.array_equal(got, want)


def test_a_q4_1_tensor_beside_it_is_still_a_verbatim_chunk_copy():
    m = Q40Container()
    got = _q4_0_pool()[NCH * CH:].reshape(NCH, CH)
    src = m.raw(Q4_NAME.replace("{l}", "0")).reshape(NCH, CH)
    assert np.array_equal(got, src[pack.std_perm(NCH, IN_DIM)])


def test_the_numpy_and_cpp_packers_agree_on_a_q4_0_container():
    assert _fnv1a(_q4_0_pool().tobytes()) == Q40_POOL_FNV1A, \
        "the FLM Q4_0 pool changed; update src/open_qwen36/pools_test.cpp too"


def test_q8_perm_refuses_a_q4_0_source():
    m = Q40Container()
    with pytest.raises(ValueError) as e:
        pack.q8_chunks_of(m, Q40_NAME.replace("{l}", "0"), m.raw(Q40_NAME.replace("{l}", "0")))
    msg = str(e.value)
    assert "gate_proj" in msg and "4608" in msg and "8704" in msg


def test_a_chunk_width_that_is_none_of_the_four_is_still_refused():
    m = Q40Container()
    with pytest.raises(ValueError) as e:
        pack.q4_chunks_of(m, BAD_NAME.replace("{l}", "0"), m.raw(BAD_NAME.replace("{l}", "0")))
    msg = str(e.value)
    assert "up_proj" in msg and "1280" in msg and "4608" in msg


# ------------------------------------------------------------------- the spec's view
def test_a_4608_container_derives_the_q4_1_role_map():
    """The chunk size names the FORMAT the kernels stream, and this one is transcoded to
    q4_1, so the derived role map is empty and the spec hash does not move -- the property
    that lets a container this packer has never seen run an existing kernel set."""
    from recipes.spec import quant_map_from_chunk_sizes
    names = {Q40_NAME.replace("{l}", "0"): 4608}
    assert quant_map_from_chunk_sizes("hunyuan", names) == {}


# ------------------------------------------------------------------- the replica reader
def test_dq_tile_reads_the_pool_by_default_and_the_container_on_request():
    """`dq_tile` mirrors the packer's decision as it does for Q4_K: by default a 4608-byte
    tensor reads as the transcoded q4_1 the NPU holds, `requant=False` reads the
    container's own values. Here the two are the same weights, which is the point."""
    from q4nx import CHUNK_Q40, Q4NX, dq_chunks_q4_0, dq_chunks_q4_1

    class _Bare:
        requant_q8 = True
        def _refuse(self, name, cb):                     # pragma: no cover - not reached
            raise AssertionError(cb)

    src = _q4_0_vector(NCH)
    raw = src.reshape(-1).tobytes()
    W = Q4NX.dq_tile(_Bare(), raw, OUT_DIM, IN_DIM, CHUNK_Q40, True)
    W0 = Q4NX.dq_tile(_Bare(), raw, OUT_DIM, IN_DIM, CHUNK_Q40, False)
    want = dq_chunks_q4_1(pack.q4_0_4608_to_q4_1(src)).reshape(-1, 32, 256)
    ncol = IN_DIM // 256
    ref = np.empty((OUT_DIM, IN_DIM), np.float32)
    for f in range(want.shape[0]):
        ref[32 * (f // ncol): 32 * (f // ncol) + 32, 256 * (f % ncol): 256 * (f % ncol) + 256] = want[f]
    assert np.array_equal(W, ref)
    assert np.array_equal(W0, ref)
    assert np.array_equal(dq_chunks_q4_0(src).reshape(-1, 32, 256), want)
