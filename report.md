# GPU Wavelets (JPEG2000-style MRA) — Report

**Course:** GPU Computing (Master), bonus project.
**Transform:** floating-point CDF-5/3-style lifting, ported from the provided
`wavelets/1D_wavelets.cpp` reference.

> **Measurements.** All timings below were produced by running
> `wavelets_colab.ipynb` on a Colab **T4** (sm_75, CUDA 13.0). Methodology: each configuration is
> warmed up 5× then averaged over 30 timed runs with `cudaEvent` timers.
> "Useful GB/s" models a full-plane pass as one read + one write of the plane
> (`2·W·H·sizeof(T)` bytes) — a throughput proxy, so read the **before/after
> ratios** as the signal, not the absolute number. Register/shared-memory counts
> come from the `--ptxas-options=-v` build log (captured in the notebook).

---

## Items implemented

| # | Item | Pts | Where | Status |
|---|------|----:|-------|--------|
| 1 | 1D GPU fwd/inv | 10 | `src/wavelet.cuh`, `src/wavelet_gpu.cuh` | ✅ validated vs CPU oracle, n=16…16M |
| 2 | 2D separable | 10 | `src/wavelet2d.cuh` | ✅ round-trip validated |
| 3 | MRA (multi-level) | 15 | `src/wavelet2d.cuh` | ✅ multi-level + non-square |
| 4 | Image I/O | 5 | `src/image_io.hpp` (+stb) | ✅ PNG load/save, `test_io` |
| 5 | Data layout | 10 | `src/morton.hpp`, bench | ✅ discussion + Morton stencil measurement |
| 6 | Coalesced column pass | 10 | `src/wavelet_coalesced.cuh` | ✅ transpose path + bandwidth A/B |
| 7 | Shared-memory tiling | 15 | `src/wavelet_tiled.cuh` | ✅ whole-row tile + traffic A/B |
| 8 | Multi-res access kernel | 20 | `src/mra_access.cuh` | ✅ level-L region extraction, `test_access` |
| 9 | 3D extension | 10 | `src/wavelet3d.cuh` | ✅ 3D MRA round-trip, `test_3d` |
| 10 | FP16 path | 5 | templated kernels + `main.cu` | ✅ `__half` MRA, `test_fp16` |
| 11 | Performance study | 10 | `bench/bench.cu` | ✅ tables below |
| 12 | CI + reproducibility | 10 | `.gitlab-ci.yml` | ✅ build stage; GPU test stage |
| 13 | CLI | 5 | `src/main.cu` | ✅ `wavelet --input … --levels …` |
| 14 | Code quality & tests | 10 | `tests/`, headers | ✅ 6 test binaries, modular headers |

---

## 1–3 · Core transform (1D, 2D separable, MRA)

**Design — step-per-launch.** The lifting transform is a fixed sequence of
steps (deinterleave, predict1, update, predict2). Each step reads what the
previous wrote, so there is a cross-step data dependency. We make **each step one
kernel launch**; the implicit global barrier between launches resolves the
dependency. This mirrors the CPU passes exactly, has **no signal-size limit**,
and keeps every kernel trivially correct — the baseline against which the
optimised paths (items 6, 7) are compared.

**Strided reuse.** Every step kernel operates on a *strided logical signal*
(`element i at base[i·stride]`). One set of kernels therefore serves row passes
(`elem_stride=1`), column passes (`elem_stride=W`), and MRA sub-bands (the
sub-image shrinks each level; the image stride `W` never changes). Getting
"sub-region shrinks, stride stays" right is the classic MRA indexing bug the
brief warns about.

**Correctness.** `test_1d` validates GPU forward coefficients against the CPU
oracle and round-trip error across n = 16 … 2²⁴. `test_2d` validates 2D + MRA
round-trip across sizes and level counts, non-square included. Reconstruction is
FP-exact only to `WAVELET_RECON_TOL = 1e-4` (fp32) — the transform is floating
point, not integer lifting.

All six test binaries pass (`ctest` 6/6). Round-trip error stays below the
`1e-4` tolerance for every size/level; as a concrete end-to-end check the CLI
reconstructs the real `1024×1023` castle image (padded to `1024×1024`, 3 levels)
with **max_err = 0.000e+00** in fp32 — bit-exact here because 8-bit integer
pixel values fall on representable fp32 values.

### MRA throughput
_Tesla T4, warmup=5, mean of 30 runs._

| Size | time (ms) | Melem/s |
|---|---|---|
| 256² | 0.069 | 956 |
| 512² | 0.180 | 1453 |
| 1024² | 0.668 | 1571 |
| 2048² | 2.684 | 1563 |
| 4096² | 10.326 | 1625 |

Throughput saturates around **1.6 Gelem/s** past 1024² — the transform is
memory-bound, so once the launch overhead is amortised the kernels sit at the
DRAM-bandwidth ceiling regardless of size.

---

## 6 · Coalesced column pass

The row pass is already coalesced (`elem_stride=1`). The **column pass** in the
baseline uses `elem_stride=W`: consecutive threads touch addresses `W` apart, so
each warp scatters across cache lines. Fix (`src/wavelet_coalesced.cuh`):
**transpose → row pass → transpose back**, using a shared-memory tiled transpose
with a padded tile (`[32][33]`) to avoid shared-bank conflicts. The column data
is then read with `elem_stride=1`.

Both variants produce identical results on a full plane; the bench times them
head-to-head.

Correctness: naive vs transpose max diff at 1024² = **0.00e+00** (identical).

| Size | naive (ms) | naive GB/s | transpose (ms) | transp GB/s | speedup |
|---|---|---|---|---|---|
| 512² | 0.103 | 20.4 | 0.029 | 71.4 | **3.50×** |
| 1024² | 0.374 | 22.4 | 0.189 | 44.4 | **1.98×** |
| 2048² | 1.696 | 19.8 | 0.888 | 37.8 | **1.91×** |
| 4096² | 8.075 | 16.6 | 3.556 | 37.7 | **2.27×** |

**Discussion.** The transpose path wins **1.9–3.5×** at every size. The naive
strided column pass is stuck near **17–22 GB/s** — a small fraction of the T4's
~320 GB/s — because each warp's `elem_stride=W` loads scatter across cache
lines. The transpose replaces that with coalesced accesses and reaches
**38–71 GB/s** despite doing *more* total work (two extra plane passes for the
transposes). So the coalescing win dwarfs the transpose overhead. The absolute
GB/s is still below peak because the transpose itself is bandwidth-bound and we
now move the plane several times; a fused transposed-load would close that gap.

---

## 7 · Shared-memory tiling

`src/wavelet_tiled.cuh` gives each **row** to one block, stages it in shared
memory, runs the **entire** forward lifting there, and writes once. Global
traffic drops from ~5 plane read/writes (step-per-launch) to **1 read + 1
write**. Because the tile is the whole row, boundaries are exact and the result
matches the baseline row pass bit-for-bit; the win is purely traffic.

Constraint: the row must fit in shared memory (checked at launch;
`row_forward_shared` returns false and the bench prints `n/a` otherwise). Full
2D tiles with **halo exchange** — where results legitimately differ at tile
seams, as the brief notes — are the natural next step and are discussed as
future work; this version isolates the traffic reduction cleanly.

| Size | baseline (ms) | base GB/s | shared (ms) | shared GB/s | speedup |
|---|---|---|---|---|---|
| 512² | 0.020 | 106.1 | 0.007 | 313.5 | **2.95×** |
| 1024² | 0.117 | 71.9 | 0.019 | 438.9 | **6.10×** |
| 2048² | 0.583 | 57.5 | 0.141 | 238.5 | **4.15×** |
| 4096² | 2.308 | 58.2 | 0.580 | 231.6 | **3.98×** |

**Discussion.** Staging the whole row in shared memory gives **3–6×**. The
step-per-launch baseline stalls at **57–106 GB/s** because it streams the plane
through global memory ~5 times (deinterleave + 3 steps + copy-back); the shared
version touches global memory exactly twice and reaches **230–440 GB/s**. The
512²/1024² figures exceed the ~320 GB/s DRAM peak, which is the tell that at
those sizes the plane fits comfortably in L2 and part of the "useful GB/s" is
served from cache rather than DRAM — so treat the numbers as an *effective*
throughput and the **speedup ratio** as the real result. `k_row_forward_shared`
uses 18 registers + 1 barrier + `W·4` bytes dynamic shared (per cell 4/12 ptxas
log), so occupancy stays high.

---

## 8 · Multi-resolution access kernel

After `mra_forward`, the level-L LL band occupies the top-left `(W≫L)×(H≫L)`
corner at full stride `W`. `extract_region` (`src/mra_access.cuh`) gathers an
arbitrary `(w×h)` region of that band into a **compact** buffer (row stride `w`)
with a single coalesced kernel, then one `cudaMemcpy` to the host — the "cheap
coarse view of a gigantic image" MRA sells. `test_access` validates extracted
pixels against the resident image across levels, offsets, and non-square sizes.

---

## 9 · 3D extension

`src/wavelet3d.cuh` extends the separable scheme to volumes (W×H×D, strides
x:1, y:W, z:W·H). A 3D level is three 1D passes (x, y, z), each **reusing** the
strided `pass_forward/pass_inverse`. Passes over a sub-volume loop across
orthogonal slices so every `(line_stride, elem_stride)` pair is a real
arithmetic stride — the same discipline as 2D MRA, extended one axis. MRA
recurses into the **LLL octant**. `test_3d` round-trips cube and non-cube
volumes over multiple levels within `1e-4`.

---

## 10 · FP16 path

The step kernels are templated on the element type, so the `__half` path is the
same code instantiated with `T=__half` (`cuda_fp16.h`); `main.cu`'s `--dtype
fp16` and `test_fp16` exercise it. Values are converted on the host, transformed
in half, converted back.

| Size | fp32 (ms) | fp16 (ms) | speedup |
|---|---|---|---|
| 512² | 0.213 | 0.223 | 0.96× |
| 1024² | 0.659 | 0.606 | 1.09× |
| 2048² | 2.941 | 2.687 | 1.09× |
| 4096² | 13.247 | 10.628 | **1.25×** |

**Discussion.** `test_fp16` passes within the loose half-precision tolerance
(< 5.0; fp16 keeps ~3 decimal digits, vs `1e-4` in fp32) — fine for a lossy
image pipeline, not for exact reconstruction. The speedup **grows with size**,
from 0.96× at 512² (kernel-launch overhead dominates, so smaller elements don't
help and even lose slightly) to **1.25×** at 4096². It falls short of the naive
2× because these kernels do scalar `__half` loads: to halve traffic *and* keep
the memory system busy you need vectorised `__half2` loads (which would also
matter for shared-memory bank conflicts). That `__half2` packing is the clear
next optimisation.

---

## 5 · Data layout discussion

Row-major is ideal for the row pass and terrible for the column pass (stride
`W`), which items 6/7 address at the algorithm level. An alternative is to change
the **storage** so 2D neighbourhoods are contiguous: a **Morton (Z-order)**
layout (`src/morton.hpp`) interleaves x/y bits. The bench measures a
vertical-neighbour stencil (the layout-sensitive access) in row-major vs Morton:

| Size | row-major (ms) | Morton (ms) | speedup |
|---|---|---|---|
| 512² | 0.0051 | 0.0070 | 0.74× |
| 1024² | 0.0366 | 0.0443 | 0.83× |
| 2048² | 0.1404 | 0.1750 | 0.80× |
| 4096² | 0.5569 | 0.7180 | 0.78× |

**Discussion — Morton loses (0.74–0.83×).** For this access pattern the T4's L2
already absorbs the adjacent-row reuse of a row-major vertical stencil, so the
row-major version is memory-efficient *and* has trivial indexing. Morton adds
per-access bit-spreading (`part1by1`) on the critical path and buys no locality
the cache wasn't already giving — net loss. This is a legitimate "no clear
winner" result (the brief only asks for a good discussion). Morton would matter
where the working set genuinely exceeds L2 — deep MRA pyramids over very large
images, or true 2D-block access — but at these sizes the arithmetic cost
dominates. A **tiled/block-linear** layout is the pragmatic middle ground:
neighbourhood locality without per-access bit math.

---

## 11 · Performance study

Tables above (MRA throughput, dtype, column, tiled, layout) constitute the
study: bandwidth across sizes, dtypes, and the before/after optimisation
comparisons. Register and shared-memory usage per kernel:

From the `--ptxas-options=-v` build log (sm_75):

| Kernel | Registers | Barriers | Shared |
|---|---|---|---|
| `k2_fwd_predict1` / `k2_fwd_update` / `k2_inv_*` | 10 | 0 | — |
| `k2_fwd_predict2` | 14 | 0 | — |
| `k2_interleave` / `k2_deinterleave` | 15 | 0 | — |
| `k2_copy` | 8 | 0 | — |
| `k_row_forward_shared` (item 7) | 18 | 1 | `W·4` B dynamic |

All step kernels use **≤15 registers** with **no stack/spills**, so occupancy is
register-unconstrained — consistent with a memory-bound workload where the goal
is threads-in-flight to hide DRAM latency, not compute.

**Limiting factors.** The transform is **memory-bound**: arithmetic intensity is
a few flops per element touched, far below the T4 ridge point, so effective
bandwidth (not FLOPs) governs. Evidence: MRA throughput flat-lines at ~1.6
Gelem/s once launch overhead is amortised; the naive column pass is pinned at
~20 GB/s by uncoalesced loads; and both optimisations (coalescing, tiling) win by
*removing traffic*, not adding compute. The step-per-launch baseline is
additionally **launch- and traffic-bound** (many kernels, ~5 redundant global
passes), which is exactly what items 6/7 target.

---

## 12 · CI + reproducibility

`.gitlab-ci.yml`: a `build` stage compiles everything on a stock
`nvidia/cuda:*-devel` image (no GPU needed — nvcc compiles device code without a
device), catching regressions on every push. A `test:gpu` stage runs the
deterministic `ctest` suite (no NaNs, near-perfect reconstruction) on a
GPU-tagged runner; where no GPU runner exists it is `allow_failure` and
reproduced via the Colab notebook instead.

---

## 13 · CLI

```
wavelet --input in.png --output out.png [--levels N] [--dtype fp32|fp16] [--forward]
```
Default mode round-trips (forward then inverse), writes the reconstruction, and
prints max error vs input — the end-to-end proof on a real image. `--forward`
writes the MRA coefficient image (the Figure-1b representation). The image is
edge-padded to a multiple of `2^levels` and cropped back.

---

## 14 · Code quality & tests

Six deterministic test binaries (`test_1d/2d/access/3d/fp16/io`) wired into
`ctest`, each returning non-zero on failure and checking finiteness +
reconstruction tolerance. Kernels and host orchestration are split into small,
documented headers (`wavelet*.cuh`), reused across 1D/2D/3D/MRA via the strided
abstraction rather than duplicated.

---

## Failed attempts / findings

**Boundary stencil bounds the level count.** The 3D round-trip test initially
failed at `64×32×16`, levels=3 (reconstruction error ~17). Root cause: the
lifting boundary formula `predict2` reads `s[n2-3]`, which underflows for
`n2 < 3` — i.e. any transformed dimension below **n=8**. At `64×32×16` the depth
axis shrinks `16→8→4` over three levels, hitting n=4 and reading out of bounds.
This is a property of the *provided* transform (the CPU baseline shares it; its
1D test never goes below n=16), not a 3D-specific bug. Consequence: the valid
level count is bounded by `min(W,H,D) >> (levels-1) >= 8`. Cube tests (min n=8)
pass; the 3D test now uses `128×64×32`, whose deepest level `32×16×8` stays in
range. Fixing it "properly" would mean redefining the boundary handling and
diverging from the reference oracle — out of scope.

**Morton layout did not pay off (0.74–0.83×).** The hypothesis was that Z-order
storage would speed up the layout-sensitive vertical-neighbour access; measured,
it was consistently *slower* than row-major because the T4's L2 already caches
the adjacent row and Morton only adds bit-spreading arithmetic. A useful negative
result: locality optimisations lose when the cache already provides the locality.

**fp16 didn't reach the naive 2×** (peaked at 1.25×): scalar `__half` loads
halve the element size but don't saturate the memory system; `__half2`
vectorisation is needed. The coalescing (1.9–3.5×) and shared-tiling (3–6×) wins,
by contrast, matched expectations — both are pure traffic-reduction plays, which
is where a memory-bound kernel has the most to gain.
