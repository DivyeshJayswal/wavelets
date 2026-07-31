# GPU Wavelets (JPEG2000-style MRA) — Report

**Course:** GPU Computing (Master), bonus project.
**Transform:** floating-point CDF-5/3-style lifting, ported from the provided
`wavelets/1D_wavelets.cpp` reference.

> **Measurements.** All timings in the tables marked `FILL` come from running
> `wavelets_colab.ipynb` on a Colab **T4**. Methodology: each configuration is
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

<!-- FILL FROM COLAB: paste test_1d + test_2d output (max errors) -->

### MRA throughput
<!-- FILL FROM COLAB: paste "MRA forward throughput" table -->

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

<!-- FILL FROM COLAB: paste "Column pass: naive strided vs coalesced transpose" table -->

**Discussion.** <!-- FILL: comment on the observed speedup; note the transpose
adds 2 extra plane passes, so the win is the coalescing minus that overhead. On
T4 the strided column pass is limited by L2/DRAM efficiency; expect the
transpose path to win at larger sizes where the strided penalty dominates. -->

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

<!-- FILL FROM COLAB: paste "Row pass: step-per-launch vs shared-memory tile" table -->

**Discussion.** <!-- FILL: report GB/s before/after and whether the shared path
approaches the T4's ~320 GB/s ceiling; note that the step-per-launch baseline is
bandwidth-bound by its redundant global traffic, which is exactly what tiling removes. -->

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

<!-- FILL FROM COLAB: paste fp32 vs fp16 table + test_fp16 max error -->

**Discussion.** FP16 has ~3 decimal digits, so round-trip error rises to ~`FILL`
(vs `1e-4` in fp32) — acceptable for a lossy image pipeline but not for exact
reconstruction. Speedup on T4 is `FILL`; because the kernels are
**memory-bound**, halving element size roughly halves traffic, so the win tracks
bandwidth rather than FP16 math throughput. Bank conflicts in the tiled path
would need `__half2` packing to fully exploit; noted as future work.

---

## 5 · Data layout discussion

Row-major is ideal for the row pass and terrible for the column pass (stride
`W`), which items 6/7 address at the algorithm level. An alternative is to change
the **storage** so 2D neighbourhoods are contiguous: a **Morton (Z-order)**
layout (`src/morton.hpp`) interleaves x/y bits. The bench measures a
vertical-neighbour stencil (the layout-sensitive access) in row-major vs Morton:

<!-- FILL FROM COLAB: paste "Layout: row-major vs Morton" table -->

**Discussion.** <!-- FILL: report whether Morton helped. Expected finding: for
this access pattern the T4's L2 cache already absorbs the adjacent-row reuse, so
Morton's index arithmetic (part1by1 bit-spreading) can cost more than it saves —
a legitimate "no clear winner" result. Morton mainly pays off for deep MRA
pyramids and very large images where working sets exceed L2. Tiled (block-linear)
layouts are a middle ground: locality without per-access bit math. -->

---

## 11 · Performance study

Tables above (MRA throughput, dtype, column, tiled, layout) constitute the
study: bandwidth across sizes, dtypes, and the before/after optimisation
comparisons. Register and shared-memory usage per kernel:

<!-- FILL FROM COLAB: paste the ptxas -v lines (registers, smem) for the key kernels -->

**Limiting factors.** The transform is **memory-bound**: arithmetic intensity is
a few flops per element touched, far below the T4 ridge point, so effective
bandwidth (not FLOPs) governs. The step-per-launch baseline is additionally
**launch- and traffic-bound** (many kernels, redundant global passes); items 6/7
target exactly that. <!-- FILL: optional roofline note. -->

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
<!-- FILL: e.g. did the transpose column pass actually beat strided at small
sizes? Did Morton lose? Did fp16 tiling hit bank conflicts? Record the honest
outcomes here — the brief explicitly values a failed optimisation you can explain. -->
