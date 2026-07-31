# GPU Wavelets (JPEG2000-style MRA)

GPU-accelerated separable wavelet transform and multi-resolution analysis (MRA),
ported from the provided 1D CPU reference (`wavelets/1D_wavelets.cpp`). The
transform is the baseline's floating-point CDF-5/3-style lifting scheme.

## Build & run

### Google Colab (GPU runtime)
Open `wavelets_colab.ipynb`, set the runtime to GPU, and run all cells. It
detects the GPU arch, builds with CMake, runs the test suite, the CLI image
demo, and the benchmarks (writing `bench_results.md`).

### Local (needs CUDA toolkit + CMake ≥ 3.18)
```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75   # 75=T4, 80=A100, 86=RTX30xx...
cmake --build build -j
cd build && ctest --output-on-failure
./bench                                          # performance tables (item 11)
./wavelet --input ../assets/Castle_Lichtenstein.jpg --output recon.png --levels 3
```

> **Submission:** this repo is not yet a git repo. `git init && git add -A &&
> git commit` then push to LRZ GitLab and open a Merge Request (see the brief).
> `.gitlab-ci.yml` builds on push; the GPU test stage needs a GPU runner.

## Items implemented (see `report.md` for the full write-up)

| Item | File(s) | Status |
|------|---------|--------|
| CPU oracle | `src/wavelet_cpu.hpp` | validation ground truth |
| 1 · 1D GPU fwd/inv | `src/wavelet.cuh`, `src/wavelet_gpu.cuh`, `tests/test_1d.cu` | done |
| 2 · 2D separable | `src/wavelet2d.cuh`, `tests/test_2d.cu` | done |
| 3 · MRA | `src/wavelet2d.cuh`, `tests/test_2d.cu` | done |
| 4 · Image I/O | `src/image_io.hpp` (+`third_party/stb_*`), `tests/test_io.cu` | done |
| 5 · Data layout | `src/morton.hpp`, `bench/bench.cu` | done — discussion + Morton stencil |
| 6 · Coalesced column pass | `src/wavelet_coalesced.cuh`, `bench/bench.cu` | done — transpose A/B |
| 7 · Shared-memory tiling | `src/wavelet_tiled.cuh`, `bench/bench.cu` | done — whole-row tile A/B |
| 8 · Multi-res access | `src/mra_access.cuh`, `tests/test_access.cu` | done |
| 9 · 3D extension | `src/wavelet3d.cuh`, `tests/test_3d.cu` | done |
| 10 · FP16 path | templated kernels, `src/main.cu`, `tests/test_fp16.cu` | done |
| 11 · Performance study | `bench/bench.cu` | done — run on Colab |
| 12 · CI | `.gitlab-ci.yml` | build stage + GPU test stage |
| 13 · CLI | `src/main.cu` | done |
| 14 · Code quality & tests | `tests/`, headers | 6 test binaries |

## Design notes
- **Step-per-launch:** each lifting step is one kernel; the global barrier between
  launches resolves the cross-step data dependency. Mirrors the CPU passes exactly,
  no signal-size limit.
- **Strided signals:** the same step kernels serve rows (`elem_stride=1`), columns
  (`elem_stride=W`), and MRA sub-bands (sub-image shrinks, image stride `W` stays).
- **Reconstruction is FP-exact only to `~1e-4`** (fp32), not bit-exact — the
  transform is floating point, not integer lifting.
