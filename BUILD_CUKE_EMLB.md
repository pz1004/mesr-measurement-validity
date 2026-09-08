# Building the cuke-emlb classical denoisers

Status: **built and verified, 2026-07-26.** All six classical denoisors run through
`dataset_assessment/denoisors.py`.

## What this unlocks

`dv_toolkit` plus `dwf`, `evflow`, `knoise`, `red`, `ts`, `ynoise` from
`cuke-emlb/configs/denoisors.py`. Without them the benchmark has only the `raw` row and the
Task 9 rank-agreement gate is undefined.

## Environment

| | |
|---|---|
| OS | Ubuntu 22.04 (jammy) under WSL2, kernel 6.6.87.2, 20 cores |
| CMake | 3.22.1 |
| Compiler used | **g++-13 / gcc-13 (13.4.0)** — *not* the default `c++` (11.4.0) |
| Python | 3.13.9 (Anaconda at `/home/pz1004/anaconda3`) |
| dv-processing | 2.0.3 |
| dv-runtime | 1.7.2-1~jammy |

## The recipe that worked

### 1. Check out the missing submodule

`cuke-emlb/external/dv-toolkit/` shipped **empty** and `cuke-emlb/external/CMakeLists.txt`
does `add_subdirectory(dv-toolkit)`, so CMake failed before any dependency check.

`cuke-emlb` *is* a git repository and does declare the submodule, so the canonical fix looks
like `git submodule update --init --recursive`. That does not work here, because
`.gitmodules` gives an **SSH** URL and this machine has no GitHub SSH authentication:

```
[submodule "external/dv-toolkit"]
	path = external/dv-toolkit
	url = git@github.com:KugaMaxx/yam-toolkit.git
```

```
ssh_askpass: exec(/usr/bin/ssh-askpass): No such file or directory
Host key verification failed.
fatal: Could not read from remote repository.
```

The workaround is to clone the same repository over HTTPS:

```bash
cd cuke-emlb
git clone --recursive https://github.com/KugaMaxx/yam-toolkit.git external/dv-toolkit
```

This also supplies `python/external/pybind11` (v2.11.1); pybind11 is not installed as a
Python package on this machine.

#### ⚠️ Do not use the pinned submodule commit — it does not work (tested 2026-07-29)

`cuke-emlb` pins `external/dv-toolkit` to **`486c70e053c562d5d397a5b42cccb96a8e224460`**
(`git ls-tree HEAD external/dv-toolkit`). A plain clone lands on upstream HEAD, which here is
**`0fe3b3832187c07e09069d1df4006f4682268ab5`**. Everything in this document, and every number
in `results/`, was produced against `0fe3b38`.

We built the pinned commit to check. It fails, for **two independent reasons**.

**1. Packaging (workaroundable).** `486c70e` ships **no `pyproject.toml`**, so pip's isolated
build environment contains no NumPy, while `python/CMakeLists.txt:1` requires the NumPy
component:

```
Could NOT find Python3 (missing: Python3_NumPy_INCLUDE_DIRS NumPy) (found version "3.13.9")
Call Stack (most recent call first):
  /usr/share/cmake-3.22/Modules/FindPython3.cmake:490 (include)
  python/CMakeLists.txt:1 (find_package)
```

`0fe3b38` added the file with `requires = ["setuptools>=42", "wheel", "numpy>=1.24"]`.
Passing `--no-build-isolation` gets past this and installs `dv_toolkit 0.1.3`.

**2. Binding (fatal, no workaround).** Once installed, the very first call into the toolkit
raises:

```
TypeError: Unable to convert function return value to a Python type! The signature was
	(self: dv_toolkit.lib._lib_toolkit.EventStorage, timestamp: int, x: int, y: int,
	 polarity: bool) -> dv::Event
```

That is `EventStorage.push_back`. At `486c70e` it returns `dv::Event`, for which no Python
conversion is registered, so the adapter dies while *filling* the storage — before any
denoising runs. All six classical methods are unusable. `0fe3b38` is the fix:

> `mst(!import): update binding method --- Fix pybind11 type binding issue where
> EventStorage() method could not convert dv::Event return value to Python type.`

Note the failure is on the **write** path (`push_back`), not the read path
(`generateEvents().numpy()`) — an earlier draft of this document guessed the latter from the
commit diff, which showed lambdas wrapping `at`/`__getitem__`/`front`/`back`. `push_back` is
affected too. Right conclusion, wrong mechanism; the measurement settled it.

**Reproduce this build exactly:**

```bash
cd cuke-emlb/external/dv-toolkit && git checkout 0fe3b3832187c07e09069d1df4006f4682268ab5
```

Verified after restoring: all six methods return byte-identical score vectors to the run that
produced `results/` — native retentions `dwf 0.65541`, `evflow 0.03856`, `knoise 0.33032`,
`red 0.215975`, `ts 0.240655`, `ynoise 0.47873` on the first 200,000 events of
`DND21/1hz_hotel-bar`, with matching SHA-1 over each score array.

### 2. System packages

```bash
sudo apt-get install -y libopenblas-dev libboost-dev dv-runtime-dev
```

Already present beforehand: `dv-processing` 2.0.3, OpenCV 4.5.4, Eigen 3.4.0, `libfmt-dev`
9.1.0, `liblz4-dev`, `libzstd-dev`, and `gcc-13`/`g++-13`.

### 3. Configure — two overrides are mandatory

```bash
cd cuke-emlb && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE \
         -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
         -DPython3_ROOT_DIR=/home/pz1004/anaconda3 \
         -DPython3_EXECUTABLE=/home/pz1004/anaconda3/bin/python3 \
         -DPython3_FIND_STRATEGY=LOCATION
```

Both overrides are required, and each was found by hitting its error:

**(a) gcc 11 is rejected by dv-processing itself.**

```
CMake Error at /usr/lib/x86_64-linux-gnu/cmake/dv-processing/dv-processing-config.cmake:154 (MESSAGE):
  The gcc compiler version is not supported (11.4.0), please use gcc compiler
  of at least version 13.0
Call Stack (most recent call first):
  /usr/lib/x86_64-linux-gnu/cmake/dv/dv-config.cmake:19 (FIND_PACKAGE)
  CMakeLists.txt:53 (find_package)
```

`gcc-13` was already installed; only the `cc`/`c++` alternatives point at 11. Installing the
package is not enough — the compiler must be named explicitly.

**(b) CMake finds the system Python, not the conda one.**

```
CMake Error at /usr/share/cmake-3.22/Modules/FindPackageHandleStandardArgs.cmake:230 (message):
  Could NOT find Python3 (missing: Python3_INCLUDE_DIRS Development.Module
  NumPy) (found version "3.10.12")
Call Stack (most recent call first):
  /usr/share/cmake-3.22/Modules/FindPython/Support.cmake:3180 (find_package_handle_standard_args)
  /usr/share/cmake-3.22/Modules/FindPython3.cmake:490 (include)
  python/CMakeLists.txt:2 (find_package)
```

System Python 3.10.12 has no dev headers. The extension must be built against the
interpreter that will import it (3.13.9), or the `.so` will not load.

### 4. Build and install

```bash
make -j"$(nproc)"                       # ~1 min on 20 cores
cd .. && CC=gcc-13 CXX=g++-13 pip install external/dv-toolkit/.
```

`make` writes six modules into `cuke-emlb/modules/python/`:

```
double_window_filter.cpython-313-x86_64-linux-gnu.so
event_flow.cpython-313-x86_64-linux-gnu.so
khodamoradi_noise.cpython-313-x86_64-linux-gnu.so
reclusive_event_denoisor.cpython-313-x86_64-linux-gnu.so
time_surface.cpython-313-x86_64-linux-gnu.so
yang_noise.cpython-313-x86_64-linux-gnu.so
```

`pip` builds and installs `dv_toolkit-0.2.0-cp313-cp313-linux_x86_64.whl`.

### 5. Verify

```bash
python -c "import sys; sys.path.insert(0,'cuke-emlb'); import dv_toolkit; from configs.denoisors import knoise; print(type(knoise((346,260), {})).__name__)"
```

```
knoise
```

End to end, on the first 200,000 events of `DND21/1hz_hotel-bar`
(`dataset_assessment.denoisors.score_events`, native operating point = share the method keeps):

| method | native retention | wall |
|---|---|---|
| raw | 1.0000 | 0.00 s |
| dwf | 0.6554 | 0.11 s |
| evflow | 0.0386 | 0.14 s |
| knoise | 0.3303 | 0.10 s |
| red | 0.2160 | 0.44 s |
| ts | 0.2407 | 0.11 s |
| ynoise | 0.4787 | 0.10 s |

Every classical method lands strictly between 0 and 1, which is what Task 7 Step 2 requires.

## Two API facts the plan got wrong

1. **The class is `dv_toolkit.EventStorage`, not `EventStore`.** `EventStore` is
   dv-processing's type, reachable via `EventStorage.toEventStore()`; the denoisors'
   `accept()` takes the toolkit storage. Authoritative usage is `cuke-emlb/eval_denoisor.py`.
2. **`generateEvents()` returns an ordered subsequence of the input**, and
   `.numpy()` exports it as a `(timestamp, x, y, polarity)` structured array. Recovering
   which input events survived is therefore an O(n) forward merge, not a set lookup. This
   matters: `DND21/1hz_hotel-bar` contains 54 groups of events sharing a `(t, x, y)`
   coordinate that a `(t, x, y)` set cannot disambiguate, and a set cannot recover indices
   at all. Measured on 897,000 events: 0.63 s to fill the storage, 0.03 s to denoise,
   0.11 s to merge (198,550/198,550 matched, indices strictly increasing).

## Caveats

- `CMakeLists.txt:29` sets `-Ofast -march=native`. These binaries are not portable to
  another CPU — rebuild rather than copying `modules/`.
- CMake warns that conda's `libssl.so.3`, `libcrypto.so.3`, `liblz4.so.1` and `libzstd.so.1`
  may shadow the system copies at runtime. Nothing has misbehaved, but if a denoisor starts
  crashing on load, suspect this first.
- `cuke-emlb/modules/net/` holds only `.gitkeep`, so `edncnn` and `mlpf` remain unavailable.
  That matches the plan's §0.3 stretch-goal note.
- Nothing under `cuke-emlb/` was modified. The additions are the `external/dv-toolkit/`
  checkout, the `build/` directory, and the six `.so` files in `modules/python/`.
