# vLLM Ascend PR 17392 Environment

Start with [the vLLM-Ascend README](../README.vllm-ascend.md) for the retained
implementation, measured results, and validation limits. The committed
`vllm-ascend.lock.json` and `scripts/prepare-vllm-ascend-sources.py` restore the
pinned archives and all patches. Run the preparation command only when the
upstream directories do not exist; existing working sources are never replaced.

## Source Layout

- `vllm-ascend/`: official source snapshot at
  `fcc49fe2520cea586efc59ac04d9a57442bfef0e` (PR 17392 head).
- `vllm/`: official source snapshot at
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, cited by the PR.
- `vllm-ascend/csrc/third_party/catlass/`: Git checkout at
  `41bf90da655bba3c66d0acd7e00abe33960ecfd6`, pinned by `.gitmodules`.
- `.venv-vllm-ascend/`: isolated Python environment; no system site packages.

The two top-level upstream directories are source archives, not Git clones;
GitHub Git transport was unavailable. They are ignored by the ECHO repository.
The original archives and provenance are retained under `build/reference/`.

## Environment Entry

```bash
source scripts/vllm-ascend-env.sh
# Or run one command without modifying the calling shell:
./scripts/vllm-ascend-run.sh python --version
```

Uses project-local CANN 9.1.0 at `build/cann-vllm/cann`, Python 3.11,
PyTorch 2.10.0 and torch-npu 2.10.0.post4. This does not alter `.venv-npu`
or the existing CANN 8.5 ECHO setup. `SOC_VERSION` defaults to
`ascend910_9391`. Override device selection explicitly with `ECHO_NPU_DEVICE`.
Import-only checks must not be confused with actual device or Graph validation.
The launcher places both upstream source roots on `PYTHONPATH` so the outer
`ECHO/vllm` directory does not shadow the editable `vllm` Python package.

## Isolated CANN Installation

The system CANN 9.1 installation was missing `libhccl.so`. A complete official
toolkit and A3 operator package were installed separately; no substitute library
symlinks, CANN 8.5 runtime mixing, system upgrades, or driver changes were used.
The existing SGLang environment remains unchanged.

Official installers retained in `build/cann-vllm-downloads/`:

```text
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-aarch64.run
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-nnal_9.1.0_linux-aarch64.run
```

Installation commands (run from the repository root):

```bash
bash build/cann-vllm-downloads/Ascend-cann-toolkit_9.1.0_linux-aarch64.run \
  --full --quiet --install-path="$PWD/build/cann-vllm"
bash build/cann-vllm-downloads/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run \
  --install --quiet --install-path="$PWD/build/cann-vllm"
bash build/cann-vllm-downloads/Ascend-cann-nnal_9.1.0_linux-aarch64.run \
  --install --quiet --whitelist=atb --install-path="$PWD/build/cann-vllm"
```

NNAL's ATB component is required for the model worker's Torch-NPU ATB plugin.
It is installed under `build/cann-vllm/nnal/atb` and sourced by the launcher;
the matching 9.1.0 version is used instead of the old ECHO ATB installation.

## Source Compatibility

An upstream dependency conflict was found: the cited vLLM source requires
FastAPI >=0.133.0,<0.137.0, while this vllm-ascend snapshot declares
FastAPI <0.124.0. Record any local compatibility adjustment and actual
validation below before describing the environment as ready.

Local compatibility adjustment: the FastAPI constraint in both
`vllm-ascend/requirements.txt` and `vllm-ascend/pyproject.toml` was changed
from `<0.124.0` to `>=0.133.0,<0.137.0`, matching the cited vLLM commit.
This is not an unmodified upstream snapshot. Runtime compatibility still
requires validation; package resolution alone is not sufficient.

Additional local adjustments:

- `vllm/requirements/common.txt`: OpenCV fixed to `4.11.0.86` instead of
  `>=4.13.0`. Triton Ascend requires NumPy `1.26.4`, while OpenCV 4.13+
  requires NumPy 2 for Python 3.11. Video/multimodal inference is not validated.
- `vllm-ascend/csrc/cmake/func.cmake`: operator-info JSON generation now
  depends on its generated INI inputs. Without this, reusing the build directory
  after switching from two operators to all operators retained stale metadata.
- `vllm-ascend/csrc/build_aclnn.sh`: make the prior vendor `scripts` directory
  owner-writable before replacement, as the installer makes it read-only.

## Validation Record (2026-10-23)

- Python 3.11.6 virtual environment created without system site packages.
- Installed PyTorch 2.10.0, torch-npu 2.10.0.post4, torchvision 0.25.0,
  torchaudio 2.10.0 and build tools.
- Launcher selects CANN 9.1.0 and resolves its `ccec` compiler correctly.
- Importing torch/torch-npu passes with the complete project-local CANN.
- A basic NPU computation, Graph capture, and two replays pass on physical
  device 6. The second replay changes the input and checks the changed output.
  Log: `build/reference/vllm-env-npu-graph.log`.
- Triton Ascend 3.2.2 is available from the Ascend index used in the upstream
  Dockerfile: `https://mirrors.huaweicloud.com/ascend/repos/pypi`.
  It and pytest 8.3.2 were installed successfully.
- The pinned vLLM is installed editable with `VLLM_TARGET_DEVICE=empty` and
  `VLLM_VERSION_OVERRIDE=0.0.0+ced6857`. Log: `build/reference/vllm-install.log`.
- Full dependency installation uses `--no-build-isolation` so arctic-inference's
  build does not install its separate PyTorch 2.7 build requirement. This alone
  does not establish arctic-inference runtime compatibility.
- Both PR fusion operators compiled and installed successfully. The complete
  Torch C++ extension and direct AscendC kernels also compiled and loaded;
  both fused operators are present in `torch.ops._C_ascend`.
  Logs: `build/reference/vllm-fused-ops-build-r3.log`,
  `vllm-fused-ops-install.log`, `vllm-native-build.log`,
  `vllm-native-import.log` in the same directory.
- The repeatable runtime smoke test passed on card 6 for batches 1/2/4/8,
  four replays each. Log: `build/reference/vllm-runtime-smoke.log`.
- The full A3 ACLNN custom-operator package compiled and installed successfully.
  Logs: `build/reference/vllm-all-ops-build-r2.log`, `vllm-all-ops-install.log`.
- The two upstream fused-operator test files passed: **106 passed, 6 deselected**.
  The excluded tests are the very-long-sequence matrix. Coverage includes mixed
  request/cache states, dtype/head variants, replacements/hits, and the fused
  attention test's Graph path. Log: `build/reference/vllm-fused-tests.log`.
- Virtual-environment-only `pip check` passes. Package versions are retained in
  `build/reference/vllm-environment-freeze.txt`; check output is in
  `build/reference/vllm-pip-check.log`.
- The extra indexer Graph check passes for batches 1/2, four replays each,
  switching states `-2, -1, -3, -1` and negating query data. Cache and all output
  tensors match eager execution exactly; output/cache addresses remain fixed.
  Log: `build/reference/vllm-indexer-graph.log`.
- Tiny local Qwen2 model inference passes on physical card 6. Decode Graphs
  for batches 1/2 were captured, and the log explicitly reports
  `ACL graph replay is active`. Four generation calls with batch sizes
  1/2/1/2 each returned six tokens per request.
  Log: `build/reference/vllm-model-smoke-r4.log`.
- This establishes single-device runtime, operator and tiny-model decode
  coverage only. DeepSeek V3.2 AWQ, end-to-end sparse KV offloading, prefill
  Graphs, multi-device/HCCL execution, prefetch and performance are not validated.

Repeat the basic runtime test with an available, healthy NPU:

```bash
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh \
  python scripts/smoke-vllm-ascend-runtime.py
```

This tests fixed-address buffers, batches 1/2/4/8, and four changed-input
replays per capture. It does not test sparse attention, KV offloading,
multi-device communication, or model generation.

The launcher adds a local `patch` executable under `build/vllm-tools/usr/bin`
for the upstream protobuf build. CMake is pinned to 3.31.10. Triton Ascend
3.2.2 probes the entire machine with `npu-smi info` during import; unhealthy
cards can cause this probe to take its 100-second timeout even when the
selected card is healthy.
CPU affinity initialization also queries the whole machine and can add startup
latency. The tiny token-ID-only model smoke logs a chat-template warmup warning
because tokenizer initialization is intentionally disabled; all four generation
checks still pass. It also warns that `/proc/irq` is not writable; no system IRQ
settings or privileges were changed. Optional CUDA DeepSelect is absent in this
`VLLM_TARGET_DEVICE=empty` build; Ascend-specific operator tests pass independently.

## Rebuild Commands

After creating the Python 3.11 virtual environment, installing Torch 2.10.0 /
torch-npu 2.10.0.post4, and installing the three CANN packages above:

```bash
./scripts/vllm-ascend-run.sh python -m pip install --no-build-isolation \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  -r vllm-ascend/requirements.txt -r vllm/requirements/common.txt \
  pytest attrs googleapis-common-protos setuptools-rust nanobind
./scripts/vllm-ascend-run.sh python -m pip install 'cmake==3.31.10'
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.0.0+ced6857 \
  ./scripts/vllm-ascend-run.sh python -m pip install \
  --no-build-isolation --no-deps -e ./vllm
./scripts/build-vllm-ascend-native.sh
```

The native build script compiles and installs the real Torch extension, direct
AscendC kernels, and all A3 ACLNN custom operators before registering the editable
Python plugin. Its final `COMPILE_CUSTOM_KERNELS=0` applies only to that Python
registration step to avoid building the same binaries twice. It is not a
kernel-disabled installation or a replacement for operator validation.

## Operator And Model Checks

Run these sequentially on one healthy, available NPU:

```bash
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh python -m pytest \
  --confcutdir=vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops \
  vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_lightning_indexer_manage.py \
  vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_scatter_copy_sparse_flash_attention.py \
  -k 'not long_sequence' -q
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh \
  python scripts/check-vllm-fused-indexer-graph.py
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh \
  python scripts/smoke-vllm-ascend-model.py
```

The operator command intentionally excludes the separate very-long-sequence
matrix. The indexer replay check compares Graph and eager output exactly while
changing request states and query data, keeping cache/output addresses fixed.
The model smoke needs the pre-existing local tiny Qwen2 model in
`build/npu-smoke-model`, runs batches 1/2 repeatedly, and generates six tokens
per request. Verify the model log reports active ACL Graph replay; successful
generation without replay is not enough.

For Python dependency consistency, check only the isolated virtual environment:

```bash
./scripts/vllm-ascend-run.sh env PYTHONPATH= python -m pip check
```

CANN adds its own Python distribution directory to `PYTHONPATH`; its bundled
optional `ms-service-profiler` has additional package/version requirements that
are not part of the isolated inference environment. Those profiler tools are
not validated by this setup or by the virtual-environment-only check.
