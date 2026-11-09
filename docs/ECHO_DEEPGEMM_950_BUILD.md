# ECHO DeepGEMM Ascend 950：偏移映射版编译与验证

仓库：`rooikeee/ECHO`，分支：`echo-deepgemm-950`。
本文对应带 **staging 偏移登记、跨 query 去重、每轮映射清理**的版本。
`06d5b9a` 是不含这些更新的初版；更新时应获取本分支最新提交，并重编 Host 扩展。

截至 2026-11-09：66 个 CPU 并发/状态场景、4 项 Python 接口测试通过；
**950 编译、设备正确性、Graph 重放和性能尚未验证**。下面是目标机执行步骤，
不是已在 950 跑通的结果记录。

## 1. 构建的是什么

这是 `deep_gemm` 包内的 **paged MQA 打分 + miss KV 预取**实验实现：

```text
每次调用：logits 初始化 → 清理上一轮 staging 映射/重置输出 → 生成调度 metadata

融合内核：512-token 分块打分
            → score > threshold
            → 原子查询/占用 miss host token，跨 block/query 去重
            → 分配有界 staging 槽位并登记偏移
            → 源 KV → UB 双缓冲 → staging HBM
            → 处理后续分块，结束时完成 outstanding DMA
```

输出包括 logits、各行预取数量、token/host ID、staging 字节和更新后的映射。
当前没有接入最终精确 TopK、TopK 后的 recall、正式缓存槽位分配或 LRU 淘汰，
也没有自动接到 SGLang 的生产 indexer 路径。

**构建分为两阶段：**

| 阶段 | 编译对象 | 触发方式 |
| --- | --- | --- |
| Host 扩展 | Python 包 `deep_gemm` 的 C++ 扩展 `_C`，包含上游接口和新增 ECHO 接口 | 构建脚本执行 `pip install` |
| 设备 JIT | cleanup、metadata、融合 MQA/预取内核 | 测试脚本首次实际调用 |

这不是 `ops-transformer` 的 ACLNN vendor 包。不要用原来的
`build_and_test_950.sh` 编译它；本入口不需要安装自定义 OPP `.run` 或设置
`ASCEND_CUSTOM_OPP_PATH`。不需要编译 SGLang/vLLM，也不需要下载模型权重。
设备内核按实际调用 JIT，不会在安装阶段提前编译全部 DeepGEMM 内核。

## 2. 950 环境要求

| 项目 | 要求 |
| --- | --- |
| 硬件 | Ascend 950，当前入口检查真实设备架构 `dav-3510` |
| CANN | 上游标注 CANN 9.20，包含匹配的头文件、运行库、`bisheng`、`ld.lld` |
| Python | 3.10+ |
| Torch / TorchNPU | 使用目标 950 环境验证过的匹配组合；本仓库尚无已验收的 950 版本锁定文件 |
| Host 编译器 | 编译器和 C++ 标准库支持 C++20 `<format>` |
| 构建依赖 | `setuptools`、`wheel`、`ninja`、`packaging`、`tree-sitter`、`tree-sitter-cpp` |
| 运行依赖 | 目标环境兼容的 `tilelang`；上游包导入时会加载它 |
| 子模块 | 固定提交的 DeepJIT、magic_enum、clangd-ascend |
| 真实 H2D | 额外配置 `memfabric_hybrid` 的 LOCAL/GIANT_PAGE 映射池 |

以下构建/JIT/设备测试只在 950 上运行。当前 910 开发机仅执行 CPU 检查。
`CANN 9.20` 沿用上游版本写法，实际安装目录以目标服务器为准。
不要沿用当前开发机 `scripts/npu-run.sh` 默认的 CANN 8.5.0/`.venv-npu` 环境；
它仅用于本地 CPU 检查，不是这里的 950 配置。

## 3. 获取或更新分支

新机器使用独立目录：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch echo-deepgemm-950 --single-branch \
  https://github.com/rooikeee/ECHO.git ECHO-deepgemm-950
cd ECHO-deepgemm-950
```

如果已有仓库，在确认工作区干净后更新：

```bash
git status --short
git switch echo-deepgemm-950
git pull --ff-only origin echo-deepgemm-950
```

这两种方式都继续执行以下命令。子模块从 **ECHO 根目录**初始化：

```bash
git submodule update --init --recursive -- \
  deepgemm-ascend/third-party/deep-jit \
  deepgemm-ascend/third-party/magic_enum \
  deepgemm-ascend/third-party/clangd-ascend

export ECHO_DG_ROOT="$PWD"
mkdir -p "$ECHO_DG_ROOT/build/deepgemm-echo"
git rev-parse HEAD | tee build/deepgemm-echo/source-commit.txt
git submodule status --recursive | tee build/deepgemm-echo/submodules.txt
```

保持仓库固定的子模块提交，不使用 `--remote` 升级。独立算子测试使用生成数据，
不依赖 ECHO 论文的 Git LFS 数据文件。已有未提交改动时先保留它们，不强制重置工作区。

## 4. 激活并检查目标环境

先激活目标服务器已有的 950 Python/虚拟环境。
将下面 CANN 示例路径替换为实际安装根目录，之后的命令在同一个 Bash shell 中执行：

```bash
export ECHO_DG_CANN_ROOT=/path/to/your/cann-9.20
source "$ECHO_DG_CANN_ROOT/set_env.sh"
export ASCEND_HOME_PATH="$ECHO_DG_CANN_ROOT"
export ECHO_DG_PYTHON=python3
export ASCEND_RT_VISIBLE_DEVICES=0   # 示例物理卡号，按空闲设备修改
set -o pipefail

npu-smi info
"$ASCEND_HOME_PATH/bin/bisheng" --version
test -x "$ASCEND_HOME_PATH/bin/ld.lld"
"$ECHO_DG_PYTHON" -m pip --version
"$ECHO_DG_PYTHON" - <<'PY'
import sys
import torch
import torch_npu
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("TorchNPU:", torch_npu.__version__)
assert sys.version_info >= (3, 10)
assert torch.npu.is_available()
print("Device:", torch.npu.get_device_name())
assert "950" in torch.npu.get_device_name(), "Requires actual Ascend 950"
PY
```

补齐通用构建依赖，不自动替换目标服务器的 Torch/TorchNPU：

```bash
"$ECHO_DG_PYTHON" -m pip install \
  setuptools wheel ninja packaging tree-sitter tree-sitter-cpp
"$ECHO_DG_PYTHON" - <<'PY'
import setuptools, wheel, ninja, packaging
import tree_sitter, tree_sitter_cpp
from tree_sitter import QueryCursor
import tilelang
print("Build/runtime imports OK")
PY
```

缺少 TileLang 时安装目标 950 环境提供或验证过的发行包；本文不指定未经验证的版本组合。
另外单独确认 host 编译器和标准库支持 `<format>`：

```bash
cat > "$ECHO_DG_ROOT/build/deepgemm-echo/check_format.cpp" <<'CPP'
#include <format>
#include <iostream>
int main() { std::cout << std::format("{}\n", "C++20 format OK"); }
CPP
"${CXX:-c++}" -std=c++20 \
  "$ECHO_DG_ROOT/build/deepgemm-echo/check_format.cpp" \
  -o "$ECHO_DG_ROOT/build/deepgemm-echo/check_format"
"$ECHO_DG_ROOT/build/deepgemm-echo/check_format"
```

失败时先配置兼容的 host 编译器/标准库，并设置 `CXX`。
仅加 `-std=c++20` 不会补齐缺失的标准库实现。

## 5. 重编安装偏移映射版 Host 扩展

本次 `_C.echo_paged_mqa_prefetch_out` 增加了 `resident_capacity` 尾参，
并新增 `_C.echo_clear_prefetch_mappings`。**必须重编，不能搭配初版 `_C` 使用。**

```bash
cd "$ECHO_DG_ROOT"
export DG_SKIP_BUILD=0
export MAX_JOBS=8
bash scripts/build-echo-deepgemm-950.sh \
  2>&1 | tee build/deepgemm-echo/build-host.log
```

脚本在 `deepgemm-ascend/` 内执行：

```bash
DG_USE_LOCAL_VERSION=0 "$ECHO_DG_PYTHON" -m pip install --no-build-isolation --no-deps .
```

上面的单行用于解释脚本内部操作；正常使用构建脚本即可。
`--no-deps` 要求依赖事先装好，`DG_USE_LOCAL_VERSION=0` 避免从外层 ECHO 工作区推导包版本。

从 ECHO 根目录检查安装位置及两个接口：

```bash
cd "$ECHO_DG_ROOT"
"$ECHO_DG_PYTHON" - <<'PY'
import inspect
import deep_gemm
from deep_gemm.echo import EchoPrefetchBuffers
print("Package:", deep_gemm.__file__)
print("Extension:", deep_gemm._C.__file__)
assert hasattr(deep_gemm._C, "echo_paged_mqa_prefetch_out")
assert hasattr(deep_gemm._C, "echo_clear_prefetch_mappings")
p = inspect.signature(EchoPrefetchBuffers.allocate).parameters["resident_capacity"]
assert p.default is inspect.Parameter.empty
assert hasattr(EchoPrefetchBuffers, "clear_prefetch_mappings")
print("Tagged-mapping host API OK")
PY
```

不要在 `deepgemm-ascend/` 目录内做此导入检查，以免 Python 优先加载未构建的源码包。
接口检查成功只表示 Host 包可以加载；下一步才触发设备内核 JIT 与真实执行。

## 6. 首次 JIT 与 HBM 正确性/Graph 验证

DeepJIT 首次调用使用 `bisheng` 和 `ld.lld` 生成并加载设备内核。
测试脚本先图外运行两次完成预热，再捕获 Graph、修改设备元数据并重复 replay。
给偏移映射版使用独立缓存目录：

```bash
export DG_JIT_CACHE_DIR="$ECHO_DG_ROOT/build/deepgemm-echo/jit-950-tagged-map"
export DG_JIT_PRINT_COMPILER_COMMAND=1
export DG_JIT_PRINT_LOAD_TIME=1

"$ECHO_DG_PYTHON" scripts/check-echo-deepgemm-950.py \
  --requests 1 2 --seq-len 2048 --capacity 64 --resident-capacity 4096 \
  --token-bytes 1152 --device 0 \
  --output build/deepgemm-echo/950-hbm-validation.json \
  2>&1 | tee build/deepgemm-echo/950-hbm-validation.log
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `--requests 1 2` | 分别验证 1、2 个请求；每请求固定 4 条 query，即 MTP=3 |
| `--capacity 64` | 每条 query 可拥有的 staging 槽位上限 P |
| `--resident-capacity 4096` | 正式缓存槽位数 D，作为 staging 编码的偏移基址 |
| `--token-bytes 1152` | 每个 KV 记录的测试字节数；对应 BF16 的 512 维 latent KV + 64 维 RoPE |
| `--device 0` | 经过 `ASCEND_RT_VISIBLE_DEVICES` 过滤后的逻辑设备编号 |

合成测试不分配正式 resident KV 池，4096 用来验证地址编码与命中判定。
接入实际缓存时，`resident_capacity` 必须等于真实正式缓存槽位数，不能把 P 误传为 D。
满足 `D + rows * P <= INT32_MAX`，否则接口报错。

HBM 模式输出 `source="HBM-diagnostic"`，仅验证设备内复制。
检查包括完整有效 logits、唯一 host 预取、staging 字节、偏移解码、反复重放、动态长度/预算、
零 active 清理，以及消费者把临时条目转为正式缓存后该条目能否保留。
异常退出或没有完成 replay 都不能算验收通过。

## 7. 真实 Host DDR → HBM 验证

准备好目标环境的 `memfabric_hybrid` 和 LOCAL/GIANT_PAGE 映射支持后运行：

```bash
"$ECHO_DG_PYTHON" -c 'from memfabric_hybrid import offload; print("MemFabric import OK")'
"$ECHO_DG_PYTHON" scripts/check-echo-deepgemm-950.py \
  --requests 1 2 --seq-len 2048 --capacity 64 --resident-capacity 4096 \
  --token-bytes 1152 --device 0 --host-ddr \
  --output build/deepgemm-echo/950-ddr-validation.json \
  2>&1 | tee build/deepgemm-echo/950-ddr-validation.log
```

脚本创建映射池，用 `offload.get_dva()` 获取设备可访问地址，源池持续存活至测试结束。
输出必须为 `source="MemFabric-DDR"`。缺包、映射失败或空 DVA 会报错，不会退回 HBM。
普通 CPU 指针不能当作 DDR DVA。

## 8. B24 / 140k / MTP=3 性能对照

前两种正确性/Graph 验证完成后，再测长序列：

```bash
"$ECHO_DG_PYTHON" scripts/check-echo-deepgemm-950.py \
  --requests 24 --seq-len 143360 --capacity 64 --resident-capacity 4096 \
  --token-bytes 1152 --device 0 --host-ddr --repeats 100 \
  --output build/deepgemm-echo/950-b24-140k.json \
  2>&1 | tee build/deepgemm-echo/950-b24-140k.log
```

该配置有 96 条 query，源 KV 测试池约占 3.69 GiB，另需 indexer 数据和输出空间。
容量饱和时跨块先后顺序可能不同，预取子集和每行拥有的数量不保证固定。
同一 host 被多个 query 使用时只预取一份，其他行按编码共享它。

| JSON 字段 | 含义 |
| --- | --- |
| `unique_staged_hosts` | 相应验证场景中实际预取的唯一 host 数 |
| `resident_capacity` / `mapping` | 位置编码配置 |
| `baseline_mqa_with_metadata_graph_us` | 原 DeepGEMM metadata + MQA logits 图执行时间 |
| `target_mqa_prefetch_graph_us` | 清理映射/重置 + metadata + 融合打分/预取图执行时间 |
| `complete_speculative_round_us` | 当前为 `null`，没有测完整推测解码轮次 |

性能采样使用最后恢复的 cold/低阈值场景。完整脚本还运行正确性检查，进程总耗时不是算子时延。
两条计时路径均不含最终 TopK、attention、最终 miss 回补；不能直接拿这里的 logits 时间
对比包含 TopK 的原生完整 indexer，也不能据此宣称完整 ECHO 推理已加速。

## 9. 实际调用与偏移映射

Python `allocate()` 新增必填的 `resident_capacity`：

```python
from deep_gemm.echo import EchoPrefetchBuffers

# source：NPU uint8 [host_tokens, token_bytes]，这里展示 HBM 诊断模式。
# 其余张量已按本版本的 MXFP8/paged MQA 接口准备好。
buffers = EchoPrefetchBuffers.allocate(
    rows, max_context, prefetch_capacity, source,
    resident_capacity=resident_cache.shape[0],
)
logits = buffers.run(
    q, q_scales, kv, weights, context_lens, pages, indices,
    thresholds, stable, active, budgets, host_ids, host_to_device, written,
)
# 在同 stream 的后继消费者中读取 logits、映射及 staging。
```

真实 DDR 使用 CPU 映射池的 `source`，并显式传
`source_dva=offload.get_dva(source.data_ptr())`、`source_kind="MemFabric-DDR"`。
第一版要求 MXFP8 E4M3、32 heads、head_dim=128、page_size=128，以及已打包的
MN-major int16 Q scales；原生 indexer 的 FP32 scale 不能直接传入。
测试脚本提供量化和 paged 数据打包示例。

| `host_to_device[h]` | 含义 |
| --- | --- |
| `-1` | 真正 miss |
| `-2` | 算子执行中的临时原子占用，成功结束后不残留 |
| `[0,D)` | 正式缓存槽位 |
| `[D,D+rows*P)` | staging：`flat=value-D`、`owner_row=flat//P`、`slot=flat%P` |

例如 `D=1000,P=64`，值 `1069` 表示 `staging[1,5,:]`。
后续任意 query 可查同一张表定位它，不用重新扫描 staging 列表。
映射和 counts 只表示位置/预留，消费者仍须等**整个预取算子完成**后才能读字节。

## 10. 每轮清理和 Graph 生命周期

`allocate()` 会初始化 counts=0、tokens/hosts=-1。每次 `run()` 的 native 流程先依据旧 hosts
清理本实例上一轮的 staging 编码，再清 counts/tokens/hosts；这一流程被 Graph 捕获并重复执行。
不要在调用前手动清空旧 hosts，否则无法清理仍在映射表里的旧偏移。
清理只在 map 仍等于原 `D+flat_slot` 时改回 miss，保留已被消费者提升成正式缓存槽位的条目。
零 active、零预算也执行清理。

同一 buffers 实例绑定同一张 map，D、输出地址和布局固定；
不同 stream 或不同 buffers 实例不能同时修改同一张映射表。
变更 shape/容量/映射地址需要结束旧消费者，完成旧映射清理，再分配并重新捕获。

最后一个消费者完成后：

```python
buffers.clear_prefetch_mappings()  # 同 stream，幂等，保留 resident 条目
# 清理完成后才能释放 buffers、图引用的张量及 DDR 映射源池。
torch.npu.synchronize()
```

详细并发与预算边界见 [ECHO_PREFETCH_MAPPING.md](ECHO_PREFETCH_MAPPING.md)。
实际执行链包含初始化、cleanup、metadata 和融合内核，不是整个 Graph 只有一个 kernel。

## 11. 常见问题与结果记录

| 现象 | 处理 |
| --- | --- |
| `Requires Ascend 950` | 确认设备选择；不删除检查在 910 上运行 |
| `ASCEND_HOME_PATH` 未设置或 C API 类型缺失 | 检查当前 shell 是否确实使用 950/CANN 9.20 的头文件和库 |
| `<format>` 不存在 | 先通过第 4 节 C++20 编译器/标准库检查 |
| 缺少 DeepJIT 或 magic_enum 头文件 | 在 ECHO 根目录初始化固定子模块 |
| 缺少 `QueryCursor` / `tree_sitter_cpp` | 检查构建 Python 与 tree-sitter 包，类型桩生成依赖这些 API |
| `libhccl`、`libopapi`、`libtorch_npu` 找不到 | 检查匹配的 CANN/TorchNPU 与库加载环境，不混入 910 的库路径 |
| 缺少 `echo_clear_prefetch_mappings` 或参数数量不匹配 | 仍加载初版扩展；重新安装偏移版，核对 `_C.__file__` |
| `resident_capacity` 未传或 mapping span 报错 | 传入真实 D，并检查 `D+rows*P` 的 int32 上界 |
| `Keep ... fixed` / `same host_to_device` | buffers 绑定后的地址/布局改变；结束旧生命周期后重建 |
| JIT 使用旧缓存 | 重新安装源码包，使用新的 `DG_JIT_CACHE_DIR`，保存编译命令日志 |
| MemFabric 初始化/DVA 失败 | 先检查目标服务器映射池、大页和运行库配置，不能替换成普通 CPU 地址 |
| 950 Graph 或设备执行失败 | 保存现场，不能以 eager 或仅 Host 导入成功代替 Graph 验收 |

请保留 `source-commit.txt`、`submodules.txt`、`build-host.log`、目标机验证日志/JSON，
以及 Torch/TorchNPU 和 `bisheng --version` 记录。

本地已通过 66 个 CPU 并发/状态场景、4 项 wrapper 测试，包含 ASan/UBSan 检查。
目标 950 的 Host 扩展、cleanup/metadata/融合 JIT、实际 HBM/DDR Graph 和性能均待验证。
实现细节见 [ECHO_DEEPGEMM_ASCEND.md](ECHO_DEEPGEMM_ASCEND.md)。
