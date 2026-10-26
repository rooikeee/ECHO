# ECHO / LIM / CopySFA on vLLM-Ascend

本目录入口记录 Ascend 910C 上的实验性实现，结果截至 **2026-10-25**，
交付整理日期为 **2026-10-26**。

**已验证的是单设备算子和 NPU Graph；完整模型端到端性能、模型任务精度尚未验收。**
ECHO indexer 预取保留用于研究，默认不启用，因为当前实现明显变慢。

## 实现内容

| 部分 | 实现 | 当前状态 |
| --- | --- | --- |
| 原 PR 的 LIM | AscendC：Indexer、TopK、MTP miss union、淘汰和缓存映射更新 | 固定 PR 17392 源码 |
| ECHO 有界预取 | 在 LIM 内扫描已发布 score，每 query 最多预取64个 token 到 HBM 暂存区 | 功能验证通过，性能不接受 |
| ECHO 消费阶段 | Triton + PyTorch 设备算子：匹配、请求级去重、提交 KV、更新 CopySFA 元数据 | 支持 Graph；额外开销大 |
| LIM 优化 | 用 Gather / GatherMask 生成四路 miss 的目标槽位，减少逐元素标量回填 | 保留；与原 PR 输出完全一致 |
| CopySFA 优化 | 原生4-head 支持，省去 GLM TP16 的4→8 head补零及准备操作 | 保留；算子数值验证通过 |

不启用预取时，执行 `LIM → CopySFA`，不调用 ECHO 消费阶段。
预取仍保留原 TopK 语义，预算是**每行 query 跨 block 累计64**，不是每 block 64。
MTP3 每请求有4行 query；请求级 miss 是这4行合并去重后的数量。

这些优化没有移植进 SGLang 服务，也没有把小模型、假 KV 或少量层测试算作完整模型验收。

## 源码组织

**`vllm-ascend` 分支直接包含当前完整的 `vllm-ascend/` 源码及优化后的算子**，
不是只有补丁的交付。接手说明见 [开发交接文档](docs/VLLM_ASCEND_HANDOFF.md)。
该目录已经应用以下补丁，不要再次应用。vLLM 核心与 CATLASS 作为固定版本依赖准备，
编译产物、模型及本机环境不提交。

| 来源 | 固定提交 |
| --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend PR 17392 | `fcc49fe2520cea586efc59ac04d9a57442bfef0e` |
| 原版 vLLM-Ascend 对照 | `cc3cca4f2e7f038025e4c782b160753509de12b0` |
| CATLASS | `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |

归档 SHA256 和应用顺序见 [vllm-ascend.lock.json](vllm-ascend.lock.json)：

1. [环境兼容补丁](patches/vllm-ascend-environment.patch)：FastAPI、NumPy/OpenCV 约束、构建依赖修复。
2. [ECHO 预取补丁](patches/vllm-ascend-echo-indexer-prefetch.patch)：原生算子、可选接口、ready/cutoff 协议。
3. [LIM 槽位回填优化](patches/vllm-ascend-lim-slot-expansion.patch)。
4. [CopySFA 4-head 优化](patches/vllm-ascend-copysfa-four-heads.patch)：适配层、tiling 校验、服务 query 准备。

Python 预取 owner 和消费 kernel 在 [scripts/echo_indexer_prefetch.py](scripts/echo_indexer_prefetch.py)。
原生实现直接位于 `vllm-ascend/csrc/attention/fused_lightning_indexer_manage/`
和 `fused_scatter_copy_sparse_flash_attention/`。

## 当前结果

### 保留优化

下面是 B16、MTP3、上下文64000、热缓存12288的**单设备**测试，TP16只用于构造本地
query head 形状，不代表实际启动16设备。开启预算预取的路径不参与这组优化比较。

| 实验 | 原路径 | 保留实现 | 口径 |
| --- | ---: | ---: | --- |
| LIM 独立 kernel | 474.477 us | 458.845 us | 32 indexer heads；请求级 union miss400；约3.3%下降 |
| LIM 独立 Graph | 481.999 us | 466.027 us | 30次 replay；缓存重置在计时外 |
| CopySFA + query准备 | 453.581 us | 417.408 us | 同一有效 query；四行共享400 miss；50次 replay |
| CopySFA kernel对应Graph，不含准备 | 416.256 us | 417.408 us | 原8-head与原生4-head基本相当 |

CopySFA 的主要收益来自省去补零准备，不应表述为矩阵计算提速8%。
这些是初步微基准结果，没有跨多轮实验的统计验收。

另一次形状匹配的链路测试为 `807.062 → 796.233 us`，其中 CopySFA kernel 为
`282.383 → 273.082 us`。两次调用的有效 Attention query 随机数布局不同，
因此该组只作为辅助测量；严格同输入的4-head对照使用上表。

### ECHO 预取

B16、MTP3、上下文40960、热缓存8192、**每请求 union miss400**：

| 模式 | LIM | 消费阶段 | CopySFA | 完整Graph |
| --- | ---: | ---: | ---: | ---: |
| 无预取 | 496 us | 0 | 275 us | 0.781 ms |
| 预取上下文，预算0 | 516 us | 2134 us | 308 us | 2.988 ms |
| 每query预取64 | 2219 us | 2138 us | 289 us | 4.671 ms |

分阶段 profiler 与独立 Graph event 计时不可直接相加。预取能减少剩余 miss，
但暴露的扫描、同步和消费开销超过收益，故默认关闭。

原 LIM 在该形状主要受 AIV 侧工作和缓存管理尾部限制；详细证据见
[LIM bottleneck profiling](docs/ECHO_LIM_BOTTLENECK_PROFILING.md)。
页表缓存、AIV交错搬运、16-token搬运批量及额外AIV候选协作均未取得净收益，已撤回。

## 精度与 Graph 边界

- LIM：TopK、有效 miss 输出、槽位和缓存映射与未修改 PR **完全一致**。
- LIM：15个 occurrence 边界用例，覆盖1到4097，每个用例重复 replay 和后续全命中。
- CopySFA：13个 miss 边界，0到2048；原生4-head与补零8-head的KV写回完全一致。
- Attention：浮点对照为 `rtol=0.02, atol=0.002`，不是逐位相同，也不是模型准确率指标。
- 全命中/首填充对照和 B16 miss400 CPU FP32 Attention 参考检查通过。
- ECHO：验证固定地址、每轮 ready/count 重置、预算变化、长度变化、混合首填充/稳态及淘汰。
- 尚无完整 GLM/DeepSeek 的 logits、生成质量、MTP接受率或任务精度验收。

## 准备与构建

需要匹配的 Ascend 910C、CANN 9.1、Python 3.11、Torch 2.10、torch-npu 2.10、
Triton-Ascend 3.2.2。环境隔离与安装步骤见 [环境文档](docs/VLLM_ASCEND_ENVIRONMENT.md)。

克隆 `vllm-ascend` 分支后，在**尚未存在** `vllm/` 和 CATLASS 的工作区执行：

```bash
python3 scripts/prepare-vllm-ascend-sources.py --use-checked-in-ascend --references
# 配置环境文档中的独立 .venv-vllm-ascend 和 CANN 后：
bash scripts/build-vllm-ascend-native.sh
```

准备脚本验证上游归档 SHA256，保留已提交的 Ascend 源码，仅对新建的 vLLM 核心应用兼容补丁。
不覆盖已有源码。要在独立空目录验证归档和补丁恢复，可使用不带
`--use-checked-in-ascend` 的 `--destination /path/to/empty-directory` 模式。
`--skip-catlass` 仅用于源码重放检查；缺少 CATLASS 时不能构建。
原版/PR对照可使用 `bash scripts/build-pr17392-reference.sh` 单独编译；
原版与PR的原始 `csrc` 相同，可将该参考构建的 `.so` 和 `_cann_ops_custom/`
复制到 `versions/original/vllm_ascend/`。三个Python源码树由
`bash scripts/vllm-ascend-variant-run.sh original|pr|optimized ...` 选择。

仅改动现有原生 kernel 时：

```bash
bash scripts/build-lim-copysfa-kernels.sh fused_lightning_indexer_manage
```

该脚本**不重编译 Torch 适配层**。首次安装或改变接口/header时使用完整构建；
`build-echo-indexer-prefetch.sh` 另提供已配置环境中重建 ECHO 接口的入口。

## 测试入口

`ECHO_NPU_DEVICE` 必须明确选择空闲健康设备。以下不启动完整模型。

```bash
./scripts/vllm-ascend-run.sh python scripts/test-echo-profile-workload.py
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh python \
  scripts/check-lim-occurrence-graph-npu.py --output build/lim-boundaries-new.json
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh python \
  scripts/check-copysfa-four-heads-npu.py --output build/copysfa-head4-new.json
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh python \
  scripts/profile-echo-lim-bottleneck-npu.py \
  --output build/profiling/lim-new --context 64000 --hot-tokens 12288 \
  --indexer-heads 32 --modes baseline --metrics PipeUtilization
ECHO_NPU_DEVICE=6 ./scripts/vllm-ascend-run.sh python \
  scripts/collect-echo-prefetch-profile-npu.py \
  --output build/profiling/chain-new --batches 16 --mtp 3 \
  --context 64000 --hot-tokens 12288 --indexer-heads 32 \
  --attention-heads 64 --tp 16 --native-four-heads --attention-reference \
  --misses-per-request 400 --modes baseline
```

测试输出目录不得复用。`baseline` 在这些算子脚本中表示无预取路径，
不是“自动选择原版 vLLM-Ascend”；对比PR必须通过隔离版本启动器执行。
`--write-reference` / `--reference` 支持跨版本精确校验。

## 完整模型状态

计划只比较原版 vLLM-Ascend、原 PR 和优化版三组。对应GLM测试配置为64k、TP16/EP16、
MTP3，原版BS5，offload两组BS16/热缓存12288；这些只是待执行配置。
`prepare-glm52-three-way.py` 生成启动计划，**不代表测试已经运行**。

完整模型未验收：当时TP16设备初始化失败；PR的省显存路径还需要真实PD消费者KV输入。
`keep_device_kv_cache=true` 的调试模式保留完整设备KV，不能证明容量收益。
当前固定 vLLM-Ascend 版本也不直接支持 `--quantization awq`。

## 结果与限制文档

- [版本化结果摘要](docs/results/vllm-ascend-20261025.json)
- [预取实现与同步协议](docs/ECHO_INDEXER_PREFETCH.md)
- [去重后400 miss实验](docs/ECHO_PREFETCH_MISS400_PROFILING.md)
- [完整历史profiling口径](docs/ECHO_PREFETCH_PROFILING.md)
- [环境与构建记录](docs/VLLM_ASCEND_ENVIRONMENT.md)

仓库只保存实现、脚本、补丁及精简结果摘要。模型、完整trace、编译产物和本机原始日志
位于被忽略的 `build/` 中，不会随提交上传。
