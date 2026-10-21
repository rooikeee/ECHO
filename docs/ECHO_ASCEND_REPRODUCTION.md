# Ascend ECHO 半容量 PD 合部实验复现

更新日期：2026-10-21。面向另一台机器上的执行 agent。
先读 [当前实现](ECHO_ASCEND_CURRENT_IMPLEMENTATION.md) 和根目录 `AGENTS.md`。
本次目标是 baseline 对比 **exact offload/recall，无 prefetch**，不是完整论文复现。
不得把准备成功、服务启动、小模型通过或静默缩池当成正式实验通过。

## 1. 获取源码

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch ascend https://github.com/rooikeee/ECHO.git
cd ECHO
git rev-parse HEAD
```

SGLang Ascend 源码直接纳入仓库，不需要另外 clone SGLang。
DeepGEMM 不参与当前 recall 路径，可不初始化其子模块，也不需要下载旧 CUDA 分析数据。
需要研究 DeepGEMM 时才执行 `git submodule update --init --recursive`。
`build/`、模型、数据、venv、CANN、wheel、日志均不在仓库中，需要另行准备。

## 2. 环境准备

本机已验证的组合如下，其他组合不能自动视为等价：

| 项目 | 版本 |
| --- | --- |
| 主机 | ARM64 / openEuler，Ascend910_9382 |
| Python | 3.11.6 |
| CANN Toolkit / 对应 A3 ops / NNAL ATB | 8.5.0 |
| torch / torchvision | 2.8.0 / 0.23.0 |
| torch-npu | 2.8.0.post2 |
| triton-ascend | 3.2.0 |
| sgl-kernel-npu | 2026.3.1，发布标签 2026.03.31.rc3 |
| DeepEP / torch-memory-saver | 同一 NPU kernel 发布包里的配套 wheel |

安装器地址、wheel 发布包和 ABI 说明见 [NPU 环境](NPU_ENVIRONMENT.md#安装来源)，
上游步骤见 `sglang-ascend/docs/platforms/ascend/ascend_npu.md`。
不要照抄上游最新包替代此版本组合，也不要覆盖另一用户的系统驱动/CANN。
先准备匹配的 CANN、ops、NNAL 和 Python 3.11，再创建环境：

```bash
python3.11 -m venv .venv-npu
source .venv-npu/bin/activate
python -m pip install -c scripts/npu-constraints.txt \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c scripts/npu-constraints.txt \
  torch-npu==2.8.0.post2 triton-ascend==3.2.0
# 按 NPU_ENVIRONMENT.md 下载匹配硬件/ABI 的发布包，安装其中配套 wheel。
python -m pip install -c scripts/npu-constraints.txt -e ./sglang-ascend/python
```

`scripts/npu-requirements.lock.txt` 是原机完整包版本记录，含下载/工具等依赖，
不包含 CANN、本地 editable SGLang 和发布包内所有 wheel，不能当成一条命令完成的安装器。
ARM64 torch 使用 C++11 ABI；按环境文档处理 DeepEP so 导入路径。

脚本默认使用仓库内路径。另一台机器可指定现有环境的入口，无需搬运整套 CANN：

```bash
export ECHO_CANN_ENV=/your/cann/ascend-toolkit/set_env.sh
export ECHO_ATB_ENV=/your/nnal/atb/set_env.sh
export ECHO_VENV="$PWD/.venv-npu"
```

这三个路径必须真实存在。后续全部通过 `scripts/npu-run.sh` 执行。

## 3. 模型和数据

当前模型是 ModelScope `QuantTrio/DeepSeek-V3.2-AWQ`，固定 revision：

```text
3a6e8af042a84d4dfd5b09dc6dbdfcd4569beeef
```

约 362 GB、121 个权重分片，AWQ 4-bit/group128/zero-point，61 层。
不是论文引用的 V3.2-Exp-AWQ，不可混用 checkpoint 后复用 baseline。
可以搬运同 revision 的已校验文件到 `build/models/DeepSeek-V3.2-AWQ/`；
也可以安装原机 lock 中的 ModelScope 下载依赖后执行：

```bash
./scripts/npu-run.sh python scripts/download-deepseek-v32-awq.py
```

脚本在清单不存在时请求固定 revision 的 ModelScope 文件清单，SDK 校验 SHA256，
并记录下载状态。这只证明模型文件准备好，不证明推理通过。

ShareGPT 固定来源与原始脚本默认文件一致，下载并验证：

```bash
./scripts/npu-run.sh python -c 'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered", revision="192ab2185289094fc556ec8ce5ce1e8e587154ca", filename="ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset", local_dir="build/datasets/sharegpt")'
sha256sum build/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
```

必须得到：

```text
35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4
```

## 4. 固定实验口径

入口：`scripts/bench-echo-sharegpt-paper-npu.py`。
来源：历史提交 `bc1b75c1000010d0ac6f032ebaac283255c050b1` 的
`sglang/e2e_test_mix_sharegpt.sh`，不是旧的并发 2 benchmark。

| 配置 | 本轮值 |
| --- | --- |
| 模型 | 本地完整 V3.2-AWQ，真实权重，无 dummy/裁层 |
| MoE / Attention | TP8 / DP8，每 rank 独立 KV 命名空间 |
| PD | 合部 |
| ECHO 逻辑池 | 请求 500,000，页对齐目标 499,968 token/rank |
| ECHO 热池 | 请求 100,000，页对齐目标 99,968 token/层/rank |
| Baseline KV 池 | 框架按可用显存估算，不设 500K 的设备 KV 上限 |
| 请求 | 每个请求率 100 条，两边 seed=1、完全相同的样本顺序 |
| 输出 | 每条取数据集回答 token 长度作为预算，ignore_eos=True，不固定 128/256 |
| 请求率 | 按顺序 inf, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 |
| 客户端并发上限 | 不设置 |
| 服务端并发上限 | 框架默认，不人为固定为 2 |
| Graph | decode 开启，最大捕获 batch 32 |
| Prefill | chunk=-1，遵循发布的 ShareGPT 脚本；论文正文的 2048 与该脚本冲突 |
| Radix / overlap | 都关闭，和原始脚本一致 |
| 静态显存比例 | 默认 0.8，和原始脚本一致 |
| 计时 warmup | 0；另有图验证 probe，不计入测量并在测量前 flush |
| prefetch | 未实现；明确不计入本轮 |

原脚本只有 ECHO 指定逻辑/热容量，baseline 使用可用设备 KV；不能把 baseline
也配成 50 万设备 token。这不是两边都设置相同逻辑池的实验。
一边跑完全部请求率后退出，再启动另一边，不复用历史 baseline 指标。
两边各 1,000 请求，总共 2,000 请求；低请求率本身就需要较长时间。

该入口当前设备探测假设每 board 两个 chip（`physical_id // 2`、`% 2`），
其他设备拓扑需要先适配探测，不能跳过占用检查。空闲检查不是资源预留锁。

## 5. 验证与执行

先运行不占用 NPU 的测试：

```bash
./scripts/npu-run.sh python scripts/test-bench-echo-sharegpt-paper.py
./scripts/npu-run.sh python scripts/test-bench-echo-sharegpt.py
./scripts/npu-run.sh python scripts/test-echo-parallel.py
./scripts/npu-run.sh python scripts/test-summarize-echo-sharegpt-prefix.py
./scripts/npu-run.sh python scripts/bench-echo-sharegpt-paper-npu.py --prepare-only
```

原机 prepare-only 的 100 条样本共 33,966 input token、21,520 output token。
新机核对 `samples.json`、tokenizer 和数据校验和；不同就先排查，不能静默继续。

选空闲的一张卡执行基线和组件 Graph 回归，以下物理 ID 仅为示例：

```bash
ECHO_NPU_DEVICE=14 ./scripts/npu-run.sh python scripts/check-npu.py
ECHO_NPU_DEVICE=14 ./scripts/npu-run.sh python scripts/smoke-sglang-npu.py
ECHO_NPU_DEVICE=14 ./scripts/npu-run.sh python scripts/check-echo-recall-npu.py
```

确认八张卡独占且健康后，正式运行：

```bash
ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15 ./scripts/npu-run.sh \
  python scripts/bench-echo-sharegpt-paper-npu.py \
  --devices 8,9,10,11,12,13,14,15
```

只有 8 张卡的机器可改为 `--devices 0,1,2,3,4,5,6,7`，环境变量顺序必须相同。
允许通过 `--model`、`--dataset` 指定其他磁盘路径；内容和校验口径不变。
默认输出在 `build/echo-sharegpt-paper/<时间>/`，也可传 `--output-dir` 指向新目录。
不要覆盖失败目录，不执行 killall 或重置其他用户设备。

## 6. 当前可能阻断及处理原则

**半容量配置尚未在完整模型上启动通过，不能承诺本脚本开箱跑出结果。**
原机最后一次启动前被其他任务占卡拦截，还未实测本轮显存/图容量。

- `mem-fraction-static=0.8` 曾在完整 AWQ TP8/DP8 下预算不足；不要静默改成 0.96。
  `--mem-fraction-static` 支持显式覆盖，但报告会列为额外配置差异。
- 当前 ECHO 逻辑池受保守 full-KV profiling 上限约束，很可能被降到小容量。
  脚本会检查八个 rank 的实际逻辑池/热池及 scheduler 容量，不符合目标即停止。
  不能为了跑通去掉检查；需要先单独修正 ECHO 显存预算，再验证实际分配和 Graph。
- 50 万逻辑池的设备 Indexer K 仍完整驻留，单卡缓存主数据约 13.81 GiB，
  加权重约 58.95 GiB，未包含 Graph、元数据、HCCL 和临时工作区，64 GiB 仍可能不够。
- 大 batch、长输入和服务容量下要重新验证 Graph，不能用 batch1/2 的旧通过替代。
- probe 验证每 DP rank 的 batch1/2/1，正式运行记录观察到的其他 batch 和 Graph 状态；
  未实际出现的 batch 不算已覆盖。DP global prefill 导致的混合 eager 单独记，纯 decode
  eager 则失败；不允许关闭 Graph 跑正式测试。
- 若完整模型数值差异仍存在，性能结果可用于诊断，但不能宣称 lossless 精度验收通过。
- ShareGPT 的热 KV 可能全部命中。该实验可测管理/备份开销，不保证覆盖大量真实 miss，
  miss/eviction 必须由组件测试或独立长上下文实验提供证据。

## 7. 产物和交付清单

保留 `result.json`、`samples.json`、每边 server-command/runtime-env/server.log、
每请求率的 command/metrics/request journal/Graph evidence 和设备健康记录。
启动失败时保留错误和计划命令，不会产生伪造的测量值。
Graph probe 和 startup 不计入正式客户端指标；各 rate 不做测量前额外 flush，
遵循原始脚本在同一个服务内顺序扫描的方式。

交付报告必须写清实际 commit、硬件/软件、权重来源、实际池容量、实际 batch、
请求完成数、TTFT/ITL/E2E/吞吐、Graph 覆盖及所有偏离配置的原因。
本次随仓库交付的是源码、测试与复现方法，不含此前机器的 `build/` 测量原始日志。
