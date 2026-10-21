# Ascend ECHO ShareGPT 对照测试

## 目的与范围

比较当前 Ascend attention 基线与第一阶段 ECHO 精确 KV recall 的端到端性能。
模型为本地完整 `DeepSeek-V3.2-AWQ`，不是随机小模型或 dummy 权重。
本测试不包含预测、融合预取，也不构成生成质量或模型精度验收。

原始 ECHO 制品的 `sglang/e2e_test_mix_sharegpt.sh` 使用 ShareGPT 测延迟，
部署为 PD 合部、TP8、DP8 并启用 DP attention。参考本仓库历史提交
`bc1b75c1000010d0ac6f032ebaac283255c050b1` 中的脚本和 README。
当前第一阶段尚不支持 DP attention，因此本次使用 TP8 / DP1、PD 合部，
不是论文完整配置复现。两组部署参数保持一致，ECHO 组仅额外启用 recall。

## 固定配置

| 项目 | 设置 |
| --- | --- |
| 物理设备 | 8–15；启动前检查空闲，不停止其他服务 |
| 模型 | `build/models/DeepSeek-V3.2-AWQ` |
| 请求 | 100 条真实 ShareGPT 第一轮输入，seed 42 |
| 生成 | 固定 128 token，temperature 0，ignore EOS |
| 输入筛选 | 输入与生成预算之和不超过 2048 token；不截断输入 |
| 并发与到达率 | 最大并发 2，request rate 为 inf |
| 部署 | TP8 / DP1，PD 合部 |
| NPU Graph | 开启；验证 batch 1/2 实际 decode replay，无 eager 回退 |
| 服务上下文 / 逻辑容量 | 4096 / 8192 token |
| ECHO 热缓存 | 每层 4096 token |
| Prefill | chunk 512，max-prefill-tokens 2048 |
| 其他 | 禁用 radix cache、overlap schedule、shared experts fusion |
| 预热 | 每组 2 个请求；预热后 flush cache |
| Graph 独立检查 | 计时前先串行 1 个请求，再并发 2 个请求，每条生成 32 token |
| 重复 | 每组 1 次，先 baseline 后 ECHO |

本次是短上下文、低并发的初步测试，不能推导长上下文或高并发性能。
两组客户端使用服务端相同的 tokenizer 加载函数，包含本地 Transformers v5
兼容处理。采样、流式请求和性能统计复用当前 SGLang benchmark 实现。

## 数据版本

- 仓库：`anon8231489123/ShareGPT_Vicuna_unfiltered`
- 版本：`192ab2185289094fc556ec8ce5ce1e8e587154ca`
- 文件：`ShareGPT_V3_unfiltered_cleaned_split.json`
- 文件大小：`672837942` 字节
- SHA-256：`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`
- 本地位置：`build/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json`

文件通过 Hugging Face 镜像下载，实际校验值与固定版本的 LFS 文件记录一致。
测试脚本再次验证校验值，保存输入、token 长度和逐条 prompt 校验值。

## 运行与产物

CPU 脚本检查：

```bash
python3 scripts/test-bench-echo-sharegpt.py
```

仅准备样本，不启动服务：

```bash
ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15 \
  ./scripts/npu-run.sh python scripts/bench-echo-sharegpt-npu.py --prepare-only
```

启动顺序对照测试：

```bash
ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15 \
  ./scripts/npu-run.sh python scripts/bench-echo-sharegpt-npu.py
```

每次运行创建独立 `build/echo-sharegpt/<时间>/` 目录，保存样本、服务和客户端
命令、日志、完整 JSONL 指标、前后设备健康状态与汇总 `result.json`。
脚本只清理自己启动的进程组，不重置设备，不停止外部服务。

中断恢复可使用 `--reuse-baseline-dir <已有运行目录>`。脚本校验保存的完整输入、
服务与客户端命令、请求成功数和输出长度，并要求原始计时日志证明 batch 2
Graph replay。恢复时仍重新启动相同配置的 baseline 完成 batch 1/2 独立检查，
随后才复用计时指标并执行 ECHO；不会伪装为重新测量的 baseline。

关注 output token/s、TTFT、TPOT、ITL 和请求端到端延迟。
请求指标不包含模型加载和预热，但另记包含启动的整体运行时间。
速度比大于 1 表示 ECHO 更快，小于 1 表示更慢。
生成文本一致率仅作诊断，不作为 ShareGPT 答案准确率。
设备健康状态为 Alarm 时，完成的指标也只能作为带硬件告警的临时结果。

## 2026-10-09 恢复记录

- CPU 测试 8 项通过。
- `build/echo-sharegpt/20261009-153216/` 已完成采样：100 条输入共
  27170 token，生成预算共 12800 token。
- `build/echo-sharegpt/20261009-153514/` 在启动前空闲检查失败：物理设备
  8–13 被外部 vLLM TP 服务占用，每卡进程占用约 47 GB。
  该服务使用 `v41-flat-verify3` 模型，端口 19190，服务主进程 PID 为
  `3288653`。记录只表示检查当时状态，后续操作前必须重新确认进程。
- 本次没有启动 baseline 或 ECHO 服务，没有停止外部进程，尚无性能结果。
- 后续确认设备空闲后，在 `build/echo-sharegpt/20261009-161823/` 启动新的
  对照运行。baseline 的 100 条请求全部成功，生成 12800 token，用时
  485.92 秒，输出吞吐 26.34 token/s。该轮计时日志只有 batch 2 Graph replay，
  缺少 batch 1 证据，因此完整验收状态为 `failed`，没有启动 ECHO。
- 增加计时前的独立 batch 1/2 Graph 检查和严格 baseline 复用校验，CPU 检查
  10 项通过。`build/echo-sharegpt/20261009-163312/` 已完成恢复运行，
  复用前一轮 baseline 的计时指标，两组新的独立 Graph 检查均通过。
- 本次 Graph 检查针对 decode，不表示 prefill 也启用了 Graph。

## 已完成结果

本节记录的是修正前“新 KV 先存主机、被选中后回搬”的实现。
后续设备热缓存优先的写入修正尚未复测该 100 请求配置，不能沿用这些指标评估新版本。

恢复运行于 2026-10-09 17:09 左右完成，脚本退出码为 0。
汇总状态为 `completed_with_hardware_alarm`：测试执行完成，但设备告警没有消除。
两组各成功完成 100 条请求、生成 12800 token，无请求错误。
baseline 使用前一轮保存的性能指标，新启动同配置服务补充 Graph 检查；
ECHO 的性能请求在恢复运行中实际执行。没有把复用指标标记为新测量。

| 指标 | Baseline | ECHO |
| --- | ---: | ---: |
| 计时请求总用时 | 485.92 s | 1809.23 s |
| 输出吞吐 | 26.34 token/s | 7.07 token/s |
| 平均请求端到端延迟 | 9.72 s | 36.18 s |
| 平均 TTFT | 0.989 s | 9.128 s |
| 平均 TPOT | 68.73 ms | 213.04 ms |
| P99 请求端到端延迟 | 11.63 s | 43.49 s |

当前短上下文、并发 2 的配置下，ECHO 输出吞吐为 baseline 的 `0.269` 倍，
平均端到端延迟约为 baseline 的 `3.72` 倍，**没有性能收益**。
这不是最终 ECHO 融合预取方案，也不是论文配置复现；尚未做算子级 profiling，
不能仅凭这些指标认定具体性能瓶颈。应分别分析 prefill、主机 KV 写入、
缓存元数据处理、回搬和稀疏注意力适配的开销。

完整生成文本一致的请求为 `25/100`。该比例只是两条执行路径的诊断数据，
不是答案准确率，也不证明模型级数值等价；之前严格 logprob 对照仍未通过。

两组独立检查均确认 batch 1/2 decode Graph replay；ECHO 计时请求也使用 Graph，
无 decode eager 回退。服务日志末尾的 SIGTERM / SIGQUIT / EOF 出现在完成指标
后清理子进程阶段，不是本轮请求执行失败。
测试服务已退出，结束后再次检查物理设备 8–15，没有发现运行进程。

原始产物：

- `build/echo-sharegpt/20261009-161823/baseline-metrics.jsonl`
- `build/echo-sharegpt/20261009-163312/baseline-reused-metrics.json`
- `build/echo-sharegpt/20261009-163312/echo-metrics.jsonl`
- `build/echo-sharegpt/20261009-163312/result.json`

## 设备优先写入后的同配置复测

2026-10-10 启动 `build/echo-sharegpt/20261010-091033/`，使用修正后的
`write_policy=device_first, host_backup=same_stream` 实现。
本轮 baseline 与 ECHO 均重新启动服务并实际测量，不复用之前的 baseline 指标。

启动后逐项核对了 24 个测试配置字段，均与
`build/echo-sharegpt/20261009-161823/result.json` 相同；100 条样本的完整文本、
顺序、校验值和输入/输出 token 预算完全一致。baseline 服务命令除了动态端口之外
也完全一致。两组均执行计时外的 batch 1/2 Graph 检查。

本轮配置仍是 100 请求、每条 128 token、seed 42、最大并发 2、TP8 / DP1、
PD 合部、decode Graph、prefill chunk 512，输入加输出预算不超过 2048 token。
这对齐的是此前本机的 100 请求对照，不是原论文 DP8 配置复现；4 请求集成检查
不作为性能对照。

### 用户停止后的前 40 条汇总

用户随后要求测到 40 条停止。本轮 baseline 的 100 条已经完成；处理停止时 ECHO
已完成 42 条、开始 44 条，服务和客户端均已退出，没有继续测剩余请求。
比较只使用两边相同的前 40 条，不将 baseline 全部 100 条与 ECHO 部分请求直接比较。

停止服务后，客户端在查询 `/server_info` 时异常退出，未保存 ECHO 的逐请求指标。
这是本次停止处理的缺口；原始 `result.json` 保留 `failed`，另外生成
`stopped-prefix40-summary.json`，状态为 `user_stopped_partial_log_recovery`。

| 前 40 条指标 | Baseline | ECHO |
| --- | ---: | ---: |
| 服务日志估算窗口 | 192 s | 1115 s |
| 估算端到端输出吞吐 | 26.67 token/s | 4.59 token/s |
| Graph decode 日志吞吐中位数 | 29.54 token/s | 6.32 token/s |
| 客户端平均 TTFT | 902.39 ms | 未保存 |
| 客户端平均 ITL | 68.92 ms | 未保存 |

两边均为输入合计 14628 token、输出预算 5120 token。估算窗口由第 1 条性能请求
接收时间到第 42 条接收时间计算：两边前 42 条观察到成对请求在同一秒开始，在
固定并发 2、FIFO、请求执行超过一个时间桶、无前序失败或跨批超越的假设下，
用下一对开始估算前 40 条结束。仅凭并发和 FIFO 不能一般性地证明前缀完成。
日志只有秒级精度，窗口包含下一对接收开销；HTTP 200
本身不是生成完成事件。这是相同口径的日志估算，不能标成客户端精确计时结果。

ECHO 估算输出吞吐约为 baseline 的 17.2%，当前没有收益；decode 日志中位数不是
端到端吞吐。缺失的 ECHO TTFT、TPOT、E2E、P99 和文本一致率无法恢复，不用旧结果
补齐。本轮两边 decode 使用 Graph；8–15 卡在结束后没有运行进程，原有硬件 Alarm
仍在，因此结果只作为临时诊断。

离线恢复脚本为 `scripts/summarize-echo-sharegpt-prefix.py`。完整算法和限制见
[Recall 实现详解](ECHO_RECALL_DESIGN.md)。本次没有额外重跑 40 条。

## Recall 热路径优化后的复测尝试

用户随后要求优化设备优先路径。已接入设备空闲栈、epoch 全命中快路径、分块
LRU 选择和固定 worker 回搬，移除无条件全排序；prefill 错误检查改到整个 forward
结束后。25 项 NPU 回归通过，组件 Graph 的 batch 2 写入加 recall 耗时由
3.848 ms 降到 0.335 ms，但这不是模型吞吐提升倍数。

尝试启动 `build/echo-sharegpt/20261010-110825/` 的两边各 40 条 fresh 测量。
样本与上一轮前 40 条完全一致，并启用逐请求增量 JSONL；但 8–15 卡被新启动的
root vLLM TP8 进程占用，空闲检查报错，模型服务没有启动。
因此没有新模型吞吐数据，不能确认是否超过 host-first 的 7.07 token/s。
占用的 vLLM 进程未被终止；本次测试进程已退出。

改动、组件原始数据及重跑命令见 [优化记录](ECHO_RECALL_OPTIMIZATION.md)。

## 优化后完成的 40 条 Fresh 对照

2026-10-10 用户释放 8–15 卡后，在 `build/echo-sharegpt/20261010-111911/` 完成两边
各 40 条 DeepSeek-V3.2-AWQ 测试。TP8/DP1、PD 合部、并发 2、seed 42、输出 128、
prefill chunk 512、Graph batch 1/2，样本与上一轮前 40 条完全一致。
两边独立 fresh 服务和 fresh 客户端计时，不复用 baseline，不包含加载和预热。

| 指标 | Baseline | 优化后 ECHO |
| --- | ---: | ---: |
| 成功请求数 | 40/40 | 40/40 |
| 输入 / 输出 token | 14628 / 5120 | 14628 / 5120 |
| 正式计时 | 191.93 s | 809.39 s |
| 端到端输出吞吐 | 26.68 token/s | 6.33 token/s |
| 平均 TTFT | 886.03 ms | 24117.37 ms |
| 平均 TPOT | 68.58 ms | 128.66 ms |
| 平均 E2E | 9595.07 ms | 40456.95 ms |

ECHO 为 baseline 吞吐的 23.71%，没有端到端收益。其吞吐数值低于旧 host-first
100 条的 7.07 token/s，但不同样本范围不能作为严格同样本回归比较。
decode 日志约 18 token/s 不能替代完整吞吐；首 token 前路径仍有很大额外开销，
需要 profiling 批量 prefill 写入、备份和注意力路径，不能仅凭 TTFT 锁定某个算子。

两边 Graph 捕获、batch 1/2 重放检查通过，没有 decode eager fallback。
两边 40 条逐请求记录和完整指标均已保存且一致，生成文本完全一致 12/40，
没有测任务准确率或完成严格数值验收。
最终状态为 `completed_with_hardware_alarm`；原有硬件 Alarm 仍在，结果为临时诊断。
测试进程和服务均已退出，结束后检查 8–15 卡没有运行进程。

完整分析和产物列表见 [优化记录](ECHO_RECALL_OPTIMIZATION.md)。
