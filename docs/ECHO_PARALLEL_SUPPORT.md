# ECHO Ascend TP 与 DP Attention 接入

更新日期：2026-10-10。本次范围是 PD 合部下的 TP/DP attention 接入，不包含模拟
PD、真实 PD 传输、预取或性能验收。第一阶段 recall 与性能记录见
[Recall 实现与测试进展](ECHO_RECALL_DESIGN.md)。

## 1. 并行方式

沿用 SGLang 已有的 DP attention 通信、调度和 NPU Graph，不重新实现并行框架。
开启 `--enable-dp-attention` 时，同一组 TP ranks 被划分为多个 attention DP 组：

```text
attention_tp_size = tp_size / dp_size       # 当前要求 CP=1
attention_tp_rank = tp_rank % attention_tp_size
attention_dp_rank = tp_rank / attention_tp_size  # 整数除法
```

| 配置 | 单个服务卡数 | Attention 分组 |
| --- | ---: | --- |
| TP8 / DP1 | 8 | 一个 8-rank attention TP 组 |
| TP8 / DP2 | 8 | 两个 DP 组，每组 attention TP4 |
| TP8 / DP4 | 8 | 四个 DP 组，每组 attention TP2 |
| TP8 / DP8 | 8 | 八个 DP 组，每组 attention TP1 |

FFN/MoE 的收集、计算、分发继续由框架处理。这里的 DP8 不是额外创建 8 个
完整 TP8 模型副本；普通多副本 DP 不属于本次多卡验收目标。

## 2. ECHO 缓存的归属

每个 worker 有独立的 ECHO KV 池、注册主机内存、index K 和每层热缓存。
同一 attention TP 组中的 ranks 处理相同请求，各自持有 MLA KV；不同 attention
DP 组处理独立请求，其完整 KV 池 ID 允许相同，但属于不同的缓存命名空间。

例如 DP0 与 DP1 都分配 ID 128：DP0 的 `host[128]` 和 DP1 的 `host[128]`
可以是完全不同的 KV。DP0 释放这个 ID 不得清除 DP1 的 `written` 或映射。
不能按全局 TP rank 把别的 DP 请求页表拼进本地 recall，也不能共享空闲栈。

`--max-total-tokens` 表示各 worker 的逻辑池容量；`--echo-device-cache-size`
表示各 worker、各层的热槽位容量。本次接入没有把这两个值再按 DP 除一次。
总主机内存预算需要累加所有进程实际分配量，不能仅报一个池的大小。

初始化新增可审计日志：

```text
ECHO parallel layout: attention_tp_rank=0/1, attention_dp_rank=3/8;
logical_capacity_per_rank=8192, hot_cache_per_layer_per_rank=4096
```

## 3. 改动与 Graph 边界

- `server_args.py`：移除 ECHO 对 DP attention 的统一禁止；校验 DP 整除 TP；
  仍禁止 attention CP、PP、PD 分离及原有不兼容功能。
- `model_runner_kv_cache_mixin.py`：将框架初始化好的 attention TP/DP rank 与 size
  传给 ECHO 池，不自行用 GPU/NPU 物理编号推导通信分组。
- `echo_memory_pool.py`：记录并校验所属 attention rank，保留逐进程独立分配，
  输出池容量与 rank 日志。
- `echo_recall.py`：空闲 DP rank 的零 query 输入直接返回空索引，不启动零 grid
  kernel，不访问不存在的请求元数据。
- `scheduler_metrics_mixin.py`：补充 NPU DP 的全局 prefill 标记，区分混合步骤
  的预期 eager 与纯 decode 的意外 Graph 回退。

decode 的命中检测、slot 分配、淘汰、映射更新、备份及回搬继续在 NPU Graph 内。
DP 通信继续使用框架捕获路径，不增加 CPU miss 数决策。负载不均时，空闲 rank
仍需参与框架的 FFN/通信；padding query 不得转成真实逻辑 ID。
请求调度、逻辑页生命周期和错误检查仍在 Graph 外，prefill 仍为 eager。

PD 合部的 DP attention 存在全局混合步骤：某个 DP rank 做 prefill，另一个做 decode。
框架对各 rank 的 Graph 可执行标记取最小值，只要有 prefill，整个同步步骤就不能
使用 decode Graph。因此本地 `Decode batch` 日志可能显示 `npu graph: False`，
这不是 recall 不兼容 Graph。新增 NPU DP 日志 `dp global prefill: True/False`
记录这一区别；测试仅允许有明确 `True` 标记的混合 eager 步骤，纯 decode/idle
步骤回退或缺少标记仍判失败。没有通过关闭 Graph 或强制把 prefill 捕获进图规避问题。

## 4. 验证入口

CPU-only 配置、rank 分组、测试脚本及 Graph 日志验收检查：

```bash
python3 scripts/test-echo-parallel.py
```

单卡 recall、空闲/padding Graph replay、独立 DP 池隔离回归：

```bash
ECHO_NPU_DEVICE=4 ./scripts/npu-run.sh python scripts/check-echo-recall-npu.py
```

新的 paired 模型级测试会依次启动 baseline 与 ECHO，退出时只清理自己启动的服务。
默认生成小型随机 DeepSeek NSA 配置并使用 dummy 权重，包含 dense 与 MoE 层，
不是模型质量或吞吐测试。每个 DP rank 定向执行 batch 1/2 和重复请求，然后并发
执行不同本地 batch/长度，检查路由返回值、全部 rank 的 capture、各 DP rank 的
batch 1/2 实际 Graph replay，以及 baseline/ECHO token 与 logprob 对照。
混合 eager 日志条数会单独保存；不能把“稳定 decode Graph 通过”写成全流程 Graph。

在确认所选设备空闲并且可创建上下文后，将 `ECHO_TEST_DEVICES` 设置为两张可用卡
的逗号分隔 ID，运行最小双卡验证：

```bash
ECHO_NPU_DEVICE="$ECHO_TEST_DEVICES" ./scripts/npu-run.sh \
  python scripts/smoke-echo-parallel-npu.py --tp-size 2 --dp-size 2
```

改变为 `--tp-size 2 --dp-size 1` 验证纯 TP；四卡 `--tp-size 4 --dp-size 2`
用于验证同时存在 attention TP 与 DP 的分组。完整 TP8/DP8 入口：

```bash
ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15 ./scripts/npu-run.sh \
  python scripts/smoke-echo-parallel-npu.py --tp-size 8 --dp-size 8 --long-prompt
```

以上是独立命令，不会由脚本自动扫描或占用其他用户的卡。实测目录位于
`build/echo-parallel/<时间>/`，保存命令、服务日志、逐测试响应和 `result.json`。
有服务占用时直接失败，不停止它们；启动前 process 检查不是跨用户的资源预留锁。

替换为完整 AWQ 模型时使用 `--model build/models/DeepSeek-V3.2-AWQ --quantization awq`；
用 `--long-prompt` 增加超过 2048-token Top-K 的输入验证。该测试不能替代长上下文
持续 miss/淘汰压力测试，也不能替代正式任务精度评测。

## 5. 当前验收状态

代码与测试入口已接入，**小型随机 NSA 模型的 TP8/DP8 八卡实测通过**。
这不等于完整 DeepSeek-V3.2-AWQ 的 DP 精度、性能或稳定性验收通过；所选八卡
均有硬件 Alarm，当前结果属于该设备状态下的集成验证，仍需健康设备复测。

| 检查 | 本轮结果 | 验证边界 |
| --- | --- | --- |
| CPU 并行检查 | 10/10 通过 | ECHO 参数块、TP/DP rank 分组、驱动、端口、完整模型启动参数及日志验收逻辑 |
| 原 ShareGPT 驱动回归 | 11/11 通过 | 旧性能测试入口未被本次接入修改 |
| 真实 `ServerArgs` 初始化及校验 | TP2/DP1、TP2/DP2、TP8/DP2、TP8/DP4、TP8/DP8 通过 | 没有加载模型或建立这些多卡通信组 |
| NPU recall 回归 | NPU 4 上 28/28 通过，5.477 秒 | 组件行为、独立 DP 池隔离、空闲与活动切换、实际 Graph capture/replay |
| 小型 NSA 模型 baseline/ECHO | TP1/DP1 通过 | 真实模型服务，含 dense/MoE，batch 1/2 Graph；不验证多卡 DP |
| 原 Qwen2 baseline smoke | NPU 4 上通过 | batch 1/2 capture/replay、重复请求 token 一致，非 ECHO/NSA 测试 |
| 小型 NSA TP2/DP1 | 通过 | attention TP2，batch 1/2 Graph，对照 token 一致、logprob 差 0 |
| 小型 NSA TP2/DP2 | 通过 | 两个 DP rank 定向与并发请求、idle/padding、batch 1/2 Graph |
| 小型 NSA TP4/DP2 | 通过 | 两个 attention TP2 组及跨组 DP 通信、batch 1/2 Graph |
| 小型 NSA TP8/DP8 | 通过 | 八个 DP rank 的定向、并发、idle/padding、batch 1/2 Graph 和 2200-token 长输入 |

小型 NSA 模型测试目录为 `build/echo-parallel/20261010-151952/`，两边分别完成
single、pair、reuse、concurrent 四组，共各 5 条请求、各输出 80 token。
两边输出 token 全部一致，最大 logprob 绝对误差为 0（容差 0.05），Graph capture
和 batch 1/2 实际 replay 通过。133.76 秒包含两次服务启动，不是性能指标。
该轮 `result.json` 的 `hardware_alarm` 记录的是整机 `npu-smi` 输出中存在 Alarm，
不能解读为 NPU 4 自身有 Alarm；新驱动已区分整机告警与所选卡健康状态。

独立 DP 池隔离测试是在同一张 NPU 上创建两个独立池，验证相同 ID 的不同内容及
释放互不影响；它不是双进程 HCCL 或多卡服务测试。组件测试耗时包含已有编译缓存，
同样不是模型吞吐或优化加速比。

2026-10-10 本轮设备检查：NPU 4 可以分配张量；2、3、7 创建上下文报 `507033`；
8–15 有另一组 vLLM TP8 worker，占用约 38473 MiB/卡，没有停止这些进程。
随后再次检查发现该组 vLLM 已退出，8–15 无运行进程，才开始多卡验证。
这些设备状态仅代表各检查时刻，不能作为后续设备空闲保证。

### 已完成的多卡模型对照

| 配置 | 测试目录 | 两边各请求数 | Token 不一致数 | 最大 logprob 差 |
| --- | --- | ---: | ---: | ---: |
| TP2/DP1 | `build/echo-parallel/20261010-154237/` | 5 | 0 | 0 |
| TP2/DP2 | `build/echo-parallel/20261010-153253/` | 11 | 0 | 0 |
| TP4/DP2 | `build/echo-parallel/20261010-153819/` | 11 | 0 | 0 |
| TP8/DP8 | `build/echo-parallel/20261010-154752/` | 45 | 0 | 0 |

每条输出 16 token，以上是相同并行配置的 baseline/ECHO 对照，没有跨拓扑比较
dummy 权重生成质量。所有 TP rank 完成 capture；每个 DP rank 都有 batch 1/2
实际 Graph replay 日志。TP2/DP2 baseline/ECHO 分别出现 1/3 条混合 eager 日志；
TP4/DP2 分别为 0/2 条，TP8/DP8 分别为 2/4 条；纯 decode eager 日志均为 0。混合日志数量受请求到达时序
影响，不作为两边性能对比指标。原有 8–15 硬件 Alarm 仍在，未执行重置。

TP8/DP8 两边各输出 720 token，包含八个 rank 的重复请求和负载不均的并发请求，
以及 DP0 的 2200-token 长输入。218.77 秒包含两次服务启动，不是吞吐指标。
测试结束后确认 8–15 没有运行进程；没有停止其他用户的服务。

初次 TP2/DP2 和 TP4/DP2 测试分别在 `20261010-152823/`、`20261010-153015/`
被旧的“任何本地 Decode eager 都算失败”规则拦截，原结果保留为失败。
新增全局 prefill 标记后使用新目录重跑；没有在旧日志中猜测或补写标记。
`20261010-153318/` 的 TP4/DP2 baseline 已完成，但 ECHO 启动被 DP TCP 端口
冲突阻断。新驱动选择并检查 HTTP 与 DP ZMQ 端口块，避开本机默认临时连接端口
范围后重跑成功。端口探测仍不是资源预留锁。

此前完整 DeepSeek-V3.2-AWQ 的 TP8/DP1 Graph 验证和 40 条性能对照不改写；
它们不能作为新增 DP attention 的验证结果，也不能证明并行接入后已有性能收益。

### 完整模型 TP8/DP8 对照

2026-10-10 已启动完整本地 DeepSeek-V3.2-AWQ 的 TP8/DP8 对照，
重跑目录为 `build/echo-parallel/20261010-160757/`，两次启动均失败，
未完成任何模型生成，不能计作通过。
使用全部 61 层和真实 AWQ 权重，不设置 dummy 加载或 debug 层数限制。
输入改用模型的 chat template 与正式 tokenizer 编码，包括中文算术、首都、
矩阵乘法和超过 Top-K 的背景记录；不使用小模型的占位 token。
baseline/ECHO 使用相同输入与配置，每边计划完成 45 条请求。

```bash
ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15 HCCL_BUFFSIZE=256 ./scripts/npu-run.sh \
  python scripts/smoke-echo-parallel-npu.py --tp-size 8 --dp-size 8 \
  --model /home/l50066329/code/ECHO/build/models/DeepSeek-V3.2-AWQ \
  --quantization awq --long-prompt --timeout 1800
```

初次 `20261010-160340/` 在 baseline 初始化 KV 池时失败，没有完成生成或 ECHO
对照。权重加载后每卡剩余 3.39–3.68 GiB，原 `mem-fraction-static=0.80` 使
计算出的 KV 容量为负；原失败目录和结果不改写。其中 `long_prompt_tested: true`
是旧驱动对请求选项的记录，不代表这次启动失败实际完成了长输入。

重跑把完整模型的 `mem-fraction-static` 调为 0.96，HCCL buffer 显式设为
256 MiB；保留每个 DP rank 的 512-token chunk 和 8192-token 池上限。
使用 1800 秒 watchdog 和 expandable segments；小模型仍使用 180 秒 watchdog。
两边保留 batch 1/2 Graph，
本轮是生成与数值对照，不是正式任务数据集的准确率或吞吐评测。

第二次重跑在分布式 TCPStore 初始化时报 `Ping failed, invalid nonce returned`，
尚未加载权重，不能据此判断调整后的显存预算是否足够。再次尝试时占用检查发现
NPU 8 上已有 root 用户的 `VLLM::Worker_TP0_EP0`（PID 7767），测试立即退出，
没有停止该服务。八卡资源重新被占用，完整模型验收目前未通过，等待资源释放。
后续驱动已将 `long_prompt_requested` 与 `long_prompt_tested` 分开记录，
仅两边都完成服务生成与 Graph 验证后才记长输入已测试；旧失败结果保持原样。
