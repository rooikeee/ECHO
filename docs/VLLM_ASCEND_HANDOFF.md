# vLLM-Ascend 开发交接

更新日期：2026-10-26。交付分支：`vllm-ascend`。

## 交付范围

本分支直接提交 `vllm-ascend/` 的完整项目源码，包括 Python 插件、AscendC
算子、Torch 适配层、构建配置、上游测试和文档。源码已经包含我们保留的优化，
不需要再应用 `patches/`。这些补丁用于审阅改动及恢复独立对照版本。

不提交模型、虚拟环境、CANN 安装包、编译产物、完整 profiling trace，以及
尚未完成的 SGLang 适配。仓库历史中已有的其他项目不属于本次交付的测试范围。
vLLM 核心和 CATLASS 是外部依赖，固定提交记录在
[版本锁文件](../vllm-ascend.lock.json)。保留上游许可证和版权声明。

## 先读哪些代码

| 内容 | 入口 |
| --- | --- |
| LIM / indexer / TopK / MTP miss union / 缓存映射 | `vllm-ascend/csrc/attention/fused_lightning_indexer_manage/` |
| LIM 槽位生成优化 | 上述目录的 `op_kernel/fused_lightning_indexer_manage_union.h`，查找 `PrepareTopkRowsVector` |
| CopySFA / KV gather / 稀疏注意力 | `vllm-ascend/csrc/attention/fused_scatter_copy_sparse_flash_attention/` |
| Torch 算子注册与接口 | `vllm-ascend/csrc/torch_binding.cpp` |
| 服务侧 KV offload / query 准备 | `vllm-ascend/vllm_ascend/attention/sfa_kv_offload.py` |
| ECHO 预取 owner 与消费阶段 | `scripts/echo_indexer_prefetch.py` |
| 精度、Graph 和性能测试 | 根目录 `scripts/check-*`、`scripts/profile-*`、`scripts/collect-echo-prefetch-profile-npu.py` |

LIM 负责产生 TopK、识别未命中、合并 MTP 的重复 miss，并决定目标缓存槽位。
CopySFA 根据这些元数据从 CPU host KV 搬回缺失数据，再计算稀疏注意力。
正常无预取路径是 `LIM → CopySFA`，不经过额外 ECHO 消费 kernel。

### 保留的优化

1. LIM：在适用的 MTP3 四行 query 路径中使用 Gather / GatherMask 生成目标槽位，
   减少 AIV 逐元素标量回填；其他形状保留原路径。TopK 和缓存管理语义不变。
2. CopySFA：支持原生 4-head，并在 `prepare_copy_sfa_queries` 中避免不必要的
   4→8 head 补零和复制。收益主要来自准备操作减少，不是矩阵计算明显加速。

### 实验性预取

ECHO 在 LIM 已发布的 score 上扫描候选，默认研究预算为每个 query 最多预取
64 个 token，跨 block 累计。数据先进入暂存区，不直接覆盖正式热缓存；后续
消费阶段匹配最终 TopK、去重并提交缓存。MTP3 每请求包含 4 行 query。

**预取默认关闭，不能作为已验收的加速方案。** 当前 AIV 扫描、同步和消费阶段
的额外开销超过减少 CopySFA miss 的收益。普通 LIM / CopySFA 优化不依赖预取。
具体协议见 [预取实现](ECHO_INDEXER_PREFETCH.md)。

## 拉取与构建

```bash
git clone --branch vllm-ascend --single-branch https://github.com/rooikeee/ECHO.git
cd ECHO
python3 scripts/prepare-vllm-ascend-sources.py --use-checked-in-ascend --references
```

准备脚本下载并验证归档 SHA256，恢复固定版本 vLLM 核心、原版和 PR 对照，
获取固定提交 CATLASS；保留已提交的 Ascend 源码。要求新工作区尚无 `vllm/`
及 CATLASS 目录，避免覆盖已有开发内容。`--skip-catlass` 仅用于源码验证，
不能用于正式构建。网络需要访问 GitHub codeload 与 CATLASS 上游。

硬件环境是 Ascend 910C / A3、Linux aarch64；已验证软件组合为 Python 3.11、
CANN 9.1、Torch 2.10、torch-npu 2.10.0.post4、Triton-Ascend 3.2.2。
按 [环境文档](VLLM_ASCEND_ENVIRONMENT.md) 安装隔离环境、CANN 和 Python 依赖，
并安装 vLLM 核心，再执行：

```bash
bash scripts/build-vllm-ascend-native.sh
```

首次构建必须包含 Torch 适配层和 ACLNN 自定义算子，不能只编译 kernel。
后续只改现有 LIM kernel 时，可以使用：

```bash
bash scripts/build-lim-copysfa-kernels.sh fused_lightning_indexer_manage
```

## 验证顺序

使用明确空闲的设备，例如设置 `ECHO_NPU_DEVICE=6`，不要默认占用整机。
vLLM 路径使用 `scripts/vllm-ascend-run.sh`；项目 SGLang 基线环境另用
`scripts/npu-run.sh`，两套环境不可混用。

1. 按环境文档执行 runtime Graph 检查及上游两个 fused 算子测试。
2. 按 [主说明](../README.vllm-ascend.md) 执行 LIM occurrence 边界和 CopySFA
   4-head 精度检查。必须验证实际 replay 与连续 replay，不能只看 capture 成功。
3. 执行 B16、MTP3、context 64000、热缓存 12288、每请求去重 miss 400 的微基准。
   MTP3 的 400 是 4 行 query 合并去重后的数量，不是每行 400。
4. 使用独立参考构建对比原版、PR、优化版。测试脚本里的 `baseline` 仅表示关闭
   预取，**不会自动切换到原版源码或原版二进制**。

Graph 使用的输入、输出、暂存区和缓存映射必须保持地址稳定；每次 replay 的
动态索引、有效计数、ready 状态和缓存更新应按协议刷新。覆盖长度变化、命中、
未命中、首填充、淘汰及不同 batch，分别验证流顺序和 host buffer 生命周期。
tiny Qwen2 的单卡 batch 1/2 Graph 通过不代表完整模型 offload 通过。

## 已有结果与未完成项

已记录的单设备 B16 / MTP3 / context 64000 / 热缓存 12288 结果：

| 测量 | 对照 | 优化后 |
| --- | ---: | ---: |
| LIM 独立 kernel | 474.477 us | 458.845 us |
| LIM 独立 Graph | 481.999 us | 466.027 us |
| CopySFA 加 query 准备的 Graph | 453.581 us | 417.408 us |

这些是微基准，不是完整模型吞吐量或单 token 端到端时间。CopySFA 单独 kernel
基本相当；不要把减少补零准备的收益解释为 attention 计算本身加速。
精度检查涵盖 LIM 精确元数据对照、CopySFA KV 精确写回和浮点 attention 参考，
尚未建立完整模型 logits、任务精度、MTP 接受率及服务吞吐量验收。

热缓存 12288 token 的含义是**每请求、每层、每个 rank**，不是整机共享总额。
真正省设备 KV 容量的 PR 路径还需要真实 PD 消费端 KV 输入；
`keep_device_kv_cache=true` 调试模式保留完整设备 KV，不能证明容量收益。
固定版本不直接支持 NPU AWQ；不要套用尚未验证的完整 DeepSeek AWQ 启动方案。

后续优先补齐完整模型真实 CPU offload 路径、Graph 动态状态正确性以及
原版 / PR / 优化版三组同输入对照，再开展真实长上下文服务测试。
历史测量、精度覆盖和限制汇总见 [主说明](../README.vllm-ascend.md) 及
[版本化结果](results/vllm-ascend-20261025.json)。
