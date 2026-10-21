# Ascend 源码替换与待迁移功能

## 来源与导入方式

2026-09-30 导入：

- SGLang：`https://github.com/Ascend/ascend-sglang.git`，`main` 分支提交
  `f30df723bf146d4a8037010b0b11939a601df423`（提交日期 2026-04-02）。
- DeepGEMM：`https://github.com/deepseek-ai/DeepGEMM-Ascend.git`，`main` 分支提交
  `8491bbb4b8c02a094a2318965f50c70438a3e73c`（提交日期 2026-09-30）。

主体源码按固定提交导入为普通文件，保留各自的许可证和上游文档。
DeepGEMM 的 `clangd-ascend`、`deep-jit`、`magic_enum` 依赖保持上游固定提交，
并在仓库根目录 `.gitmodules` 注册加上目录前缀的路径。
`deepgemm-ascend/.gitmodules` 保留的是上游原始记录。

本地 SGLang 安装元数据修改：

1. `python/pyproject_cuda.toml` 是上游默认 `pyproject.toml` 的原样备份。
2. 当前 `python/pyproject.toml` 使用上游 NPU 依赖配置。
3. 将动态版本改为 `0.0.0+ascend.f30df723bf14` 并移除其 SCM 配置，
   避免从外层 ECHO Git 仓库推导出错误的包版本；这不是 SGLang 官方版本号。

除此之外，本次不修改两份上游主体实现。来源清单见根目录 `upstreams.lock.json`。

## ECHO 功能迁移表

原始代码统一从提交 `bc1b75c1000010d0ac6f032ebaac283255c050b1` 读取。
表中的原路径属于该提交，不能直接当成本分支的现有路径使用。

| 范围 | 原始 ECHO 入口 | 当前状态 / 后续工作 |
| --- | --- | --- |
| 模型与 NPU 注意力 | `sglang/python/sglang/srt/models/deepseek_v2.py`、`layers/attention/nsa/nsa_indexer.py` | 昇腾上游实现已导入，尚未在本机验证模型与量化路径 |
| 卸载缓存池接入 | `sglang/python/sglang/srt/model_executor/model_runner.py` | 待将 ECHO 缓存池接入新的 NPU 后端 |
| 主机 KV、设备缓存及淘汰 | `sglang/python/sglang/srt/mem_cache/memory_pool_host.py`、`recall_ops.py` | 待适配设备缓存布局、索引映射、主机内存访问和异步回搬 |
| 图执行与同步 | `sglang/python/sglang/srt/model_executor/cuda_graph_runner.py` | NPU Graph 兼容是实现要求；待适配 Stream、Event 及捕获后的状态重置，并验证重复重放 |
| 索引打分与融合预取 | `DeepGEMM/csrc/apis/attention.hpp`、`DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh`、`sm90_fp8_paged_mqa_logits.cuh` | Ascend 上游有普通 MQA logits 算子，但没有 ECHO 的融合预取扩展；需迁移算法和接口 |
| 实验脚本 | `sglang/e2e_test_*.sh`、`DeepGEMM/test_fuse.py` | 原脚本可从原提交取回，待改成 NPU 环境、资源参数和数据路径 |
| 图 7–9 数据分析 | `top-k-figures/` | 保留原工具；CUDA / TileLang 路径及 LFS 数据准备尚待处理 |

原来的 `NSA_KV_OFFLOAD`、`NSA_DEV_CACHE_SIZE` 和
`SGLANG_NSA_FUSE_LOGITS_RECALL_*` 不能视为本分支已有的有效 ECHO 开关。

上游 DeepGEMM 的 API 兼容声明不包括 ECHO 的私有扩展。
例如新源码提供 `fp8_fp4_mqa_logits` 和 `fp8_fp4_paged_mqa_logits`，
不能直接代替旧 ECHO 的融合预取调用；参数、布局、量化比例格式和精度都需验证。
SGLang 的 NPU 注意力目前有自己的 `torch_npu` 算子路径，导入两个源码目录不会自动把它接到 DeepGEMM-Ascend。

## 当前硬件限制与建议顺序

本机为 Ascend 910 / CANN 9.1.0，导入的官方 DeepGEMM-Ascend 面向 Ascend 950 / CANN 9.20。
不应仅通过修改架构名称或版本检查来宣称支持 910。

1. 先核实 910 上可用的算子和依赖组合，跑通支持 NPU Graph 捕获与重放的 SGLang 基线。
2. 在 NPU Graph 路径中实现不带预取的 ECHO KV 卸载与精确回搬，验证缓存映射、淘汰、输出一致性。
3. 为目标硬件移植支持 NPU Graph 的索引计算与预取融合，验证缺失 KV 总能正确回补。
4. 在图模式下评估传输与计算重叠、长上下文吞吐和延迟。

## NPU Graph 实现要求

2026-10-08 确定：后续 ECHO 的设备算子和状态更新必须支持 NPU Graph 捕获与重放。
关闭图的 eager 路径可用于定位问题，不能替代图模式的功能和性能验收。

- 缓存池、映射表、计数器和预取缓冲区预先分配；重放时更新内容，保持图引用的地址有效。
- 请求长度、Top-K 索引、缓存命中与缺失数量变化时，必须保持捕获算子的合法输入布局，
  通过元数据、有效长度和掩码表达动态工作量。
- 图内不得依赖 Python 按设备结果分支或临时分配工作缓冲区。
  CPU 调度与图外传输可以保留，但必须明确它们与图内消费者的同步和生命周期。
- 计数器、临时映射和淘汰状态必须定义每次重放的更新或重置语义。
  异步回搬与预取的 Stream/Event 顺序必须在图模式下验证。
- 每项改动都需验证图捕获、实际重放、重复重放和输出一致性；缓存命中、缺失、淘汰、
  请求长度变化及支持的 batch size 都应有对应验证。

当前上游 NPU Graph 主要覆盖 decode；prefill 的执行方式需要在相应迁移步骤明确验证。
本机环境和基线测试入口见 [NPU 环境说明](NPU_ENVIRONMENT.md)。

本次检查范围是源码完整性、子模块固定版本和安装元数据。
未执行依赖安装、算子编译、NPU 推理或论文性能实验。
