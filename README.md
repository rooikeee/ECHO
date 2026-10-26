# ECHO — Ascend 开发分支

本分支以 Ascend 上游源码作为后续移植的基础，包含两个开发目录：

| 目录 | 上游 | 固定提交 |
| --- | --- | --- |
| `sglang-ascend/` | `Ascend/ascend-sglang` | `f30df723bf146d4a8037010b0b11939a601df423` |
| `deepgemm-ascend/` | `deepseek-ai/DeepGEMM-Ascend` | `8491bbb4b8c02a094a2318965f50c70438a3e73c` |

完整来源、第三方依赖提交和本地修改记录见 [upstreams.lock.json](upstreams.lock.json)。
两个目录的主体代码直接纳入本仓库，方便在同一分支修改；DeepGEMM 的第三方依赖使用固定提交的 Git 子模块。

vLLM-Ascend 的 ECHO 有界预取、LIM/CopySFA 优化、构建和算子测试见
[README.vllm-ascend.md](README.vllm-ascend.md)。该路径使用独立的固定上游源码与补丁，
预取默认关闭；已验证算子与 NPU Graph，尚无完整模型性能或精度验收。

## 当前状态

- 跨机器交接优先阅读：[当前实现](docs/ECHO_ASCEND_CURRENT_IMPLEMENTATION.md) 与
  [半容量 PD 合部实验复现](docs/ECHO_ASCEND_REPRODUCTION.md)。新入口按原始
  ShareGPT 脚本使用 TP8/DP8、100 请求和完整请求率扫描，只测 offload/recall。
  每 rank 主机逻辑池/热池目标为 499,968/99,968 token，本轮尚未完成模型测试。

- 已用上述源码替换原来的 `sglang/` 和 `DeepGEMM/`。
- SGLang 默认安装配置已切换为上游 `python/pyproject_npu.toml` 的依赖。
  原默认配置保留为 `python/pyproject_cuda.toml`。
- SGLang 保留上游的多硬件后端代码；本分支默认安装和后续开发针对 NPU。
- 上游源码导入本身没有移植 ECHO 的 KV 卸载、回搬和融合预取。
  2026-10-09 已新增不依赖 DeepGEMM 的实验性 Ascend 精确 recall 路径，
  单卡算子及 Graph 验证已通过，完整模型验证范围见
  [Recall 实现说明](docs/ECHO_RECALL_IMPLEMENTATION.md)。预测与融合预取尚未实现。
- 2026-10-10 已接入 ECHO 的 TP/DP attention，沿用上游并行与 NPU Graph 路径。
  小型随机 NSA 模型的 TP2/DP1、TP2/DP2、TP4/DP2、TP8/DP8 对照已通过；
  尚未验收完整 AWQ 模型的 DP 精度与性能，不包含 PD 分离。
  实现、测试结果及硬件告警限制见 [TP/DP 接入记录](docs/ECHO_PARALLEL_SUPPORT.md)。
- `top-k-figures/` 是原始 CUDA 分析工具，暂未适配 NPU；其中的数据仍需 Git LFS 下载。

## 与当前机器的兼容性

2026-09-30 检查的机器为 ARM64 / openEuler、Ascend 910、CANN 9.1.0。

**导入的官方 DeepGEMM-Ascend 首发版本面向 Ascend 950，要求 CANN 9.20。当前机器不满足其声明的运行环境。**
源码替换并不意味着它已能在本机编译或运行；910 支持需要另外验证和适配。
详见 [DeepGEMM 上游要求](deepgemm-ascend/README.md#requirements)。

SGLang 的 NPU 安装文档提供了另一套依赖组合，需要与目标硬件及实际 CANN 环境核对：
[SGLang NPU 安装说明](sglang-ascend/docs/platforms/ascend/ascend_npu.md)。
源码导入阶段没有安装运行依赖或执行 NPU 测试。
2026-10-08 已另行配置项目内的 CANN 8.5.0、ATB 和 `.venv-npu`，
通过单卡微型 Qwen2 模型的服务启动、预热、NPU Graph 捕获、decode 重放和生成验证。
使用入口、固定版本和验证范围见 [本机 NPU 环境](docs/NPU_ENVIRONMENT.md)。
后续 ECHO 实现必须支持 NPU Graph；开发要求见 [AGENTS.md](AGENTS.md) 和
[Ascend 迁移说明](docs/ASCEND_MIGRATION.md#npu-graph-实现要求)。

## 准备源码与安装入口

在本仓库根目录初始化固定版本的第三方依赖：

```bash
git submodule update --init --recursive
```

先在独立 Python 环境中按上游文档准备匹配的 `torch`、`torch_npu`、Triton-Ascend 和 NPU 算子库。
完成依赖核对后，SGLang 的本地安装入口为：

```bash
python -m pip install -e ./sglang-ascend/python
```

SGLang 的包版本标记为 `0.0.0+ascend.f30df723bf14`，用于标识本仓库的固定源码快照，
并非上游发布版本。这避免嵌入式源码错误地从 ECHO 的 Git 历史推导 SGLang 版本。

仅在满足 DeepGEMM 的硬件及工具链要求后，按其 [README](deepgemm-ascend/README.md) 安装：

```bash
cd deepgemm-ascend
python -m pip install . --no-build-isolation
```

上述命令是源码安装入口。本机 SGLang 的已验证环境与使用命令见
[本机 NPU 环境](docs/NPU_ENVIRONMENT.md)；DeepGEMM 未安装验证。

## 原始 ECHO 与后续迁移

原始 CUDA ECHO 的完整源码保存在提交：

```text
bc1b75c1000010d0ac6f032ebaac283255c050b1
```

原始实验说明见 [ECHO CUDA README](docs/reference/ECHO-CUDA-README.md)，其中路径和命令仅供迁移参考。
可在不切换工作区的情况下读取原实现，例如：

```bash
git show bc1b75c1000010d0ac6f032ebaac283255c050b1:sglang/python/sglang/srt/mem_cache/recall_ops.py
```

具体移植范围见 [Ascend 迁移说明](docs/ASCEND_MIGRATION.md)。
