# 本机 SGLang Ascend 环境

2026-10-08 在 ARM64 / openEuler / Ascend910_9382 上配置。
系统 CANN 9.1.0 和驱动保持原样；本项目使用独立的 CANN 8.5.0 运行环境。

## 组件

| 组件 | 版本 / 路径 |
| --- | --- |
| Python | 3.11.6，`.venv-npu/` |
| CANN Toolkit 和 A3 算子包 | 8.5.0，`build/cann/cann-8.5.0/` |
| NNAL / ATB | 8.5.0，`build/cann/nnal/atb/` |
| PyTorch / TorchVision | 2.8.0 / 0.23.0 |
| TorchNPU | 2.8.0.post2 |
| Triton-Ascend | 3.2.0 |
| SGLang | 本地 `sglang-ascend/python/` 的 editable 安装 |
| SGLang NPU kernel | 2026.3.1，官方发布标签 `2026.03.31.rc3` |
| DeepEP / Torch Memory Saver | kernel 发布包内的配套 wheel |

版本约束在 `scripts/npu-constraints.txt`，完整 PyPI 包版本记录在
`scripts/npu-requirements.lock.txt`。这个 lock 不包含本地 SGLang editable 安装
以及官方 kernel 压缩包中的三个 wheel，它们保存在 `build/npu-wheels/`。
kernel 压缩包按 Torch 2.8.0 / Python 3.11 / CANN 8.5.0 / A3 / ARM64 选择。

## 使用

在仓库根目录激活环境：

```bash
source scripts/npu-env.sh
python -m sglang.launch_server --help
```

也可以使用包装脚本，不改变当前终端的环境：

```bash
./scripts/npu-run.sh python -m sglang.launch_server --help
```

默认只向进程暴露物理设备 14，进程内使用 `npu:0`。后续测试使用物理设备 8–15；
TP8 测试显式设置 `ECHO_NPU_DEVICE=8,9,10,11,12,13,14,15`。
通过 `ECHO_NPU_DEVICE` 改变物理设备选择，例如：

```bash
ECHO_NPU_DEVICE=14 ./scripts/npu-run.sh python -m sglang.launch_server \
  --model-path /path/to/model \
  --device npu --attention-backend ascend \
  --host 127.0.0.1 --port 30000 \
  --dtype bfloat16
```

默认保留 NPU Graph。后续 ECHO 实现必须通过图模式验收；
`--disable-cuda-graph` 仅用于问题定位，不能作为最终验收配置。
模型权重路径需要另行提供。

## 验证

```bash
./scripts/npu-run.sh python scripts/check-npu.py
./scripts/npu-run.sh python scripts/smoke-sglang-npu.py
```

第一个命令核对包版本、NPU 可用性和 BF16 矩阵计算结果。
结果保存到 `build/npu-environment.json`。
第二个命令创建两层的随机 Qwen2 模型，在临时本地端口启动 SGLang，
捕获 batch size 1、2 的 NPU Graph，等待预热完成后依次提交 batch size 1、2、1 的请求，
每个请求生成 8 个 token，并检查重复请求的 token 一致性。
测试必须在日志中确认 batch size 1、2 实际使用图重放，且 decode 没有退回 eager；
结束后停止测试服务。
该模型只验证执行链路，不具备可用的语言能力。

服务日志在 `build/npu-smoke-server.log`，成功结果在
`build/npu-smoke-result.json`。

本次验证已通过 BF16 矩阵计算、模型加载、KV cache 分配、服务预热和
NPU Graph 捕获与 decode 重放。共验证 4 个请求、每个生成 8 个 token，
重复请求的输出 token 一致；测试服务已停止。
测试使用单卡，prefill 仍走上游普通执行路径，不能据此宣称支持 prefill 图执行。
真实模型及 ECHO 设备算子的图捕获与重放需在对应迁移步骤验证。

本次配置不包含 DeepGEMM，也不意味着 ECHO 的 KV 卸载或融合预取已经移植。
DeepSeek-V3.2、NSA 算子、量化、多卡通信和真实模型性能需要分别验证。

## DeepSeek-V3.2 AWQ 权重

2026-10-08 启动了 `QuantTrio/DeepSeek-V3.2-AWQ` 的下载。
使用 ModelScope 仓库版本 `3a6e8af042a84d4dfd5b09dc6dbdfcd4569beeef`，
保存到 `build/models/DeepSeek-V3.2-AWQ/`。
配置为 AWQ 4-bit、group size 128、zero point，包含 121 个权重分片，
权重总计 361,926,413,312 字节（约 362 GB）。
ModelScope 和 Hugging Face 的全部权重分片大小及 SHA256 清单一致。

下载运行在 tmux 会话 `echo-v32-awq-download`，支持断点续传。
ModelScope SDK 在每个文件完成后校验 SHA256；脚本在全部完成后检查文件大小、
量化配置和权重索引，并将状态标记为 `complete`。
日志和状态文件分别为：

```text
build/models/deepseek-v32-awq-download.log
build/models/deepseek-v32-awq-download-status.json
```

下载进程退出后，可以在仓库根目录恢复下载：

```bash
.venv-npu/bin/python -u scripts/download-deepseek-v32-awq.py
```

下载完成不代表已加载到 NPU。当前分支包含 AWQ 的 Ascend Linear 和 MoE 路径，
该真实模型的多卡加载、NSA 与 NPU Graph 捕获及重放仍待验证。

### 2026-10-09 八卡启动排查

权重下载及 SDK SHA256 校验已完成：130 个文件、121 个权重分片。
在物理 NPU 4–11 上，完整 AWQ 模型以 TP=8 成功加载，每卡模型占用约
45.14 GiB，但服务尚未完成启动或返回真实推理结果。

本次补齐 CANN 编译链的 `decorator==5.3.1` 依赖，并在 DeepSeek NPU
Attention 初始化时，为无 bias 的 Q/K RMSNorm 显式提供 `bias=None`。
这使融合 Q/K norm 能处理 AWQ 模型；单算子已经通过变更输入的多次
Graph 重放与参考计算对比。该验证不能替代完整模型 Graph 验证。

完整模型的 Graph 预热遇到 HCCL/SDMA 错误。独立通信测试中，4–7 和
8–11 两个四卡组分别通过 AllReduce、Graph 捕获和重放；八卡 4–11
的普通通信与 AIV 路径失败。证据保存在 `build/deepseek-v32-awq/`。

按用户要求，启动脚本默认设备改为 8–15（TP=8，decode Graph 开启）：

```bash
./scripts/serve-deepseek-v32-awq.sh
```

此前切换时 14、15 上仍有其他 vLLM 任务，等待其释放后，已于
2026-10-09 09:59 在物理 NPU 8–15 上启动完整 AWQ 模型（TP=8）。
常规通信模式的完整模型预热仍报 SDMA 错误；切换
`HCCL_OP_EXPANSION_MODE=AIV` 后，八个 rank 均完成 batch size 1、2
的 NPU Graph 捕获，HTTP 服务已经就绪。启动脚本默认使用 AIV。

当前服务地址为 `http://127.0.0.1:30032`，模型名为
`deepseek-v3.2-awq`，tmux 会话为 `echo-v32-awq-server`。
`scripts/verify-deepseek-v32-awq.py` 已验证中文算术答案 `42`、
batch size 1/2/1 的生成、2581 token 长输入算术答案，以及
batch size 1、2 的真实 decode Graph 重放（日志 `npu graph: True`）。
重复的温度为 0 的自由问答均正确回答北京，但生成 token 不完全一致；
记录为 `passed_with_token_variation`，尚未确认该差异的原因或逐 token
确定性。首次严格重复 token 检查失败的记录也已保留。

日志与验收记录在 `build/deepseek-v32-awq/server.log`、
`verification.json` 和 `responses.json`。这仅验证 SGLang 原生推理和
**decode Graph**；ECHO KV 卸载/预取路径和 prefill Graph 尚未接入验证。

## 安装来源

CANN 安装器来自 Ascend 官方 OBS，保存于 `build/cann-downloads/`：

```text
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-toolkit_8.5.0_linux-aarch64.run
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-A3-ops_8.5.0_linux-aarch64.run
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-nnal_8.5.0_linux-aarch64.run
```

Toolkit 和 A3 算子包均以普通用户安装到 `build/cann/`。
NNAL 安装前先加载该 Toolkit 的 `set_env.sh`，仅安装 `atb` 组件。
安装器在用户目录创建了安装记录和日志，没有修改 shell 启动文件。

NPU kernel 发布包：

```text
https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.03.31.rc3/sgl-kernel-npu-2026.03.31.rc3-torch2.8.0-py311-cann8.5.0-a3-aarch64.zip
```

按上游安装说明，将虚拟环境内的 `deep_ep/deep_ep_cpp*.so` 链接到
site-packages 顶层，以满足 DeepEP 的导入路径。
Python 依赖使用华为云 PyPI 镜像下载。缓存集中在 `build/` 下。
