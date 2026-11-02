# CopySFA 内部消费预取缓冲（910 / v2）

本轮只做融合算子的实现、算子输出校验与单算子 profiling。没有运行整模型、
完整 MTP speculative round 或服务端到端测试。MTP3仍表示每请求4行query。

## 改动

LIM保留原来的Host→预取缓冲搬运，以及原有TopK、miss前缀和slot分配。
CopySFA新增三个可选显式输入：`prefetch_map`、`prefetch_kv`、`prefetch_rope`。
不提供时保持原接口行为；提供时，原miss前缀中的token可从预取缓冲或Host读取。

```text
LIM：打分 / TopK / slot分配 + 原有Host→staging
      └─ 发布请求共享的 token → staging位置
CopySFA：
  正式热池命中 → 正式slot → UB → attention
  预取已就绪   → staging  → UB → attention，并写回正式slot
  未预取       → Host     → UB → attention，并写回正式slot
```

Python侧不再调用 `prefetch.consume()`，因此移除了原来的候选匹配、独立KV提交、
前缀和、partition、scatter和union输出复制等20个中间kernel。
CopySFA内部仍然有必须完成的查表、MTP写回归属和KV写回成本，不能把这些成本说成零。

## 状态与生命周期

复用已有 `[T, source_capacity]` 的ready缓冲。LIM内部：

- 0：score tile尚未就绪；1：score tile已就绪。这个含义供原预取流水轮询。
- 在已有的AIV同步点之后，每个请求的唯一owner读取各query的有界候选列表，
  在请求第一行ready中写入 `2 + flattened_staging_row`。
- 候选按query逆序发布，同一token有多个候选时，最低query编号的副本成为公共来源。
- 标记发布发生在KV DMA完成之后；写出前清理并失效scalar DCache。
- 每次Graph重放仍在LIM前执行原有ready/count/source重置，地址保持固定。

这里的2是预取映射编码，不是正式反向表里的slot2。正式反向表的slot语义不变。
CopySFA每个miss只查一次请求共享映射，不再与四行共256个候选做矩阵式比较。
本轮没有增加第二次Host KV搬运。

## MTP写回归属

CopySFA内部每个请求由一个AIV在UB中建立 `slot → 最低选中query` 表，仅遍历各query的
原miss前缀。经内部MIX barrier后，只有该query的第一个head group提交正式slot。
其他query/head group从稳定的staging或Host源读取，供自己的attention计算，
不轮询正式slot是否已写回，也不对hit token执行CAS。

owner表使用CopySFA内部workspace，按512字节对齐。在B24/C8192场景中每请求表宽为
65个HBM页×128（包含tail槽位）；总workspace增量约780KiB。
全batch没有miss时，跳过owner表建立及该额外barrier，走原hit gather路径。
混合first-fill保留原先的批量回搬路径及其同步；该场景不消费staging。

## 调用方式

```python
prefetch.indexer(cache, raw_outputs)
# 不调用consume，不重排raw_outputs。
torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention(
    query_rope, query, query_ends, actual_kv, cache_tokens,
    raw_outputs[1], raw_outputs[0], raw_outputs[2],
    raw_outputs[3], raw_outputs[4], raw_outputs[5],
    hbm_table, dram_table, hbm_rope, hbm_kv, host_rope, host_kv,
    scale, attention_out, **prefetch.copy_sfa_inputs(),
)
```

staging的owner、registered Host缓冲和ready/count/source存储必须存活到Graph销毁。
CopySFA的三个新输入是显式tensor，不能省略其中一个；dtype、shape和设备有接口校验。
原LIM的Host预取描述符仍有间接指针，未在本轮变更为全显式ABI。

## 单算子 profiling 口径

使用 `--profile-scope operators`：

- LIM单独捕获和重放；开启预取时包含原来的ready/count/source重置。
- CopySFA单独捕获和重放；LIM、预取搬运和原始miss元数据准备都在计时外完成。
- 每次CopySFA前在计时外恢复相同的正式HBM KV种子，staging保持有效。
- 输出分别是两份单算子trace，不将两份时间相加宣称端到端收益。
- 小配置的事件时间可能被主机提交间隙拉长，算子耗时以profiler的kernel任务时间为准。

B24、MTP3、143360 token（140×1024）、热池8192、每query预算64；分别测每请求400个
union miss和全命中。最终实测文件在主工作区 `profiling/prefetch_v2/`。
6号设备一次全命中采集中被其他进程占用，该轮标记为不可用于性能比较；最终换7号设备。

## 验证

- 精确比较原始LIM输出、cache映射和完整KV/KPE；attention对独立CPU参考检查。
- Graph重放检查改变query、缩短长度、预算0/7/64、MTP重复候选、混合first-fill、
  恢复steady及persistent all-hit；同时检查捕获地址不变。
- 在LIM后故意把已预取token的Host KV/KPE改坏，再运行CopySFA。最终KV与attention
  仍正确，证明使用了staging，未悄悄退回Host。
- 40次边界重放覆盖INT_MIN/-1槽位、严格阈值、非对齐尾块和真实Host payload。
- 128个attention head下验证写回owner规则；8项错误输入契约测试通过。
- 本轮运行正确性/性能配置为BF16/MTP3；FP16设备实例已编译，未扩展到完整FP16测试矩阵。

## 最终单算子结果

物理device7，40次事件重放、10个active profiling step；表中是profiler kernel任务
平均耗时，单位ms。仅比较同一轮、同一设备的数据。

| 场景 | 模式 | LIM | CopySFA |
| --- | --- | ---: | ---: |
| 每请求400个miss | baseline | 2.106 | 0.413 |
| 每请求400个miss | prefetch | 2.957 | 0.540 |
| 全命中 | baseline | 1.651 | 0.414 |
| 全命中 | prefetch | 2.494 | 0.422 |

原来中间20个consumer kernel已从新调用路径移除。CopySFA现在同时承担映射查找、
MTP写回归属和预取提交，因此它仍比无预取CopySFA有额外成本；不宣称它本身必然更快。
全命中CopySFA接近无预取路径，但LIM依然会为预测扫描分数/驻留表，这项成本未在本轮消除。
本轮没有用两个单算子时间相加推算端到端加速。

数据、源码及二进制哈希见 [结果JSON](../forge/lim910/copysfa-v2-results-20261102.json)。

## 构建与复现

需要同步重建CANN算子包与Torch绑定；只替换设备kernel不够，因为CopySFA新增了三个输入。
构建保持在隔离checkout中，复用原环境的只读依赖对象：

```bash
bash scripts/build-copysfa-prefetch.sh /home/l50066329/code/ECHO/build/vllm-native
ECHO_NPU_DEVICE=7 ./scripts/vllm-ascend-run.sh python \
  scripts/collect-echo-prefetch-profile-npu.py \
  --batches 24 --mtp 3 --context 143360 --hot-tokens 8192 \
  --misses-per-request 400 --modes baseline context-budget-0 prefetch-64-history \
  --prefetch-consumer copysfa --profile-scope operators --check-staging-source \
  --repeats 40 --profile-repeats 10 --attention-reference --output build/new-operator-profile
```

默认仍保留Python consumer用于旧实验复现。新调用必须显式选择copysfa consumer或传入
`copy_sfa_inputs()`；没有新增环境开关、独立算子或模型runner修改。
