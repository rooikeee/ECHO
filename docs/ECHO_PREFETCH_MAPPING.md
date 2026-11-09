# ECHO Ascend 的 resident / staging 统一位置映射

这次修改在 `deep_gemm.echo.EchoPrefetchBuffers` 的预取过程中更新
`host_to_device`，使后续 TopK 输出的 host ID 能直接定位正式缓存或 staging。
它不实现最终 TopK、正式槽位分配、LRU 淘汰或 TopK 后的 recall 搬运。

## 编码

调用方必须显式传入真实的正式缓存槽位数 `resident_capacity=D`：

```python
buffers = EchoPrefetchBuffers.allocate(
    rows, max_context, prefetch_capacity, source,
    resident_capacity=resident_cache.shape[0],
    # DDR 模式另传 source_dva 和 source_kind="MemFabric-DDR"
)
```

设 `P=prefetch_capacity`、`R=rows`，映射是 int32：

| 值 | 含义 |
| --- | --- |
| `-1` | miss |
| `-2` | 某个 SIMT lane 已占用该 host token，正在决定是否预取；仅在执行过程中短暂出现 |
| `0 <= value < D` | 正式缓存槽位，0 也可以是有效槽位 |
| `D <= value < D+R*P` | staging 位置，`flat=value-D`，`owner_row=flat//P`，`slot=flat%P` |

要求 `D>0`、`D+R*P <= INT32_MAX`，超出范围在 Host 接口报错。源 host ID=0 仍是 padding，
与正式 cache 的槽位 0 含义不同。

例如 `D=1000`、`P=64`，host token 100 被 query 1 预取到 slot 5：

```text
host_to_device[100] = 1000 + 1*64 + 5 = 1069
```

query 0 的最终 TopK 也需要这个 host token 时，可以解码到
`staging[1,5,:]`。位置编码包含 owner query 行，因此不依赖“消费者 query 必须等于生产者 query”。
读取 staging 前可以核对 `hosts[owner_row,slot] == host_id` 及 `slot < counts[owner_row]`。

**映射和 counts 都不是 DMA ready 通知。**只有整个预取算子完成之后才允许消费；
同 stream 后继 kernel 或跨 stream 显式完成事件提供这个边界。不能在看到正偏移时提前读取。

## 预取中的原子登记与去重

1. 超过阈值的有效 token 先检查 written，再执行 `CAS(map[h], -1, -2)`。
2. 只有 CAS 成功的 lane 参与本 tile 的预算分配；看到 resident、已有 staging 或占用标记的 lane 跳过。
3. 成功分配到 staging 槽位的 token 写入 `tokens/hosts`，然后原子发布 `D+row*P+slot`。
4. 没分配到槽位的占用用 CAS 退回 `-1`，不会留下虚假的 staging 地址。
5. owner AIV 根据成功分配的列表执行搬运，结束时等待 outstanding DMA。

同一 host ID 在不同 query、不同 KV split 或同一 tile 多个逻辑位置重复出现时，
至多一个 owner 在本次成功 launch 中搬运它。后续各 query 可以共享其 staging 结果。
`counts[row]` 表示该行实际拥有的预取记录数，不再表示该行所有可命中 staging 的 token 数。

这是有界预测预取，不保证容量用满或选择固定子集。当某个占用最终因行预算不足被释放时，
其他看到占用而已跳过的查询不会自旋重试；该 token 可以留为真正 miss，交由未来精确 TopK
后的 recall 处理。这不会伪造 hit，也不会因跨块等待而死锁。

## 每轮清理与 Graph

`allocate()` 用 0 初始化 counts，用 -1 初始化 tokens/hosts。不能给原始 C++ 入口传未初始化的 hosts。
`run()` 不再在 Python 中先清掉旧 hosts；native 调用顺序为：

```text
logits 置 -inf
→ 独立 cleanup kernel：按上一轮 hosts 清理精确匹配的 staging tag，并重置 counts/tokens/hosts
→ metadata kernel：按本轮长度和分组重算调度
→ MQA + 选择 + 原子登记 + staging 搬运
→ 后续消费者
```

cleanup 遍历 `R*P` 个预取槽位，不扫描整个 host pool。对每个旧条目仅执行
`CAS(map[old_host], D+old_flat_slot, -1)`：

- 仍然指向该旧 staging 槽位：清为 miss。
- 已由消费者提升为正式槽位：保留。
- 已失效或被改为其他位置：保留，不覆盖。

清理覆盖所有旧行，与本轮 active、length、budget 无关。所以本轮 active=0 或 budget=0
仍会清除旧临时位置。cleanup 和重置被 Graph 捕获，每次 replay 都执行。

同一个 buffers 实例绑定同一张映射表，保持 D 和 staging/output 的地址、shape、stride 不变。
Graph 存活期间不能换表、换容量或释放源池。并发 kernel 不得同时修改同一张映射表；
不同 buffers 实例不能重叠拥有该表的临时编码空间。缓存管理只能在预取 launch 和消费者之外更新。

最后一次使用结束后，先等待消费者，再调用：

```python
buffers.clear_prefetch_mappings()  # 同 stream；保留已提升为 resident 的条目
# 丢弃 buffers 或释放其张量前，需要保证这个清理也已完成。
torch.npu.synchronize()
```

此方法幂等。再次 `run()` 仍使用原映射表。重建 buffers/Graph 前先完成上述清理。
不通过析构器隐式发起设备操作。发生设备异常导致 kernel 中途失败时，本协议不宣称自动恢复；
应重建/修复缓存状态，不能继续把残留的 -2 当作有效命中。

## 接口与验证

- Python `allocate()` 新增必填 `resident_capacity`。
- `_C.echo_paged_mqa_prefetch_out` 尾部新增 `resident_capacity`，映射张量变为可写输入输出。
- 新增 `_C.echo_clear_prefetch_mappings`，用于显式结束 staging 生命周期。
- 需要重编 Host 扩展；旧 `_C` 与这版 Python 接口不兼容。
- `scripts/check-echo-deepgemm-950.py` 新增 `--resident-capacity`，默认 4096，
  并检查跨 query 去重、偏移解码、重复 Graph replay、禁用轮清理和 resident 提升保留。

已通过 66 个 CPU 并发/状态场景和 4 项 Python 接口测试（含 ASan/UBSan）。
CPU 检查直接使用生产 select/cleanup/搬运控制代码，128-lane barrier/shuffle 模拟，
四个并发计算块，覆盖重复 host ID、预算竞争、旧 tag 清理、resident 提升、偏移边界及重复调用。
Python 测试确认旧 hosts 在 native cleanup 前未被清空、固定地址契约和 ABI 参数转发。

所有设备实现仍仅面向 **950 / CANN 9.20**。本机只做 CPU/源码检查；
950 编译、设备原子与 cache 一致性、Graph 重放及性能尚未验证。新清理 kernel 和 host CAS
有成本，不能据此宣称整条路径已经加速。
