/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef PREFETCH_COPY_OWNERS_H
#define PREFETCH_COPY_OWNERS_H
#include "kernel_operator.h"

namespace PrefetchCopy {
using namespace AscendC;
constexpr uint32_t TOPK = 2048U;

// One AIV per request builds a slot->first-query map in UB. No atomics or
// polling: attention starts after the enclosing MIX barrier. Only selected
// miss slots are read; all-hit requests skip this work entirely.
__aicore__ inline void PrepareOwners(
    __gm__ uint8_t *slotsAddr, __gm__ uint8_t *countsAddr,
    __gm__ uint8_t *endsAddr, __gm__ uint8_t *unionCountsAddr,
    __gm__ uint8_t *ownersAddr, uint32_t batches, uint32_t stride, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() % 2U != 0U) return;
        GlobalTensor<int32_t> slots, counts, ends, unionCounts, owners;
        slots.SetGlobalBuffer((__gm__ int32_t *)slotsAddr);
        counts.SetGlobalBuffer((__gm__ int32_t *)countsAddr);
        ends.SetGlobalBuffer((__gm__ int32_t *)endsAddr);
        unionCounts.SetGlobalBuffer((__gm__ int32_t *)unionCountsAddr);
        owners.SetGlobalBuffer((__gm__ int32_t *)ownersAddr);
        TBuf<TPosition::VECCALC> ownerBuffer, slotBuffer;
        pipe->InitBuffer(ownerBuffer, stride * sizeof(int32_t));
        pipe->InitBuffer(slotBuffer, TOPK * sizeof(int32_t));
        auto ownerUb = ownerBuffer.Get<int32_t>();
        auto slotUb = slotBuffer.Get<int32_t>();
        for (uint32_t b = GetBlockIdx() / 2U; b < batches; b += GetBlockNum()) {
            if (unionCounts.GetValue(b) == 0) continue;
            Duplicate(ownerUb, -1, stride);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            const int32_t begin = b == 0 ? 0 : ends.GetValue(b - 1U);
            const int32_t end = ends.GetValue(b);
            for (int32_t q = begin; q < end; ++q) {
                const int32_t count = counts.GetValue(q);
                ASSERT_MSG(count >= 0 && count <= static_cast<int32_t>(TOPK), "invalid query miss count");
                if (count <= 0 || count > static_cast<int32_t>(TOPK)) continue;
                SetFlag<HardEvent::S_MTE2>(0);
                WaitFlag<HardEvent::S_MTE2>(0);
                DataCopyPad(slotUb, slots[static_cast<uint64_t>(q) * TOPK],
                            DataCopyExtParams{1, static_cast<uint32_t>(count) * 4U, 0, 0, 0},
                            DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                for (int32_t i = 0; i < count; ++i) {
                    const int32_t slot = slotUb.GetValue(i);
                    if (slot >= 0 && static_cast<uint32_t>(slot) < stride && ownerUb.GetValue(slot) < 0) {
                        ownerUb.SetValue(slot, q);
                    }
                }
            }
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            DataCopyPad(owners[static_cast<uint64_t>(b) * stride], ownerUb,
                        DataCopyExtParams{1, stride * 4U, 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }
}
} // namespace PrefetchCopy
#endif
