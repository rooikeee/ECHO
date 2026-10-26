#ifndef ECHO_INDEXER_PREFETCH_H
#define ECHO_INDEXER_PREFETCH_H

#include "kernel_operator.h"

namespace EchoPrefetch {
using namespace AscendC;
using LIServiceVec::SetWaitFlag;

__aicore__ inline void PublishCutoffs(
    __gm__ uint8_t *contextAddr, __gm__ uint8_t *cutoffAddr,
    __gm__ uint8_t *statesAddr, __gm__ uint8_t *topkAddr, __gm__ uint8_t *endsAddr,
    uint32_t batches, TPipe *pipe)
{
    GlobalTensor<int64_t> context;
    GlobalTensor<int32_t> states, topk, ends;
    GlobalTensor<float> cutoffs;
    context.SetGlobalBuffer((__gm__ int64_t *)contextAddr);
    cutoffs.SetGlobalBuffer((__gm__ float *)cutoffAddr);
    states.SetGlobalBuffer((__gm__ int32_t *)statesAddr);
    topk.SetGlobalBuffer((__gm__ int32_t *)topkAddr);
    ends.SetGlobalBuffer((__gm__ int32_t *)endsAddr);
    TBuf<TPosition::VECCALC> buffer;
    pipe->InitBuffer(buffer, 32);
    auto value = buffer.Get<float>();
    const uint32_t queries = ends.GetValue(batches - 1);
    for (uint32_t q = GetBlockIdx() / 2; q < queries; q += GetBlockNum()) {
        uint32_t b = 0;
        while (q >= static_cast<uint32_t>(ends.GetValue(b))) ++b;
        GlobalTensor<float> prediction;
        prediction.SetGlobalBuffer((__gm__ float *)context.GetValue(q * 14));
        if (states.GetValue(b) != -3 && topk.GetValue(q * LIMConfig::TOPK + LIMConfig::TOPK - 1) >= 0) {
            DataCopyPad(value, cutoffs[q * LIMConfig::THRESHOLD_WORKSPACE_STRIDE],
                        {1, 4, 0, 0}, {false, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        } else {
            value.ReinterpretCast<uint32_t>().SetValue(0, 0x7f800000U);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        }
        DataCopyPad(prediction[q], value, {1, 4, 0, 0});
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
}

// Internal descriptor ABI is owned by scripts/echo_indexer_prefetch.py.
// Only one AIV owns each query's budget; no inter-core atomic return is needed.
template <typename T>
__aicore__ inline void Run(
    __gm__ uint8_t *contextAddr, GlobalTensor<float> scores,
    GlobalTensor<int32_t> slots, GlobalTensor<int32_t> entries,
    GlobalTensor<uint32_t> queryEnds, GlobalTensor<int32_t> states,
    uint32_t batches, uint32_t scoreStride, uint32_t tileSize,
    uint32_t sourceCapacity, uint32_t poolSize, TPipe *pipe)
{
    GlobalTensor<int64_t> context;
    context.SetGlobalBuffer((__gm__ int64_t *)contextAddr);
    TBuf<TPosition::VECCALC> scoreBuffer, kvBuffer, metadataBuffer;
    pipe->InitBuffer(scoreBuffer, tileSize * sizeof(float));
    pipe->InitBuffer(kvBuffer, 576 * sizeof(T));
    pipe->InitBuffer(metadataBuffer, 32);
    auto scoreUb = scoreBuffer.Get<float>();
    auto kvUb = kvBuffer.Get<T>();
    auto metaUb = metadataBuffer.Get<int32_t>();
    const uint32_t owner = GetBlockIdx() / 2;
    const uint32_t queries = queryEnds.GetValue(batches - 1);
    for (uint32_t q = owner; q < queries; q += GetBlockNum()) {
        uint32_t b = 0;
        while (q >= queryEnds.GetValue(b)) ++b;
        if (states.GetValue(b) != -1) continue;
        const int32_t row = entries.GetValue(b);
        if (row < 0 || static_cast<uint32_t>(row) >= poolSize) continue;
        const uint32_t base = q * 14;
        GlobalTensor<float> thresholds;
        GlobalTensor<int32_t> budgets, counts, sources, ready, dramTable, lengths;
        GlobalTensor<T> stagingKv, stagingRope, hostKv, hostRope;
        thresholds.SetGlobalBuffer((__gm__ float *)context.GetValue(base));
        budgets.SetGlobalBuffer((__gm__ int32_t *)context.GetValue(base + 1));
        counts.SetGlobalBuffer((__gm__ int32_t *)context.GetValue(base + 2));
        sources.SetGlobalBuffer((__gm__ int32_t *)context.GetValue(base + 3));
        stagingKv.SetGlobalBuffer((__gm__ T *)context.GetValue(base + 4));
        ready.SetGlobalBuffer((__gm__ int32_t *)context.GetValue(base + 5));
        hostKv.SetGlobalBuffer((__gm__ T *)context.GetValue(base + 6));
        hostRope.SetGlobalBuffer((__gm__ T *)context.GetValue(base + 7));
        dramTable.SetGlobalBuffer((__gm__ int32_t *)context.GetValue(base + 8));
        stagingRope.SetGlobalBuffer((__gm__ T *)context.GetValue(base + 9));
        lengths.SetGlobalBuffer((__gm__ int32_t *)context.GetValue(base + 13));
        const uint32_t tableWidth = context.GetValue(base + 10);
        const uint32_t hostBlocks = context.GetValue(base + 11);
        const uint32_t capacity = context.GetValue(base + 12);
        const int32_t requested = budgets.GetValue(q);
        const int32_t length = lengths.GetValue(b);
        const float threshold = thresholds.GetValue(q);
        if (requested <= 0 || length <= 0 || static_cast<uint32_t>(length) > sourceCapacity ||
            static_cast<uint32_t>(requested) > capacity || !(threshold < 3.402823466e38F)) continue;
        uint32_t count = 0;
        for (uint32_t first = 0; first < static_cast<uint32_t>(length) && count < requested; first += tileSize) {
            int32_t published;
            do {
                DataCopyPad(metaUb, ready[static_cast<uint64_t>(q) * sourceCapacity + first],
                            {1, 4, 0, 0}, {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                published = metaUb.GetValue(0);
            } while (published == 0);
            const uint32_t valid = first + tileSize > length ? length - first : tileSize;
            DataCopyPad(scoreUb, scores[static_cast<uint64_t>(q) * scoreStride + first],
                        {1, static_cast<uint16_t>(valid * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            for (uint32_t i = 0; i < valid && count < requested; ++i) {
                const uint32_t source = first + i;
                if (!(scoreUb.GetValue(i) > threshold) ||
                    slots.GetValue(static_cast<uint64_t>(row) * sourceCapacity + source) >= 0) continue;
                if (source / 128 >= tableWidth) continue;
                const int32_t block = dramTable.GetValue(static_cast<uint64_t>(b) * tableWidth + source / 128);
                if (block < 0 || static_cast<uint32_t>(block) >= hostBlocks) continue;
                const uint64_t hostRow = static_cast<uint64_t>(block) * 128 + source % 128;
                DataCopyPad(kvUb, hostKv[hostRow * 512], {1, 512 * sizeof(T), 0, 0}, {false, 0, 0, 0});
                DataCopyPad(kvUb[512], hostRope[hostRow * 64], {1, 64 * sizeof(T), 0, 0}, {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
                const uint64_t destination = static_cast<uint64_t>(q) * capacity + count;
                DataCopyPad(stagingKv[destination * 512], kvUb, {1, 512 * sizeof(T), 0, 0});
                DataCopyPad(stagingRope[destination * 64], kvUb[512], {1, 64 * sizeof(T), 0, 0});
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                metaUb.SetValue(0, static_cast<int32_t>(source));
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(sources[destination], metaUb, {1, 4, 0, 0});
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                ++count;
            }
        }
        metaUb.SetValue(0, static_cast<int32_t>(count));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(counts[q], metaUb, {1, 4, 0, 0});
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
}
} // namespace EchoPrefetch
#endif
