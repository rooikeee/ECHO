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
    // RunPrefetch resets the pipe before calling us. Keep all allocations
    // within C220's 184-KiB tensor region, leaving its 8-KiB API reserve alone.
    // With A=align64(tileSize), M=align32(A/8), filtering needs 20*A + 3*M.
    // Separate CKV[64,512], rope[64,64], ID[64] and ready/count scratch bring
    // the total to 84,448 bytes for A=512/BF16, replacing the single KV row.
    // Size and align in 64 bits before narrowing; retain the old allocation
    // and copy path if the batched buffers do not fit.
    constexpr uint32_t UB_BYTES = 184U * 1024U;
    constexpr uint32_t STAGING_ROWS = 64U;
    const uint64_t alignedTile = (static_cast<uint64_t>(tileSize) + 63U) / 64U * 64U;
    const uint64_t maskBytes = (alignedTile / 8U + 31U) / 32U * 32U;
    const uint64_t filterBytes = 5U * alignedTile * sizeof(uint32_t) + 3U * maskBytes;
    const uint64_t kvBytes = (576ULL * sizeof(T) + 31U) / 32U * 32U;
    const bool vectorFilter = tileSize != 0U && filterBytes + kvBytes + 32U <= UB_BYTES;
    const uint64_t scoreBytes = vectorFilter ? filterBytes :
        (static_cast<uint64_t>(tileSize) * sizeof(float) + 31U) / 32U * 32U;
    const uint64_t stagingKvBytes =
        (static_cast<uint64_t>(STAGING_ROWS) * 512U * sizeof(T) + 31U) / 32U * 32U;
    const uint64_t stagingRopeBytes =
        (static_cast<uint64_t>(STAGING_ROWS) * 64U * sizeof(T) + 31U) / 32U * 32U;
    const uint64_t sourceBytes =
        (static_cast<uint64_t>(STAGING_ROWS) * sizeof(int32_t) + 31U) / 32U * 32U;
    const bool batchBuffers = tileSize != 0U &&
        scoreBytes + stagingKvBytes + stagingRopeBytes + sourceBytes + 32U <= UB_BYTES;
    TBuf<TPosition::VECCALC> scoreBuffer, kvBuffer, ropeBuffer, sourceBuffer, metadataBuffer;
    pipe->InitBuffer(scoreBuffer, static_cast<uint32_t>(scoreBytes));
    // The first 576 elements also serve the original path for larger capacities.
    pipe->InitBuffer(kvBuffer, static_cast<uint32_t>(batchBuffers ? stagingKvBytes : kvBytes));
    if (batchBuffers) {
        pipe->InitBuffer(ropeBuffer, static_cast<uint32_t>(stagingRopeBytes));
        pipe->InitBuffer(sourceBuffer, static_cast<uint32_t>(sourceBytes));
    }
    pipe->InitBuffer(metadataBuffer, 32);
    auto scoreUb = scoreBuffer.Get<float>();
    auto kvUb = kvBuffer.Get<T>();
    auto metaUb = metadataBuffer.Get<int32_t>();
    LocalTensor<T> ropeUb;
    LocalTensor<int32_t> sourceUb;
    if (batchBuffers) {
        ropeUb = ropeBuffer.Get<T>();
        sourceUb = sourceBuffer.Get<int32_t>();
    }
    LocalTensor<int32_t> slotUb, signUb, progressionUb, compactedUb;
    LocalTensor<uint8_t> scoreMask, slotMask, candidateMask;
    if (vectorFilter) {
        const uint32_t aligned = static_cast<uint32_t>(alignedTile);
        const uint32_t maskSize = static_cast<uint32_t>(maskBytes);
        auto words = scoreBuffer.Get<int32_t>();
        slotUb = words[aligned];
        signUb = words[2U * aligned];
        progressionUb = words[3U * aligned];
        compactedUb = words[4U * aligned];
        scoreMask = words[5U * aligned].ReinterpretCast<uint8_t>();
        slotMask = scoreMask[maskSize];
        candidateMask = scoreMask[2U * maskSize];
        // Build a fixed progression once per invocation, outside both loops;
        // varying ArithProgression lengths between tiles is unsafe on C220.
        ArithProgression<int32_t>(progressionUb, 0, 1, aligned);
        PipeBarrier<PIPE_V>();
    }
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
        // Re-evaluate the descriptor/budget on every replay. The captured reset
        // owns skipped queries; only this query's [0, count) staging is published.
        const bool batchQuery = batchBuffers && capacity <= STAGING_ROWS;
        uint32_t count = 0;
        for (uint32_t first = 0; first < static_cast<uint32_t>(length) && count < requested; first += tileSize) {
            // Finish the previous tile's vector reads before reusing its DMA
            // input rows. The scalar fence also protects repeated ready polls
            // and score/meta reads from the scalar fallback on the next tile.
            if (vectorFilter) SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            int32_t published;
            do {
                SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                DataCopyPad(metaUb, ready[static_cast<uint64_t>(q) * sourceCapacity + first],
                            {1, 4, 0, 0}, {false, 0, 0, 0});
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                published = metaUb.GetValue(0);
            } while (published == 0);
            const uint32_t valid = first + tileSize > length ? length - first : tileSize;
            DataCopyPad(scoreUb, scores[static_cast<uint64_t>(q) * scoreStride + first],
                        {1, static_cast<uint16_t>(valid * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            if (vectorFilter) {
                DataCopyPad(slotUb, slots[static_cast<uint64_t>(row) * sourceCapacity + first],
                            {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0}, {false, 0, 0, 0});
            }
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            uint32_t candidates = valid;
            if (vectorFilter) {
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                const uint32_t alignedValid = (valid + 63U) / 64U * 64U;
                if (valid < alignedValid) {
                    // MTE2_S above covers DMA padding before scalar stores.
                    // valid may be unaligned, so Duplicate at [valid] can fault.
                    for (uint32_t i = valid; i < alignedValid; ++i) {
                        scoreUb.SetValue(i, 0.0F);
                        slotUb.SetValue(i, 0);
                    }
                }
                // Order tail stores and the previous compacted-ID scalar reads
                // before vector work reuses the scratch on this tile/replay.
                SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
                const uint32_t maskWords = static_cast<uint32_t>(maskBytes) / sizeof(uint16_t);
                Duplicate(scoreMask.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), maskWords);
                Duplicate(slotMask.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), maskWords);
                Duplicate(candidateMask.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), maskWords);
                PipeBarrier<PIPE_V>();
                CompareScalar(scoreMask, scoreUb, threshold, CMPMODE::GT, alignedValid);
                // Unsigned shift recognizes every negative slot sentinel; C220
                // int32 CompareScalar supports EQ, not a signed LT comparison.
                ShiftRight(signUb.ReinterpretCast<uint32_t>(), slotUb.ReinterpretCast<uint32_t>(), 31U, alignedValid);
                PipeBarrier<PIPE_V>();
                CompareScalar(slotMask, signUb, 1, CMPMODE::EQ, alignedValid);
                PipeBarrier<PIPE_V>();
                And(candidateMask.ReinterpretCast<uint16_t>(), scoreMask.ReinterpretCast<uint16_t>(),
                    slotMask.ReinterpretCast<uint16_t>(), maskWords);
                PipeBarrier<PIPE_V>();
                // Zero slot tails cannot match even when threshold is negative.
                // Clear mask padding above and use the ordered C220 count-mode
                // compaction parameters already used by MtpMissUnion.
                GatherMaskParams compactParams;
                compactParams.repeatTimes = 1;
                compactParams.src0BlockStride = 1;
                compactParams.src0RepeatStride = LIServiceVec::B32_VEC_REPEAT_STRIDE;
                compactParams.src1RepeatStride = LIServiceVec::B32_VEC_REPEAT_STRIDE;
                uint64_t gathered = 0U;
                GatherMask(compactedUb.ReinterpretCast<uint32_t>(), progressionUb.ReinterpretCast<uint32_t>(),
                           candidateMask.ReinterpretCast<uint32_t>(), true, alignedValid, compactParams, gathered);
                PipeBarrier<PIPE_V>();
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                candidates = static_cast<uint32_t>(gathered);
            }
            // GatherMask preserves ascending local IDs. Invalid host mappings
            // do not spend budget: count advances only after a successful copy.
            for (uint32_t i = 0; i < candidates && count < requested; ++i) {
                const uint32_t source = first + (vectorFilter ? static_cast<uint32_t>(compactedUb.GetValue(i)) : i);
                if (!vectorFilter && (!(scoreUb.GetValue(i) > threshold) ||
                    slots.GetValue(static_cast<uint64_t>(row) * sourceCapacity + source) >= 0)) continue;
                if (source / 128 >= tableWidth) continue;
                const int32_t block = dramTable.GetValue(static_cast<uint64_t>(b) * tableWidth + source / 128);
                if (block < 0 || static_cast<uint32_t>(block) >= hostBlocks) continue;
                const uint64_t hostRow = static_cast<uint64_t>(block) * 128 + source % 128;
                if (batchQuery) {
                    // Distinct rows let both real Host reads remain queued;
                    // tile/ready fences are unchanged and may still drain MTE2.
                    DataCopyPad(kvUb[count * 512U], hostKv[hostRow * 512],
                                {1, 512 * sizeof(T), 0, 0}, {false, 0, 0, 0});
                    DataCopyPad(ropeUb[count * 64U], hostRope[hostRow * 64],
                                {1, 64 * sizeof(T), 0, 0}, {false, 0, 0, 0});
                    sourceUb.SetValue(count, static_cast<int32_t>(source));
                } else {
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
                }
                ++count;
            }
        }
        if (batchQuery && count != 0U) {
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            const uint64_t destination = static_cast<uint64_t>(q) * capacity;
            // Explicit wide byte lengths: 64 BF16 CKV rows occupy 65,536 bytes.
            const DataCopyExtParams kvCopy{
                1, static_cast<uint32_t>(static_cast<uint64_t>(count) * 512U * sizeof(T)), 0, 0, 0};
            const DataCopyExtParams ropeCopy{
                1, static_cast<uint32_t>(static_cast<uint64_t>(count) * 64U * sizeof(T)), 0, 0, 0};
            const DataCopyExtParams sourceCopy{
                1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(stagingKv[destination * 512], kvUb, kvCopy);
            DataCopyPad(stagingRope[destination * 64], ropeUb, ropeCopy);
            // ID storage is padded/aligned in UB; GM receives exactly count IDs.
            DataCopyPad(sources[destination], sourceUb, sourceCopy);
            // Finish all three publications before exposing count or reusing UB.
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        metaUb.SetValue(0, static_cast<int32_t>(count));
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(counts[q], metaUb, {1, 4, 0, 0});
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
}
} // namespace EchoPrefetch
#endif
