#pragma once
#include <simt_api/asc_simt.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstddef>
using bfloat16_t=uint16_t;
enum event_t {EVENT_ID0,EVENT_ID1,EVENT_ID2,EVENT_ID3,EVENT_ID4,EVENT_ID5,EVENT_ID6,EVENT_ID7};
enum pipe_t {PIPE_V,PIPE_S,PIPE_MTE2,PIPE_MTE3};
enum class asc_load_l2_cache_mode {NOTALLOC_KEEP};
enum class asc_store_l2_cache_mode {NOTALLOC_CLEAN};
inline thread_local bool event_flags[4][4][8]{};
inline void asc_sync_notify(pipe_t from,pipe_t to,event_t event) {
    auto& f=event_flags[from][to][event];assert(!f);f=true;
}
inline void asc_sync_wait(pipe_t from,pipe_t to,event_t event) {
    auto& f=event_flags[from][to][event];assert(f);f=false;
}
inline void asc_copy_ub2gm_align(uint8_t* dst,uint8_t* src,uint32_t rows,uint32_t bytes,
                               asc_store_l2_cache_mode,uint32_t,uint32_t) {
    assert(rows==1);assert(reinterpret_cast<uintptr_t>(src)%32==0);
    // Transfer may only consume UB after its MTE2->MTE3 event was consumed.
    assert(!event_flags[PIPE_MTE2][PIPE_MTE3][6] && !event_flags[PIPE_MTE2][PIPE_MTE3][7]);
    std::memcpy(dst,src,bytes);
}
namespace deep_gemm {
using std::min;
inline void copy_gm_to_ub(uint8_t* dst,uint8_t* src,uint32_t rows,uint32_t bytes,uint32_t,
                         asc_load_l2_cache_mode) {
    assert(rows==1);assert(reinterpret_cast<uintptr_t>(dst)%32==0);std::memcpy(dst,src,bytes);
}
}
