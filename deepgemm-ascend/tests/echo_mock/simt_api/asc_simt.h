#pragma once
#include <array>
#include <barrier>
#include <cstdint>
#include <thread>
#include <vector>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(...)
struct ThreadIdx { uint32_t x; };
inline thread_local ThreadIdx threadIdx{};
struct SimtContext {
    std::barrier<> block{128}, warp0{32},warp1{32},warp2{32},warp3{32};
    uint32_t shuffle[4][32]{};
    std::barrier<>& warp(uint32_t id) {
        switch(id) {case 0:return warp0;case 1:return warp1;case 2:return warp2;default:return warp3;}
    }
};
inline thread_local SimtContext* simt_context;
inline uint32_t asc_shfl_up(uint32_t value,uint32_t delta,uint32_t) {
    const auto w=threadIdx.x/32,l=threadIdx.x%32;
    simt_context->shuffle[w][l]=value;
    simt_context->warp(w).arrive_and_wait();
    const auto result=l>=delta ? simt_context->shuffle[w][l-delta] : value;
    simt_context->warp(w).arrive_and_wait();
    return result;
}
inline void asc_syncthreads() {simt_context->block.arrive_and_wait();}
struct dim3 {uint32_t x; explicit dim3(uint32_t n):x(n){}};
template <auto Function,typename... Args>
void asc_vf_call(dim3 threads,Args... args) {
    if(threads.x!=128) std::abort();
    SimtContext context;
    std::vector<std::thread> workers;
    for(uint32_t lane=0;lane<128;++lane) workers.emplace_back([&,lane]{
        threadIdx.x=lane;simt_context=&context;Function(args...);
    });
    for(auto& w:workers)w.join();
}
