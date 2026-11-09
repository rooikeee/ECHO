#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <thread>
#include <vector>
#include <deep_gemm/echo_prefetch.hpp>
using namespace deep_gemm;
uint16_t bf16(float x) {uint32_t bits;std::memcpy(&bits,&x,4);return bits>>16;}
float decode(uint16_t x) {uint32_t bits=uint32_t(x)<<16;float f;std::memcpy(&f,&bits,4);return f;}
struct Case {
    static constexpr uint32_t rows=4,context=1024,hosts=1100,cap=33;
    int32_t active=4;
    uint32_t bytes;
    std::vector<float> thresholds=std::vector<float>(rows,0);
    std::vector<int32_t> stable=std::vector<int32_t>(rows,context),budgets=std::vector<int32_t>(rows,cap),visible=std::vector<int32_t>(rows,context);
    std::vector<int32_t> ids=std::vector<int32_t>(rows*context),map=std::vector<int32_t>(hosts,-1),written=std::vector<int32_t>(hosts,1);
    std::vector<int32_t> counts=std::vector<int32_t>(rows,0),tokens=std::vector<int32_t>(rows*cap,-1),selected_hosts=tokens;
    std::vector<uint16_t> logits=std::vector<uint16_t>(rows*context);
    std::vector<uint8_t> source,staging;
    explicit Case(uint32_t b):bytes(b),source(hosts*b),staging(rows*cap*b,0xA5) {
        for(uint32_t i=0;i<source.size();++i)source[i]=(i*17+11)%251;
        for(uint32_t i=0;i<ids.size();++i) {ids[i]=i%context+1;logits[i]=bf16(float(int(i%19)-9));}
    }
    EchoPrefetchArgs args() {
        return {uintptr_t(thresholds.data()),uintptr_t(stable.data()),uintptr_t(&active),uintptr_t(budgets.data()),
            uintptr_t(ids.data()),uintptr_t(map.data()),uintptr_t(written.data()),uintptr_t(counts.data()),
            uintptr_t(tokens.data()),uintptr_t(selected_hosts.data()),uintptr_t(staging.data()),uintptr_t(source.data()),
            rows,context,hosts,cap,bytes};
    }
    bool eligible(uint32_t row,uint32_t token) const {
        const int32_t h=ids[row*context+token];
        return active>0 && row<uint32_t(active) && stable[row]>0 && token<uint32_t(stable[row]) && token<uint32_t(visible[row]) &&
            decode(logits[row*context+token])>thresholds[row] && echo::valid_host(h,hosts) && written[h] && map[h]==-1;
    }
    void check() {
        for(uint32_t r=0;r<rows;++r) {
            uint32_t n=0;for(uint32_t t=0;t<context;++t) n+=eligible(r,t);
            const uint32_t budget=budgets[r]<=0 ? 0 : std::min(cap,uint32_t(budgets[r]));
            assert(uint32_t(counts[r])==std::min(n,budget));
            std::set<int32_t> seen;
            for(int32_t i=0;i<counts[r];++i) {
                const uint32_t j=r*cap+i;const int32_t token=tokens[j],h=selected_hosts[j];
                assert(token>=0 && uint32_t(token)<context && eligible(r,token));
                assert(seen.insert(token).second);assert(h==ids[r*context+token]);
                assert(std::memcmp(staging.data()+j*bytes,source.data()+h*bytes,bytes)==0);
            }
            for(uint32_t i=counts[r];i<cap;++i) {
                assert(tokens[r*cap+i]==-1 && selected_hosts[r*cap+i]==-1);
                for(uint32_t b=0;b<bytes;++b)assert(staging[(r*cap+i)*bytes+b]==0xA5);
            }
        }
    }
};
int main() {
    uint32_t runs=0;
    for(uint32_t bytes:{1U,31U,128U,1152U,2049U,4097U}) {
        Case c(bytes);
        // Reuse the same buffers across changing metadata, matching graph replay.
        for(uint32_t replay=0;replay<6;++replay) {
            std::fill(c.counts.begin(),c.counts.end(),0);std::fill(c.tokens.begin(),c.tokens.end(),-1);
            std::fill(c.selected_hosts.begin(),c.selected_hosts.end(),-1);std::fill(c.staging.begin(),c.staging.end(),0xA5);
            std::fill(c.map.begin(),c.map.end(),-1);std::fill(c.written.begin(),c.written.end(),1);
            std::fill(c.stable.begin(),c.stable.end(),Case::context);std::fill(c.thresholds.begin(),c.thresholds.end(),0);
            std::fill(c.budgets.begin(),c.budgets.end(),Case::cap);std::fill(c.visible.begin(),c.visible.end(),Case::context);c.active=4;
            if(replay==1) {for(uint32_t h=1;h<Case::hosts;h+=2)c.map[h]=0;c.stable[1]=63;c.active=3;}
            if(replay==2) {c.thresholds[0]=std::numeric_limits<float>::infinity();c.thresholds[1]=NAN;
                c.budgets[2]=1;c.budgets[3]=0;}
            if(replay==3) {for(uint32_t h=1;h<Case::hosts;h+=3)c.written[h]=0;
                for(uint32_t t=0;t<Case::context;t+=5)c.ids[t]=-1;c.ids[21]=Case::hosts+5;c.ids[22]=0;}
            if(replay==4) {std::fill(c.map.begin(),c.map.end(),5);c.active=0;}
            if(replay==5) {c.budgets[0]=10000;c.budgets[1]=-1;c.stable[2]=0;c.stable[3]=7;c.visible={513,1,511,63};}
            const auto map_before=c.map,ids_before=c.ids;
            // Two simultaneously active KV blocks compete for each row budget.
            std::vector<std::thread> blocks;
            for(uint32_t block=0;block<2;++block) blocks.emplace_back([&,block]{
                alignas(32) std::array<uint8_t,EchoPrefetch::kUBBytes> storage{};
                EchoPrefetch prefetch{c.args()};prefetch.init(storage.data());
                for(uint32_t row=0;row<Case::rows;++row) {
                    const uint32_t begin=block*512;
                    if(c.visible[row]>static_cast<int32_t>(begin))
                        prefetch.consume(row,begin,std::min(512U,uint32_t(c.visible[row])-begin),
                                         c.logits.data()+row*Case::context+begin);
                }
                prefetch.finish();
                for(auto& from:event_flags)for(auto& to:from)for(bool flag:to)assert(!flag);
            });
            for(auto& b:blocks)b.join();
            c.check();assert(c.map==map_before && c.ids==ids_before);++runs;
        }
    }
    std::cout<<runs<<" concurrent-block and replay-state cases passed\n";
}
