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
#include <deep_gemm/echo_prefetch_clear.hpp>
using namespace deep_gemm;
uint16_t bf16(float x) {uint32_t bits;std::memcpy(&bits,&x,4);return bits>>16;}
float decode(uint16_t x) {uint32_t bits=uint32_t(x)<<16;float f;std::memcpy(&f,&bits,4);return f;}
struct Case {
    static constexpr uint32_t rows=4,context=1024,hosts=4500,cap=33,device_capacity=1000;
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
        for(uint32_t i=0;i<ids.size();++i) {ids[i]=i+1;logits[i]=bf16(float(int(i%19)-9));}
    }
    EchoPrefetchArgs args() {
        return {uintptr_t(thresholds.data()),uintptr_t(stable.data()),uintptr_t(&active),uintptr_t(budgets.data()),
            uintptr_t(ids.data()),uintptr_t(map.data()),uintptr_t(written.data()),uintptr_t(counts.data()),
            uintptr_t(tokens.data()),uintptr_t(selected_hosts.data()),uintptr_t(staging.data()),uintptr_t(source.data()),
            rows,context,hosts,cap,bytes,device_capacity};
    }
    bool eligible(uint32_t row,uint32_t token,const std::vector<int32_t>& before) const {
        const int32_t h=ids[row*context+token];
        return active>0 && row<uint32_t(active) && stable[row]>0 && token<uint32_t(stable[row]) && token<uint32_t(visible[row]) &&
            decode(logits[row*context+token])>thresholds[row] && echo::valid_host(h,hosts) && written[h] && before[h]==-1;
    }
    void clear() {
        // Execute the actual reset VF with two concurrent blocks, not a model.
        std::thread a([&]{asc_vf_call<echo::clear_vf>(dim3(128),args(),0,2);});
        std::thread b([&]{asc_vf_call<echo::clear_vf>(dim3(128),args(),1,2);});
        a.join();b.join();
        for(auto n:counts)assert(n==0);
        for(auto h:selected_hosts)assert(h==-1);
        for(auto t:tokens)assert(t==-1);
    }
    void execute() {
        std::vector<std::thread> blocks;
        // Different query rows and KV splits all compete concurrently for host
        // ownership, as well as the per-row staging budget.
        for(uint32_t group=0;group<4;++group)blocks.emplace_back([&,group]{
            alignas(32) std::array<uint8_t,EchoPrefetch::kUBBytes> storage{};
            EchoPrefetch prefetch{args()};prefetch.init(storage.data());
            const uint32_t begin=(group%2)*512;
            for(uint32_t row=group/2;row<rows;row+=2)
                if(visible[row]>static_cast<int32_t>(begin))
                    prefetch.consume(row,begin,std::min(512U,uint32_t(visible[row])-begin),
                                     logits.data()+row*context+begin);
            prefetch.finish();
            for(auto& from:event_flags)for(auto& to:from)for(bool flag:to)assert(!flag);
        });
        for(auto& b:blocks)b.join();
    }
    void check(const std::vector<int32_t>& before,bool distinct) {
        std::set<int32_t> seen_hosts;
        for(uint32_t r=0;r<rows;++r) {
            uint32_t n=0;for(uint32_t t=0;t<context;++t)n+=eligible(r,t,before);
            const uint32_t budget=budgets[r]<=0 ? 0 : std::min(cap,uint32_t(budgets[r]));
            assert(counts[r]>=0 && uint32_t(counts[r])<=std::min(n,budget));
            if(distinct)assert(uint32_t(counts[r])==std::min(n,budget));
            for(int32_t i=0;i<counts[r];++i) {
                const uint32_t j=r*cap+i;const int32_t token=tokens[j],h=selected_hosts[j];
                assert(token>=0 && uint32_t(token)<context && eligible(r,token,before));
                assert(seen_hosts.insert(h).second);assert(h==ids[r*context+token]);
                assert(map[h]==echo::staging_location(device_capacity,j));
                assert(std::memcmp(staging.data()+j*bytes,source.data()+h*bytes,bytes)==0);
            }
            for(uint32_t i=counts[r];i<cap;++i)assert(tokens[r*cap+i]==-1 && selected_hosts[r*cap+i]==-1);
        }
        for(uint32_t h=0;h<hosts;++h) {
            assert(map[h]!=echo::kClaimed);
            if(before[h]!=-1)assert(map[h]==before[h]);
            else if(seen_hosts.count(h)==0)assert(map[h]==-1);
        }
        // Consumer can resolve a shared host through a different query's row.
        for(int32_t h:seen_hosts) {
            const int32_t location=map[h];assert(echo::is_staging(location,device_capacity,rows*cap));
            const uint32_t flat=location-device_capacity,owner=flat/cap,slot=flat%cap;
            assert(slot<uint32_t(counts[owner]) && selected_hosts[owner*cap+slot]==h);
        }
    }
};
int main() {
    assert(echo::staging_location(1000,69)==1069);
    assert(echo::is_resident(0,1000) && echo::is_resident(999,1000));
    assert(!echo::is_resident(-1,1000) && !echo::is_resident(1000,1000));
    assert(echo::is_staging(1005,1000,132) && !echo::is_staging(1132,1000,132));
    assert(echo::staging_location(INT32_MAX-132,131)==INT32_MAX-1);
    uint32_t runs=0;
    for(uint32_t bytes:{1U,31U,128U,1152U,2049U,4097U}) {
        Case c(bytes);
        // Prior tags remain live into clear(); never zero the map to hide bugs.
        for(uint32_t replay=0;replay<8;++replay) {
            const auto prior=c.map;
            c.clear();
            for(uint32_t h=0;h<Case::hosts;++h) {
                if(echo::is_staging(prior[h],Case::device_capacity,Case::rows*Case::cap))assert(c.map[h]==-1);
                else assert(c.map[h]==prior[h]);
            }
            // Cache manager updates persistent state only between launches.
            std::fill(c.map.begin(),c.map.end(),-1);std::fill(c.written.begin(),c.written.end(),1);
            std::fill(c.stable.begin(),c.stable.end(),Case::context);std::fill(c.thresholds.begin(),c.thresholds.end(),0);
            std::fill(c.budgets.begin(),c.budgets.end(),Case::cap);std::fill(c.visible.begin(),c.visible.end(),Case::context);c.active=4;
            for(uint32_t i=0;i<c.ids.size();++i)c.ids[i]=i+1;
            if(replay==1) {for(uint32_t h=1;h<Case::hosts;h+=2)c.map[h]=0;c.stable[1]=63;c.active=3;}
            if(replay==2) {c.thresholds[0]=std::numeric_limits<float>::infinity();c.thresholds[1]=NAN;
                c.budgets[2]=1;c.budgets[3]=0;}
            if(replay==3) {for(uint32_t h=1;h<Case::hosts;h+=3)c.written[h]=0;
                for(uint32_t t=0;t<Case::context;t+=5)c.ids[t]=-1;c.ids[21]=Case::hosts+5;c.ids[22]=0;}
            if(replay==4) {std::fill(c.map.begin(),c.map.end(),5);c.active=0;}
            if(replay==5) {c.budgets[0]=10000;c.budgets[1]=-1;c.stable[2]=0;c.stable[3]=7;c.visible={513,1,511,63};}
            // Force many duplicates within and across query rows/KV blocks.
            if(replay>=6) for(uint32_t i=0;i<c.ids.size();++i)c.ids[i]=i%(replay==6 ? 7U : 101U)+1;
            const auto before=c.map,ids_before=c.ids;
            c.execute();c.check(before,replay<6);assert(c.ids==ids_before);
            if(replay==6) {
                // Seven shared hosts, enough space for all: exactly one copy/host.
                uint32_t total=0;for(auto n:c.counts)total+=n;assert(total==7);
            }
            ++runs;
        }
        // Cleanup must preserve entries promoted to resident by a consumer.
        auto h=std::find_if(c.selected_hosts.begin(),c.selected_hosts.end(),[](auto h){return h>0;});
        assert(h!=c.selected_hosts.end());const int32_t promoted=*h;c.map[promoted]=4;
        c.clear();assert(c.map[promoted]==4);
        const auto after=c.map;c.clear();assert(c.map==after);
        ++runs;
        // A stale output list must not clear an unrelated tag with a different
        // flattened slot, even if it refers to the same host token.
        c.selected_hosts[0]=promoted;c.tokens[0]=10;
        c.map[promoted]=echo::staging_location(Case::device_capacity,Case::cap+1);
        c.clear();assert(c.map[promoted]==echo::staging_location(Case::device_capacity,Case::cap+1));
        ++runs;
        // Explicit all-zero budgets: no permanent claims, all old tags cleared.
        std::fill(c.map.begin(),c.map.end(),-1);std::fill(c.budgets.begin(),c.budgets.end(),0);
        c.execute();c.check(c.map,false);
        for(auto entry:c.map)assert(entry==-1);
        ++runs;
    }
    std::cout<<runs<<" concurrent tagged-mapping/replay cases passed\n";
}
