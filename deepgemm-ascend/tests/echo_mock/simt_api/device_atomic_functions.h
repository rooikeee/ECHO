#pragma once
inline int32_t asc_atomic_cas(int32_t* p,int32_t expected,int32_t value) {
    __atomic_compare_exchange_n(p,&expected,value,false,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST);return expected;
}
