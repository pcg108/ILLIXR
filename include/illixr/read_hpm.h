#ifndef READ_HPM_H
#define READ_HPM_H
#include <stdint.h>
#include <sstream>
#include <string>

#pragma once

#define write_csr(reg, val) ({ \
  asm volatile ("csrw " #reg ", %0" :: "rK"(val)); })

#define read_csr(reg) ({ unsigned long __tmp; \
  asm volatile ("csrr %0, " #reg : "=r"(__tmp)); \
  __tmp; })

static inline uint64_t rdcycle() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles)); // Read cycle counter
    return cycles;
}

inline void read_counters(double* read_counters) {
    // read_counters[0] = read_csr(hpmcounter3); // integer loads
    // read_counters[1] = read_csr(hpmcounter4); // integer store
    // read_counters[2] = read_csr(hpmcounter5); // integer arithmetic
    // read_counters[3] = read_csr(hpmcounter6); // conditional branch
    // read_counters[4] = read_csr(hpmcounter7); // JAL or JALR
    // read_counters[5] = read_csr(hpmcounter8); // int multiplication or division
    // read_counters[6] = read_csr(hpmcounter9); // fp load
    // read_counters[7] = read_csr(hpmcounter10); // fp store
    // read_counters[8] = read_csr(hpmcounter11); // fp add
    // read_counters[9] = read_csr(hpmcounter12); // fp mult
    // read_counters[10] = read_csr(hpmcounter13); // fp fmadd
    // read_counters[11] = read_csr(hpmcounter14); // fp div/sqrt
    // read_counters[12] = read_csr(hpmcounter15); // load interlock
    // read_counters[13] = read_csr(hpmcounter16); // long latency interlock
    // read_counters[14] = read_csr(hpmcounter17); // i cache busy
    // read_counters[15] = read_csr(hpmcounter18); // d cache busy
    // read_counters[16] = read_csr(hpmcounter19); // branch misprediction
    // read_counters[17] = read_csr(hpmcounter20); // pipeline flush
    // read_counters[18] = read_csr(hpmcounter21); // int multiplication interlock
    // read_counters[19] = read_csr(hpmcounter22); // fp interlock
    // read_counters[20] = read_csr(hpmcounter23); // i cache miss
    // read_counters[21] = read_csr(hpmcounter24); // d cache miss
    // read_counters[22] = read_csr(hpmcounter25); // d cache writeback
    // read_counters[23] = read_csr(hpmcounter26); // i TLB miss
    // read_counters[24] = read_csr(hpmcounter27); // d TLB miss
    // read_counters[25] = read_csr(cycle); // cycle count
    // read_counters[26] = read_csr(instret); // instruction count


    read_counters[0] = read_csr(hpmcounter3); // load interlock
    read_counters[1] = read_csr(hpmcounter4); // long latency interlock
    read_counters[3] = read_csr(hpmcounter5); // d cache busy
    read_counters[4] = read_csr(hpmcounter6); // branch misprediction
    read_counters[5] = read_csr(hpmcounter7); // pipeline flush
    read_counters[6] = read_csr(hpmcounter8); // int multiplication interlock
    read_counters[7] = read_csr(hpmcounter9); // fp interlock
    read_counters[9] = read_csr(hpmcounter10); // d cache miss
    read_counters[10] = read_csr(hpmcounter11); // d cache writeback
    read_counters[12] = read_csr(hpmcounter12); // d TLB miss
    read_counters[13] = read_csr(cycle); // cycle count
    read_counters[14] = read_csr(instret); // instruction count
}

inline std::string diff_to_string(double* a, double* b) {
    std::ostringstream oss;
    for (int i = 0; i < 29; ++i) {
        oss << (a[i] - b[i]);
        if (i < 28) {
            oss << ",";
        }
    }
    return oss.str();
}


#endif // READ_HPM_H