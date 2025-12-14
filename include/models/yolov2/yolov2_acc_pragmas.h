#ifndef YOLOV2_ACC_PRAGMAS_H
#define YOLOV2_ACC_PRAGMAS_H

// Centralized pragma helpers to keep host builds quiet while allowing HLS synthesis
// pragmas to pass through unchanged during hardware builds.

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#if defined(__VIVADO_HLS__) || defined(__SYNTHESIS__) || defined(__HLS__) || defined(XILINX_FPGA)
#define HLS_PRAGMA(x) _Pragma(#x)
#else
#define HLS_PRAGMA(x)
#endif

// Keep compatibility with existing DO_PRAGMA calls.
#define DO_PRAGMA(x) HLS_PRAGMA(x)

#endif // YOLOV2_ACC_PRAGMAS_H
