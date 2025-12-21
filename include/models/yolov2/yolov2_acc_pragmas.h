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

// Keep pragmas active even for host builds so HLS always sees interface/depth
// attributes (warnings are suppressed above).
#define HLS_PRAGMA(x) _Pragma(#x)

// Keep compatibility with existing DO_PRAGMA calls.
#define DO_PRAGMA(x) HLS_PRAGMA(x)

#endif // YOLOV2_ACC_PRAGMAS_H
