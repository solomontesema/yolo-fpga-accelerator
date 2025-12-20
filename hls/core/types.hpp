#ifndef YOLOV2_HLS_TYPES_HPP
#define YOLOV2_HLS_TYPES_HPP

#include <cstdint>

// Common accelerator type aliases.
// In INT16_MODE, IO_Dtype is fixed-point and Acc_Dtype is widened for accumulation.
#ifdef INT16_MODE
using IO_Dtype = int16_t;
using Acc_Dtype = int32_t;
#else
using IO_Dtype = float;
using Acc_Dtype = float;
#endif

#endif // YOLOV2_HLS_TYPES_HPP
