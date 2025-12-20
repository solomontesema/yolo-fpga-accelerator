#pragma once

#include <cstdint>

#include "params.hpp"
#include "types.hpp"
#include <core/yolo.h>
#include <core/precision.hpp>

// Public accelerator entry points. These are host-callable simulation
// shims that mirror the HLS design.
void YOLO2_FPGA(IO_Dtype *Input, IO_Dtype *Output, IO_Dtype *Weight, IO_Dtype *Beta,
                int IFM_num, int OFM_num, int Ksize, int Kstride,
                int Input_w, int Input_h, int Output_w, int Output_h,
                int Padding, bool IsNL, bool IsBN,
                int TM, int TN, int TR, int TC,
                int OFM_num_bound, int mLoopsxTM,
                int mLoops_a1xTM, int LayerType,
                int Qw, int Qa_in, int Qa_out, int Qb);

void yolov2_hls_ps(network *net, const float *input, Precision precision);
