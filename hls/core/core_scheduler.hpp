#pragma once

#include "params.hpp"
#include "types.hpp"
#include "core_io.hpp"
#include "core_compute.hpp"
#include <models/yolov2/yolov2_acc_pragmas.h>

void intra_pingpong_wrapper(IO_Dtype *Input, IO_Dtype *Weight, IO_Dtype output_buffer[Tm][Tr][Tc], IO_Dtype beta_buffer[MAX_BETA_LENGTH],
                            IO_Dtype input_buffer0[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype input_buffer1[Tn][OnChipIB_Height][OnChipIB_Width],
                            int IFM_num,int Input_w,int IW_align_256b,int Input_h,int OFM_num,int Ksize,int Kstride,
                            int TMP_R,int TMP_C,int TMP_M,int TM_MIN,int TR_MIN,int TC_MIN,int TN,int TRow,int TCol,int Padding,
                            int IHxIW,int KxK,int IFM_numxKxK,int LayerType,int TM,int TMP_X_next[1],int TX_MIN_next[1],bool pingpongx,bool input_flag,bool process_flag);
