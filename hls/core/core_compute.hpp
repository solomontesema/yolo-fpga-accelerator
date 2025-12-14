#pragma once

#include <cstdint>

#include "params.hpp"
#include "types.hpp"
#include <models/yolov2/yolov2_acc_pragmas.h>

void compute(IO_Dtype input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype output_buffer[Tm][Tr][Tc],
             IO_Dtype weight_buffer[Tm][Tn][K][K], IO_Dtype beta_buffer[MAX_BETA_LENGTH], int n_next[1],
             const int Ksize, const int Kstride, int m,
             const int TM_MIN, const int TR_MIN, const int TC_MIN, bool enable);

void pool_yolo2(IO_Dtype Input[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype Output[Tm][Tr][Tc],
                const int Ksize,const int Kstride,
                const int TM_MIN,const int TR_MIN,const int TC_MIN,bool enable);

void write_back_output_reorg(IO_Dtype output_buffer[Tm][Tr][Tc], IO_Dtype *Output,int r,int c,int m,uint16_t Output_w,uint16_t Output_h,
                             uint8_t TM_MIN,uint8_t TR_MIN,uint8_t TC_MIN,const int OHxOW, bool IsNL, bool write_flag);

void nonlinear_leaky_row(IO_Dtype output_localbuf[Tc], IO_Dtype Input[Tm][Tr][Tc], uint8_t tm, uint8_t tr, uint8_t *tm_n, uint8_t *tr_n, uint8_t TC_MIN,const bool IsNL, bool enable);
void ofm_mmcpy_row(IO_Dtype *Output, IO_Dtype local_buf[Tc], int offset, int OHxOW, int Output_w, int TC_MIN, uint8_t tm, uint8_t tr,bool enable);
void reorg_yolo2(IO_Dtype Input[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype Output[Tm][Tr][Tc],
                 const int Ksize,const int Kstride,
                 const int TM_MIN,const int TR_MIN,const int TC_MIN,bool enable);
