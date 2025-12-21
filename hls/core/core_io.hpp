#pragma once

#include "params.hpp"
#include "types.hpp"
#include <models/yolov2/yolov2_acc_pragmas.h>

// Input feature map load helpers
void ifm_mmcpy_row(IO_Dtype *input, IO_Dtype local_buf[OnChipIB_Width/8+3][8], int CurrentOffset, int IHxIW, int IW_align_256b, int TCol,
                   uint8_t t1, uint8_t t2, uint8_t *t1_n, uint8_t *t2_n, uint8_t *bn_n, bool enable);

void ifm_copy_lbuf2ibuf(IO_Dtype input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype local_buf[OnChipIB_Width/8+3][8], int TCol, int Input_w, int Input_h, int TN_MIN, IO_Dtype pad_value,
                        int Coffset, int Roffset, uint8_t t1, uint8_t t2, uint8_t bn, bool enable);

void input_load(IO_Dtype *input, IO_Dtype input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], int r, int c, int n, int Kstride, int Padding, int TRow, int TCol, int Input_w, int IW_align_256b, int Input_h, int TN_MIN, int IHxIW, int LayerType);

// Weight/Beta load helpers
void weight_load_reorg(IO_Dtype *Weight, IO_Dtype weight_buffer[Tm][Tn][K][K], bool weight_load_enable, int m, int n, int IFM_numxKxK, int KxK, int Ksize, int TM_MIN, int TN_MIN);

void copy_input_weight(IO_Dtype *input, IO_Dtype *Weight, int IFM_num, int Input_w, int IW_align_256b, int Input_h, int Ksize, int Kstride, int r, int c, int m, int n,
                       int TM_MIN, int TN, int TRow, int TCol, int Padding, IO_Dtype input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype weight_buffer[Tm][Tn][K][K], int n_next[1],
                       bool enable, bool weight_load_enable, bool initialize, const int IHxIW, const int KxK, const int IFM_numxKxK, const int LayerType);

void copy_local_beta(IO_Dtype beta_buffer[MAX_BETA_LENGTH], IO_Dtype local_beta_buffer[MAX_BETA_LENGTH], const int TM_MIN, int m);

void beta_copy(IO_Dtype beta_buffer[MAX_BETA_LENGTH], IO_Dtype *Beta, int OFM_num);
