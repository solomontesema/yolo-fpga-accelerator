#include "core_io.hpp"

#include <cassert>
#include <algorithm>
#include <cstring>

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

void ifm_mmcpy_row(IO_Dtype *input, IO_Dtype local_buf[OnChipIB_Width/8+3][8], int CurrentOffset, int IHxIW, int IW_align_256b, int TCol,
                   uint8_t t1, uint8_t t2, uint8_t *t1_n, uint8_t *t2_n, uint8_t *bn_n, bool enable)
{
    if(!enable)
        return;

    const int ifm_offset = CurrentOffset + t1*IHxIW + t2*IW_align_256b;
    const int ifm_trans_offset = (ifm_offset >> 3) << 3;
    const uint8_t begin_num = ifm_offset & 0x7;

    uint16_t TCol_a = TCol + begin_num;
    uint16_t loop_cnts = TCol_a >> 3;
    if(TCol_a & 0x7)
        loop_cnts++;

    for(uint16_t t = 0; t < loop_cnts; t++)
    {
        memcpy(local_buf[t], input + ifm_trans_offset + t*8, 8*sizeof(IO_Dtype));
    }

    *t1_n = t1;
    *t2_n = t2;
    *bn_n = begin_num;
}

void ifm_copy_lbuf2ibuf(IO_Dtype input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype local_buf[OnChipIB_Width/8+3][8], int TCol, int Input_w, int Input_h, int TN_MIN, IO_Dtype pad_value,
                        int Coffset, int Roffset, uint8_t t1, uint8_t t2, uint8_t bn, bool enable)
{
    if(!enable)
        return;

    bool TN_Enable = t1 < TN_MIN;
    int yoffset = Roffset + t2;
    bool YEnable = (yoffset >= 0)&&(yoffset < Input_h);
    bool PEnable = YEnable&&TN_Enable;

    uint16_t cnt = 1;
    uint8_t bn_local = bn;
    IO_Dtype buf_256b[8];
    memcpy(buf_256b, local_buf[0], 8*sizeof(IO_Dtype));
    for(uint8_t t3 = 0; t3 < TCol; t3++)
    {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=TCol_max)
HLS_PRAGMA(HLS PIPELINE II=1)
        int xoffset = Coffset + t3;
        bool XEnable = (xoffset >= 0)&&(xoffset < Input_w);
        if(XEnable&&PEnable)
        {
            input_buffer[t1][t2][t3] = buf_256b[bn_local];
        }
        else
            input_buffer[t1][t2][t3] = pad_value;

        bn_local++;
        if(bn_local==8)
        {
            bn_local = 0;
            memcpy(buf_256b, local_buf[cnt], 8*sizeof(IO_Dtype));
            cnt++;
        }
    }
}

void input_load(IO_Dtype *input, IO_Dtype input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], int r, int c, int n, int Kstride, int Padding, int TRow, int TCol, int Input_w, int IW_align_256b, int Input_h, int TN_MIN, int IHxIW, int LayerType)
{
    uint8_t t1,t2;
    static IO_Dtype local_buf0[OnChipIB_Width/8+3][8];
    static IO_Dtype local_buf1[OnChipIB_Width/8+3][8];

    const int Coffset = c*Kstride - Padding;
    const int Roffset = r*Kstride - Padding;
    const int CurrentOffset = n*IHxIW + Roffset*IW_align_256b + Coffset;

    uint8_t t1_n0 = 0, t1_n1 = 0, t2_n0 = 0, t2_n1 = 0;
    uint8_t bn_n0 = 0, bn_n1 = 0;
    bool pp = true;
    
    IO_Dtype pad_value = 0.0f;
    if(LayerType==1)
        pad_value = -1024*1024;
    
    int TnxTRow = Tn*TRow;
    int t = 0;
    t1 = 0; t2 = 0;
    for(t = 0;t < TnxTRow+1; t++)
    {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=TRow_max)
        if(pp)
        {
            ifm_mmcpy_row(input, local_buf0, CurrentOffset, IHxIW, IW_align_256b, TCol, t1, t2, &t1_n0, &t2_n0, &bn_n0, t!=TnxTRow);
            ifm_copy_lbuf2ibuf( input_buffer, local_buf1, TCol, Input_w, Input_h, TN_MIN, pad_value, Coffset, Roffset, t1_n1, t2_n1, bn_n1, t!=0);
            pp = false;
        }else
        {
            ifm_mmcpy_row(input, local_buf1, CurrentOffset, IHxIW, IW_align_256b, TCol, t1, t2, &t1_n1, &t2_n1, &bn_n1, t!=TnxTRow);
            ifm_copy_lbuf2ibuf( input_buffer, local_buf0, TCol, Input_w, Input_h, TN_MIN, pad_value, Coffset, Roffset, t1_n0, t2_n0, bn_n0, t!=0);
            pp = true;
        }
        
        t2++;
        if(t2==TRow)
        {
            t2 = 0;
            t1++;
        }
    }
}

void weight_load_reorg(IO_Dtype *Weight, IO_Dtype weight_buffer[Tm][Tn][K][K], bool weight_load_enable, int m, int n, int IFM_numxKxK, int KxK, int Ksize, int TM_MIN, int TN_MIN)
{
    (void)IFM_numxKxK;
    uint8_t t1,t2,t3,t4;
    static IO_Dtype local_buf[(Tm*Tn*K*K)/8 + 3][8];
    static int Woffset;

    assert((TM_MIN > 0)&&(TM_MIN <= Tm));
    assert((TN_MIN > 0)&&(TN_MIN <= Tn));
    assert((KxK > 0)&&(KxK <= K*K));

    if(!weight_load_enable)
        return;

    if(m==0&&n==0)
        Woffset = 0;

    uint16_t mm_offset = TM_MIN*TN_MIN*KxK;

    uint32_t trans_offset = (Woffset >> 3) << 3;
    uint8_t begin_num = Woffset & 0x7;
    uint16_t TCol_a = mm_offset + begin_num;
    uint16_t loop_cnts = TCol_a >> 3;
    if(TCol_a & 0x7)
        loop_cnts++;
    for(uint16_t t = 0; t < loop_cnts; t++)
    {
        memcpy(local_buf[t], Weight + trans_offset + t*8, 8*sizeof(IO_Dtype));
    }
    Woffset += mm_offset;

    uint16_t cnt = 1;
    uint8_t bn_local = begin_num;
    IO_Dtype buf_256b[8];
    memcpy(buf_256b, local_buf[0], 8*sizeof(IO_Dtype));

    for(t3 = 0;t3 <Ksize; t3++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=K)
        for(t4 = 0;t4 <Ksize; t4++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=K)
            for(t1 = 0;t1 < Tm; t1++)
                for(t2 = 0;t2 < Tn; t2++)
                {
HLS_PRAGMA(HLS PIPELINE II=1)
                    bool Enable = (t1 < TM_MIN)&&(t2 < TN_MIN);
                    if(Enable)
                    {
                        weight_buffer[t1][t2][t3][t4] =  buf_256b[bn_local];
                        bn_local++;
                        if(bn_local==8)
                        {
                            bn_local = 0;
                            memcpy(buf_256b, local_buf[cnt], 8*sizeof(IO_Dtype));
                            cnt++;
                        }
                    }
                    else
                        weight_buffer[t1][t2][t3][t4] = 0;
                }
}

void copy_input_weight(IO_Dtype *input, IO_Dtype *Weight, int IFM_num, int Input_w, int IW_align_256b, int Input_h, int Ksize, int Kstride, int r, int c, int m, int n,
                       int TM_MIN, int TN, int TRow, int TCol, int Padding, IO_Dtype input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype weight_buffer[Tm][Tn][K][K], int n_next[1],
                       bool enable, bool weight_load_enable, bool initialize, const int IHxIW, const int KxK, const int IFM_numxKxK, const int LayerType)
{
    (void)initialize; // Not used in current implementation but kept for signature compatibility
    if(!enable)
        return;

    const int TN_MIN = MIN(TN, IFM_num - n);
    n_next[0] = n;

    input_load(input, input_buffer, r, c, n, Kstride, Padding, TRow, TCol, Input_w, IW_align_256b, Input_h, TN_MIN, IHxIW, LayerType);
#ifdef REORG_TEST
    weight_load_reorg(Weight, weight_buffer, weight_load_enable, m, n, IFM_numxKxK, KxK, Ksize, TM_MIN, TN_MIN);
#else
    // Note: weight_load function not implemented in modular code, using reorg version
    weight_load_reorg(Weight, weight_buffer, weight_load_enable, m, n, IFM_numxKxK, KxK, Ksize, TM_MIN, TN_MIN);
#endif
}

void copy_local_beta(IO_Dtype beta_buffer[MAX_BETA_LENGTH], IO_Dtype local_beta_buffer[MAX_BETA_LENGTH], const int TM_MIN, int m)
{
    int offset;
    int tm;
    for(tm = 0,offset = m;tm < TM_MIN;tm++)
    {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tm)
HLS_PRAGMA(HLS PIPELINE II=1)
        local_beta_buffer[tm] = beta_buffer[offset];
        offset++;
    }
}

void beta_copy(IO_Dtype beta_buffer[MAX_BETA_LENGTH], IO_Dtype *Beta, int OFM_num)
{
    memcpy(beta_buffer, Beta, OFM_num * sizeof(IO_Dtype));
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
