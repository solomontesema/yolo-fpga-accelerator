#include "core_compute.hpp"
#include "core_io.hpp"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cassert>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

void compute(IO_Dtype input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype output_buffer[Tm][Tr][Tc],
             IO_Dtype weight_buffer[Tm][Tn][K][K], IO_Dtype beta_buffer[MAX_BETA_LENGTH], int n_next[1],
             const int Ksize,const int Kstride,int m,
             const int TM_MIN,const int TR_MIN,const int TC_MIN,bool enable)
{
    static IO_Dtype local_beta_buffer[Tm];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=local_beta_buffer complete dim=1)

    if(!enable)
    {
        copy_local_beta(beta_buffer,local_beta_buffer,TM_MIN, m);
        return;
    }

    uint8_t i,j,tr,tc,tm,tn;
    int n = n_next[0];
    IO_Dtype partial_mul[Tm][Tn];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=partial_mul complete dim=1)
HLS_PRAGMA(HLS ARRAY_PARTITION variable=partial_mul complete dim=2)
    IO_Dtype partial_add[Tm];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=partial_add complete dim=1)

    for(i =0;i < Ksize; i++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=K)
        for(j = 0;j < Ksize; j++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=K)
            for(tr = 0;tr < TR_MIN;tr++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tr)
                for(tc = 0;tc < TC_MIN;tc++)
                {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tc)
HLS_PRAGMA(HLS PIPELINE II=3)
                    for(tm = 0;tm < Tm;tm++)
                    {
HLS_PRAGMA(HLS DEPENDENCE variable=output_buffer inter false)

                        if(i==0&&j==0&&n==0)
                            partial_add[tm] = local_beta_buffer[tm];
                        else
                            partial_add[tm] = output_buffer[tm][tr][tc];

                        for(tn = 0;tn <Tn;tn++)
                        {
                            partial_mul[tm][tn] = weight_buffer[tm][tn][i][j]*input_buffer[tn][Kstride*tr+i][Kstride*tc+j];
                        }

                        IO_Dtype partial_sum = 0;
                        for(tn = 0;tn <Tn;tn++)
                        {
                             partial_sum += partial_mul[tm][tn];
                        }
                        output_buffer[tm][tr][tc] = partial_add[tm] + partial_sum;
                    }

                }
}

void nonlinear_leaky_row(IO_Dtype output_localbuf[Tc], IO_Dtype Input[Tm][Tr][Tc], uint8_t tm, uint8_t tr, uint8_t *tm_n, uint8_t *tr_n, uint8_t TC_MIN,const bool IsNL, bool enable)
{
    if(!enable)
        return ;

    uint8_t tc;
    assert((TC_MIN>0)&&(TC_MIN<=Tc));

    IO_Dtype tmp_out;
    for(tc = 0;tc < TC_MIN;tc++)
    {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tc)
HLS_PRAGMA(HLS PIPELINE II=1)
        IO_Dtype tmp = Input[tm][tr][tc];
        if((tmp < 0.0f)&&IsNL)
            tmp_out = tmp*0.1f;
        else
            tmp_out = tmp;
        output_localbuf[tc] = tmp_out;
    }

    *tm_n = tm;
    *tr_n = tr;
}

void ofm_mmcpy_row(IO_Dtype *Output, IO_Dtype local_buf[Tc], int offset, int OHxOW, int Output_w, int TC_MIN, uint8_t tm, uint8_t tr,bool enable)
{
    if(!enable)
        return;

    int ofm_offset = tm*OHxOW + tr*Output_w + offset;
    memcpy((IO_Dtype *)(Output + ofm_offset), local_buf, TC_MIN*sizeof(IO_Dtype));
}

void write_back_output_reorg(IO_Dtype output_buffer[Tm][Tr][Tc], IO_Dtype *Output,int r,int c,int m,uint16_t Output_w,uint16_t Output_h,
                             uint8_t TM_MIN,uint8_t TR_MIN,uint8_t TC_MIN,const int OHxOW, bool IsNL, bool write_flag)
{
    if(!write_flag || !Output)
        return;

    assert((TM_MIN >0)&&(TM_MIN <=Tm));
    assert((TR_MIN >0)&&(TR_MIN <=Tr));
    assert((TC_MIN >0)&&(TC_MIN <=Tc));

    const int offset = m*OHxOW + r*Output_w + c;
    static IO_Dtype local_buf0[Tc];
    static IO_Dtype local_buf1[Tc];
    uint8_t tm_n0, tm_n1, tr_n0, tr_n1;

    bool pp = true;
    uint8_t tr,tm;
    uint16_t TM_MINxTR_MIN = TM_MIN*TR_MIN;
    uint16_t t;
    tr = 0, tm = 0;
    for(t = 0;t < TM_MINxTR_MIN + 1;t++)
    {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tm*Tr)
        if(pp)
        {
            nonlinear_leaky_row( local_buf0, output_buffer, tm, tr, &tm_n0, &tr_n0, TC_MIN, IsNL, t!=TM_MINxTR_MIN);
            ofm_mmcpy_row( Output, local_buf1, offset, OHxOW, Output_w, TC_MIN, tm_n1, tr_n1, t!=0);
            pp = false;
        }else
        {
            nonlinear_leaky_row( local_buf1, output_buffer, tm, tr, &tm_n1, &tr_n1, TC_MIN, IsNL, t!=TM_MINxTR_MIN);
            ofm_mmcpy_row( Output, local_buf0, offset, OHxOW, Output_w, TC_MIN, tm_n0, tr_n0, t!=0);
            pp = true;
        }

        tr++;
        if(tr==TR_MIN)
        {
            tr = 0;
            tm++;
        }
    }
}

void pool_yolo2(IO_Dtype Input[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype Output[Tm][Tr][Tc],
          const int Ksize,const int Kstride,
          const int TM_MIN,const int TR_MIN,const int TC_MIN,bool enable)
{
    if(!enable)
        return;

    uint8_t i,j,tr,tc,of;
    IO_Dtype tmp[Tn];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=tmp complete dim=1)

    for(tr = 0;tr < TR_MIN;tr++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tr)
        for(tc = 0;tc < TC_MIN;tc++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tc)
            for(i =0;i < Ksize; i++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=K)
                for(j = 0;j < Ksize; j++)
                {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=K)
HLS_PRAGMA(HLS PIPELINE II=1)
                    for( of = 0; of < Tn; of++)
                    {
                        if(i==0&&j==0) {
#ifdef INT16_MODE
                            tmp[of] = static_cast<IO_Dtype>(-32768);
#else
                            tmp[of] = -1024*1024;
#endif
                        }

                        if(Input[of][tr*Kstride+i][tc*Kstride+j] > tmp[of])
                            tmp[of] = Input[of][tr*Kstride+i][tc*Kstride+j];

                        if(i==1&&j==1)
                            Output[of][tr][tc] = tmp[of];
                    }
                }

}

void zero_output(IO_Dtype output_buffer[Tm][Tr][Tc], int TM_MIN, int TR_MIN, int TC_MIN)
{
    for (int tm = 0; tm < TM_MIN; ++tm)
        for (int tr = 0; tr < TR_MIN; ++tr)
            for (int tc = 0; tc < TC_MIN; ++tc)
            {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tm*Tr*Tc)
HLS_PRAGMA(HLS PIPELINE II=1)
                output_buffer[tm][tr][tc] = 0;
            }
}

void accumulate_conv(IO_Dtype input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype output_buffer[Tm][Tr][Tc], IO_Dtype weight_buffer[Tm][Tn][K][K],
                     int Ksize, int Kstride, int TM_MIN, int TN_MIN, int TRow, int TCol)
{
    for(int tm = 0; tm < TM_MIN; ++tm)
        for(int tr = 0; tr < TRow; ++tr)
            for(int tc = 0; tc < TCol; ++tc)
            {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tm*Tr*Tc)
HLS_PRAGMA(HLS PIPELINE II=1)
                IO_Dtype acc = output_buffer[tm][tr][tc];
                for(int tn = 0; tn < TN_MIN; ++tn)
                    for(int i = 0; i < Ksize; ++i)
                        for(int j = 0; j < Ksize; ++j)
                        {
                            acc += weight_buffer[tm][tn][i][j] * input_buffer[tn][(tr*Kstride + i)][(tc*Kstride + j)];
                        }
                output_buffer[tm][tr][tc] = acc;
            }
}

void apply_bias_nonlinear(IO_Dtype output_buffer[Tm][Tr][Tc], IO_Dtype *beta_buffer, int m, int TM_MIN, int TRow, int TCol, bool IsNL)
{
    for(int tm = 0; tm < TM_MIN; ++tm)
        for(int tr = 0; tr < TRow; ++tr)
            for(int tc = 0; tc < TCol; ++tc)
            {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tm*Tr*Tc)
HLS_PRAGMA(HLS PIPELINE II=1)
                const int bias_index = m + tm;
                IO_Dtype v = output_buffer[tm][tr][tc] + beta_buffer[bias_index];
                if (IsNL && v < 0) v *= 0.1f;
                output_buffer[tm][tr][tc] = v;
            }
}

void reorg_yolo2(IO_Dtype Input[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype Output[Tm][Tr][Tc],
          const int Ksize,const int Kstride,
          const int TM_MIN,const int TR_MIN,const int TC_MIN,bool enable)
{
    int x, y, kx, ky;
    unsigned char Yoffset;
    unsigned char Xoffset;

    if(!enable)
        return;

    for( y = 0; y < TR_MIN; y++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tr)
        for( x = 0; x < TC_MIN; x++)
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=Tc)
            for(ky= 0;ky < 2; ky++)
            for(kx = 0;kx < 2; kx++)
            {
HLS_PRAGMA(HLS PIPELINE II=1)
                Yoffset = (y << 1) + ky;
                Xoffset = (x << 1) + kx;

                int in_index  = (ky << 1) + kx;
                Output[in_index][y][x] = Input[0][Yoffset][Xoffset];
            }
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
