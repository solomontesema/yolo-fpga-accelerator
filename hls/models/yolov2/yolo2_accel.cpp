#include "yolo2_accel.hpp"
#include "yolo2_accel_internal.hpp"
#include "core_io.hpp"
#include "core_compute.hpp"
#include "core_scheduler.hpp"

#include <cassert>
#include <cmath>
#include <cstring>

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))

#include <models/yolov2/yolov2_acc_pragmas.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

void YOLO2_FPGA(IO_Dtype *Input, IO_Dtype *Output, IO_Dtype *Weight, IO_Dtype *Beta, int IFM_num, int OFM_num,
                int Ksize, int Kstride,
                int Input_w, int Input_h, int Output_w, int Output_h, int Padding, bool IsNL, bool IsBN,
                int TM, int TN, int TR, int TC,
                int OFM_num_bound, int mLoopsxTM, int mLoops_a1xTM, int LayerType)
{
HLS_PRAGMA(HLS INTERFACE m_axi depth=512 port=Input    offset=slave bundle=DATA_BUS num_read_outstanding=1 num_write_outstanding=1 max_read_burst_length=64 max_write_burst_length=64)
HLS_PRAGMA(HLS INTERFACE m_axi depth=512 port=Output    offset=slave bundle=DATA_BUS num_read_outstanding=1 num_write_outstanding=1 max_read_burst_length=64 max_write_burst_length=64)
HLS_PRAGMA(HLS INTERFACE m_axi depth=512 port=Weight offset=slave bundle=DATA_BUS1 num_read_outstanding=1 max_read_burst_length=128)
HLS_PRAGMA(HLS INTERFACE m_axi depth=512 port=Beta   offset=slave bundle=DATA_BUS1 num_read_outstanding=1 max_read_burst_length=128)

HLS_PRAGMA(HLS INTERFACE s_axilite register port=return bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=IFM_num bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=OFM_num bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Ksize bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Kstride bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Input_w bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Input_h bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Output_w bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Output_h bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Padding bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=IsNL bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=IsBN bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=TM bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=TN bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=TR bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=TC bundle=CTRL_BUS)

HLS_PRAGMA(HLS INTERFACE s_axilite register port=OFM_num_bound bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=mLoopsxTM bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=mLoops_a1xTM bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=LayerType bundle=CTRL_BUS)

HLS_PRAGMA(HLS INTERFACE s_axilite register port=Input bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Output bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Weight bundle=CTRL_BUS)
HLS_PRAGMA(HLS INTERFACE s_axilite register port=Beta bundle=CTRL_BUS)

    assert((OFM_num > 0)&&(OFM_num <= 2048));
    assert((IFM_num > 0)&&(IFM_num <= 2048));
    assert((Kstride > 0)&&(Kstride <= S));
    assert((Ksize > 0)&&(Ksize <= K));
    assert((Input_w > 0)&&(Input_w <= 1024));
    assert((Input_h > 0)&&(Input_h <= 1024));
    assert((Output_w > 0)&&(Output_w <= 1024));
    assert((Output_h > 0)&&(Output_h <= 1024));
    assert((Padding >= 0)&&(Padding <= 4));//maybe
    assert((TM > 0)&&(TM <= Tm));
    assert((TN >= 0)&&(TN <= Tn));
    assert((TR > 0)&&(TR <= Tr));
    assert((TC > 0)&&(TC <= Tc));

    uint16_t IW_align_256b = (Input_w >> 3) << 3;
    if(Input_w & 0x7)
        IW_align_256b += 8;
    uint16_t OW_align_256b = (Output_w >> 3) << 3;
    if(Output_w & 0x7)
        OW_align_256b += 8;

    const int OHxOW = Output_h*OW_align_256b;
    const int TRow = (TR-1)*Kstride+Ksize;
    const int TCol = (TC-1)*Kstride+Ksize;
    const int IHxIW   = Input_h*IW_align_256b;
    const int KxK = Ksize*Ksize;
    const int IFM_numxKxK = IFM_num*KxK;

    static IO_Dtype input_buffer0[Tn][OnChipIB_Height][OnChipIB_Width];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=input_buffer0 complete dim=1)
    static IO_Dtype input_buffer1[Tn][OnChipIB_Height][OnChipIB_Width];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=input_buffer1 complete dim=1)
    static IO_Dtype output_buffer[Tm][Tr][Tc];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=output_buffer complete dim=1)
    static IO_Dtype output_buffer1[Tm][Tr][Tc];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=output_buffer1 complete dim=1)
    static IO_Dtype beta_buffer[MAX_BETA_LENGTH];

/////////////////////////////////param
    int r, c, m;
    int TM_MIN,TR_MIN,TC_MIN;
///////////////////////////////////////

    int m0[1], m1[1];
    int TM_MIN0[1], TM_MIN1[1];
    bool pingpongm;

    if(LayerType==0)
        memcpy(beta_buffer,Beta, OFM_num*sizeof(IO_Dtype));

    for(r = 0; r < Output_h; r += TR)
    {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=1024)
        TR_MIN = MIN(TR,Output_h-r);
        for(c = 0; c < Output_w; c += TC)
        {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=1024)
            TC_MIN = MIN(TC,Output_w-c);
            pingpongm = 0;
            for(m = 0; m < OFM_num_bound; m += TM)
            {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=2048)
                TM_MIN = MIN(TM, OFM_num-m);
                bool Mne0 = (m!=0);
                bool Mne1 = (m!=TM);
                bool MnemLps = (m!=mLoopsxTM);
                bool MneMLps_a1 = (m!=mLoops_a1xTM);
                bool input_flag = LayerType ? MnemLps&&MneMLps_a1: MnemLps;
                bool process_flag = LayerType ? Mne0&&MneMLps_a1 : MnemLps;
                bool write_flag = LayerType ? Mne0&&Mne1 : Mne0;

                if(pingpongm==0)
                {
                    intra_pingpong_wrapper(Input,Weight,output_buffer1,beta_buffer,input_buffer0,input_buffer1,
                                    IFM_num, Input_w, IW_align_256b, Input_h, OFM_num, Ksize, Kstride,
                                    r, c, m, TM_MIN, TR_MIN, TC_MIN, TN, TRow, TCol, Padding,IHxIW,KxK,IFM_numxKxK,LayerType,TM, m1,TM_MIN1, pingpongm, input_flag, process_flag);

                    write_back_output_reorg(output_buffer,Output, r, c, m0[0],OW_align_256b,Output_h, TM_MIN0[0], TR_MIN, TC_MIN, OHxOW, IsNL, write_flag);
                    pingpongm = 1;
                }else
                {
                    intra_pingpong_wrapper(Input,Weight,output_buffer,beta_buffer,input_buffer0,input_buffer1,
                                    IFM_num, Input_w, IW_align_256b, Input_h, OFM_num, Ksize, Kstride,
                                    r, c, m, TM_MIN, TR_MIN, TC_MIN, TN, TRow, TCol, Padding,IHxIW,KxK,IFM_numxKxK,LayerType,TM, m0,TM_MIN0, pingpongm, input_flag, process_flag);

                    write_back_output_reorg(output_buffer1,Output, r, c, m1[0],OW_align_256b,Output_h, TM_MIN1[0], TR_MIN, TC_MIN, OHxOW, IsNL, write_flag);
                    pingpongm = 0;
                }

            }
        }
    }
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
