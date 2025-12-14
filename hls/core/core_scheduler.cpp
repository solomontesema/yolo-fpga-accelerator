#include "core_scheduler.hpp"

#include <cstring>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

void intra_pingpong_wrapper(IO_Dtype *Input, IO_Dtype *Weight, IO_Dtype output_buffer[Tm][Tr][Tc], IO_Dtype beta_buffer[MAX_BETA_LENGTH],
                            IO_Dtype input_buffer0[Tn][OnChipIB_Height][OnChipIB_Width], IO_Dtype input_buffer1[Tn][OnChipIB_Height][OnChipIB_Width],
                            int IFM_num,int Input_w,int IW_align_256b,int Input_h,int OFM_num,int Ksize,int Kstride,
                            int TMP_R,int TMP_C,int TMP_M,int TM_MIN,int TR_MIN,int TC_MIN,int TN,int TRow,int TCol,int Padding,
                            int IHxIW,int KxK,int IFM_numxKxK,int LayerType,int TM,int TMP_X_next[1],int TX_MIN_next[1],bool pingpongx,bool input_flag,bool process_flag)
{
    static IO_Dtype weight_buffer0[Tm][Tn][K][K];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=weight_buffer0 complete dim=1)
HLS_PRAGMA(HLS ARRAY_PARTITION variable=weight_buffer0 complete dim=2)

    static IO_Dtype weight_buffer1[Tm][Tn][K][K];
HLS_PRAGMA(HLS ARRAY_PARTITION variable=weight_buffer1 complete dim=1)
HLS_PRAGMA(HLS ARRAY_PARTITION variable=weight_buffer1 complete dim=2)

    static int NOP[1];
    static int tmp_x;
    static int tmp_tx_min;

    if(LayerType==0)
    {
        if(!input_flag)
            return;

        TMP_X_next[0] = TMP_M;//consider by the inner-out loop
        TX_MIN_next[0] = TM_MIN;// like above

        bool pingpong = 0;
        int n0[1];
        int n1[1];
        int n;
        for(n = 0;n < IFM_num+TN; n += TN)
        {
DO_PRAGMA(HLS LOOP_TRIPCOUNT min=1 max=2048)
            if(pingpong == 1)
            {
                copy_input_weight(Input,Weight,IFM_num,Input_w,IW_align_256b,Input_h,Ksize,Kstride,TMP_R,TMP_C,TMP_M, n,
                    TM_MIN,TN,TRow,TCol,Padding,input_buffer1,weight_buffer1, n1, n < IFM_num,1,(TMP_M==0)&&(n==0),IHxIW,KxK,IFM_numxKxK,LayerType);
                compute(input_buffer0,output_buffer,weight_buffer0,beta_buffer, n0,Ksize,Kstride,TMP_M,TM_MIN,TR_MIN,TC_MIN, n!=0);
                pingpong = 0;
            }else
            {
                copy_input_weight(Input,Weight,IFM_num,Input_w,IW_align_256b,Input_h,Ksize,Kstride,TMP_R,TMP_C,TMP_M, n,
                    TM_MIN,TN,TRow,TCol,Padding,input_buffer0,weight_buffer0, n0, n < IFM_num,1,(TMP_M==0)&&(n==0),IHxIW,KxK,IFM_numxKxK,LayerType);
                compute(input_buffer1,output_buffer,weight_buffer1,beta_buffer, n1,Ksize,Kstride,TMP_M,TM_MIN,TR_MIN,TC_MIN, n!=0);
                pingpong = 1;
            }
        }
    }
    else if(LayerType==1)
    {
        if(pingpongx==0)
        {
            TMP_X_next[0] = tmp_x;
            TX_MIN_next[0] = tmp_tx_min;
            tmp_x = TMP_M;
            tmp_tx_min = TM_MIN;

            copy_input_weight(Input,Weight,IFM_num,Input_w,IW_align_256b,Input_h,Ksize,Kstride,TMP_R,TMP_C,TMP_M,TMP_M,
                TM_MIN,TM,TRow,TCol,0,input_buffer0,weight_buffer0,NOP,input_flag,0,0,IHxIW,KxK,IFM_numxKxK,LayerType);
            pool_yolo2(input_buffer1,output_buffer,Ksize,Kstride,TX_MIN_next[0],TR_MIN,TC_MIN,process_flag);
        }else
        {
            TMP_X_next[0] = tmp_x;
            TX_MIN_next[0] = tmp_tx_min;
            tmp_x = TMP_M;
            tmp_tx_min = TM_MIN;

            copy_input_weight(Input,Weight,IFM_num,Input_w,IW_align_256b,Input_h,Ksize,Kstride,TMP_R,TMP_C,TMP_M,TMP_M,
                TM_MIN,TM,TRow,TCol,0,input_buffer1,weight_buffer1,NOP,input_flag,0,0,IHxIW,KxK,IFM_numxKxK,LayerType);
            pool_yolo2(input_buffer0,output_buffer,Ksize,Kstride,TX_MIN_next[0],TR_MIN,TC_MIN,process_flag);
        }

    }
    else if(LayerType==2)
    {
        if(pingpongx==0)
        {
            TMP_X_next[0] = tmp_x;
            TX_MIN_next[0] = tmp_tx_min;
            tmp_x = TMP_M;
            tmp_tx_min = TM_MIN;

            copy_input_weight(Input,Weight,IFM_num,Input_w,IW_align_256b,Input_h,Ksize,Kstride,TMP_R,TMP_C,TMP_M,TMP_M,
                TM_MIN,TM,TRow,TCol,0,input_buffer0,weight_buffer0,NOP,input_flag,0,0,IHxIW,KxK,IFM_numxKxK,LayerType);
            reorg_yolo2(input_buffer1,output_buffer,Ksize,Kstride,TX_MIN_next[0],TR_MIN,TC_MIN,process_flag);
        }else
        {
            TMP_X_next[0] = tmp_x;
            TX_MIN_next[0] = tmp_tx_min;
            tmp_x = TMP_M;
            tmp_tx_min = TM_MIN;

            copy_input_weight(Input,Weight,IFM_num,Input_w,IW_align_256b,Input_h,Ksize,Kstride,TMP_R,TMP_C,TMP_M,TMP_M,
                TM_MIN,TM,TRow,TCol,0,input_buffer1,weight_buffer1,NOP,input_flag,0,0,IHxIW,KxK,IFM_numxKxK,LayerType);
            reorg_yolo2(input_buffer0,output_buffer,Ksize,Kstride,TX_MIN_next[0],TR_MIN,TC_MIN,process_flag);
        }

    }
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
