#include "yolo2_accel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core_scheduler.hpp"
#include "core_io.hpp"
#include "core_compute.hpp"
#include "model_config.hpp"
#include <core/precision.hpp>

#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <cmath>
#include <cstdint>

namespace {
void generate_iofm_offset(IO_Dtype* in_ptr[32], IO_Dtype* out_ptr[32], IO_Dtype *Memory_buf, network *net, const ModelConfig &cfg)
{
    IO_Dtype *Memory_top = Memory_buf+512;
    IO_Dtype *Memory_bottom = Memory_top + cfg.mem_len;
    for(int x=0;x<18;x++)
    {
        int out_w = net->layers[x].out_w;
        int out_w_align_256b = (out_w >> 3) << 3;
        if(out_w & 0x7)
            out_w_align_256b += 8;

        if(x%2==0)
        {
            in_ptr[x] = Memory_top;
            out_ptr[x] = Memory_bottom - net->layers[x].out_c *  net->layers[x].out_h * out_w_align_256b;
        }
        else
        {
            in_ptr[x] = out_ptr[x-1];
            out_ptr[x] = Memory_top;
        }
    }

    for(int x=18;x<25;x++)
    {
        int out_w = net->layers[x].out_w;
        int out_w_align_256b = (out_w >> 3) << 3;
        if(out_w & 0x7)
            out_w_align_256b += 8;

        if(x%2==0)
        {
            in_ptr[x] = Memory_top;
            out_ptr[x] = Memory_bottom - cfg.route16_len - net->layers[x].out_c *  net->layers[x].out_h * out_w_align_256b;
        }else
        {
            in_ptr[x] = out_ptr[x-1];
            out_ptr[x] = Memory_top;
        }
    }

    in_ptr[26] = Memory_bottom - cfg.route16_len;
    out_ptr[26] = Memory_top;

    in_ptr[27] = Memory_top;
    out_ptr[27] = Memory_bottom - cfg.route16_len - cfg.conv24_len - cfg.conv27_len;

    in_ptr[29] = out_ptr[27];
    out_ptr[29] = Memory_top;

    in_ptr[30] = Memory_top;
    out_ptr[30] = Memory_bottom - (net->layers[30].outputs + cfg.detection_workspace);

    in_ptr[31] = out_ptr[30];
}

void reorg_cpu(IO_Dtype *x, int w, int h, int c, int stride, IO_Dtype *out)
{
    int out_c = c/(stride*stride);

    for(int k = 0; k < c; ++k){
        for(int j = 0; j < h; ++j){
            for(int i = 0; i < w; ++i){
                int in_index  = i + w*(j + h*k);
                int c2 = k % out_c;
                int offset = k / out_c;
                int w2 = i*stride + offset % stride;
                int h2 = j*stride + offset / stride;
                int out_index = w2 + w*stride*(h2 + h*stride*c2);
                out[in_index] = x[out_index];
            }
        }
    }
}
} // namespace

template <typename T>
std::vector<T> read_binary(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Failed to open file: " + path);
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (sz < 0 || sz % sizeof(T) != 0) {
        std::fclose(fp);
        throw std::runtime_error("Invalid size for file: " + path);
    }
    std::vector<T> buf(sz / sizeof(T));
    size_t rd = std::fread(buf.data(), sizeof(T), buf.size(), fp);
    std::fclose(fp);
    if (rd != buf.size()) throw std::runtime_error("Short read: " + path);
    return buf;
}

struct WeightsPack {
    std::vector<IO_Dtype> weights;
    std::vector<IO_Dtype> bias;
};

WeightsPack load_weights(const network *net, Precision precision) {
    const ModelConfig &cfg = yolo2_model_config();
    int conv_layers = 0;
    for (int i = 0; i < net->n; ++i) if (net->layers[i].type == CONVOLUTIONAL) conv_layers++;

    size_t expected_w = 0;
    size_t expected_b = 0;
    for (int i = 0; i < conv_layers && i < static_cast<int>(cfg.weight_offsets.size()); ++i) {
        expected_w += cfg.weight_offsets[i];
        expected_b += cfg.beta_offsets[i];
    }

    if (precision == Precision::FP32) {
        auto w = read_binary<float>("weights/weights_reorg.bin");
        auto b = read_binary<float>("weights/bias.bin");
        if (w.size() < expected_w) throw std::runtime_error("weights_reorg.bin too small");
        if (b.size() < expected_b) throw std::runtime_error("bias.bin too small");
        std::vector<IO_Dtype> wbuf(w.begin(), w.end());
        std::vector<IO_Dtype> bbuf(b.begin(), b.begin() + expected_b);
        return {std::move(wbuf), std::move(bbuf)};
    }

    auto wq = read_binary<int16_t>("weights/weights_reorg_int16.bin");
    auto bq = read_binary<int16_t>("weights/bias_int16.bin");
    auto wQ = read_binary<int32_t>("weights/weight_int16_Q.bin");
    auto bQ = read_binary<int32_t>("weights/bias_int16_Q.bin");

    if (wQ.size() < static_cast<size_t>(conv_layers) || bQ.size() < static_cast<size_t>(conv_layers)) {
        throw std::runtime_error("Q tables too small for conv layers");
    }
    if (wq.size() < expected_w || bq.size() < expected_b) {
        throw std::runtime_error("int16 weight/bias files too small");
    }

    WeightsPack pack;
    pack.weights.resize(wq.size());
    pack.bias.resize(expected_b);

    size_t woff = 0;
    size_t boff = 0;
    for (int li = 0; li < conv_layers; ++li) {
        const int wlen = cfg.weight_offsets[li];
        const int blen = cfg.beta_offsets[li];
        const float w_scale = std::ldexp(1.0f, -wQ[li]);
        const float b_scale = std::ldexp(1.0f, -bQ[li]);

        if (woff + wlen > wq.size()) throw std::runtime_error("int16 weight truncated at layer " + std::to_string(li));
        if (boff + blen > bq.size()) throw std::runtime_error("int16 bias truncated at layer " + std::to_string(li));

        for (int i = 0; i < wlen; ++i) {
            pack.weights[woff + i] = static_cast<IO_Dtype>(static_cast<float>(wq[woff + i]) * w_scale);
        }
        for (int i = 0; i < blen; ++i) {
            pack.bias[boff + i] = static_cast<IO_Dtype>(static_cast<float>(bq[boff + i]) * b_scale);
        }

        woff += wlen;
        boff += blen;
    }

    // Zero any remaining padding slots beyond expected_w (if present).
    for (size_t i = expected_w; i < pack.weights.size(); ++i) {
        pack.weights[i] = 0;
    }

    return pack;
}

void yolov2_hls_ps(network *net, IO_Dtype *input, Precision precision)
{
    const ModelConfig &cfg = yolo2_model_config();

    WeightsPack wpack = load_weights(net, precision);
    IO_Dtype *Weight_buf = wpack.weights.data();
    IO_Dtype *Beta_buf   = wpack.bias.data();

//leave some memories for overflow, because the load_module will load extra pixels near boundary for padding
    IO_Dtype *Memory_buf = (IO_Dtype*)calloc(cfg.mem_len+512*2,sizeof(IO_Dtype));
    if (!Memory_buf || !Weight_buf || !Beta_buf) {
        printf("Allocation failed in yolov2_hls_ps\n");
        return;
    }
    IO_Dtype* in_ptr[32];
    IO_Dtype* out_ptr[32];
    IO_Dtype* tmp_ptr_f0 = nullptr;
    generate_iofm_offset( in_ptr, out_ptr, Memory_buf, net, cfg);

    memcpy(in_ptr[0], input, 416*416*3*sizeof(IO_Dtype));//416x416x3 input_pic

    IO_Dtype *region_buf = (IO_Dtype *)calloc(13*16*425,sizeof(IO_Dtype));
    IO_Dtype *region_buf2 = (IO_Dtype *)calloc(13*16*425,sizeof(IO_Dtype));

    int offset_index = 0;
    int woffset = 0;
    int boffset = 0;
    int TR,TC,TM,TN;
    int output_w,output_h;
    int mLoops;

    for(int i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        switch(l.type)
        {
            case CONVOLUTIONAL:
                output_w = (l.w - l.size + 2*l.pad)/l.stride + 1;
                output_h = (l.h - l.size + 2*l.pad)/l.stride + 1;

                TR = std::min(((OnChipIB_Height-l.size)/l.stride+1),Tr);//keep Kstride>=1
                TR = std::min(output_h,TR);
                TC = std::min(((OnChipIB_Width-l.size)/l.stride+1),Tc);
                TC = std::min(output_w,TC);
                TM = std::min(l.n,Tm);
                TN = std::min(l.c,Tn);
                mLoops = (int)ceil(((float)l.n)/TM);

                YOLO2_FPGA(in_ptr[i],out_ptr[i],Weight_buf+woffset,Beta_buf+boffset,
                    l.c,l.n,l.size,
                    l.stride,l.w,l.h,output_w, output_h, l.pad,l.activation==LEAKY?1:0,l.batch_normalize?1:0,
                    TM,TN,TR,TC, (mLoops + 1)*TM, mLoops*TM, (mLoops + 1)*TM, 0);

                woffset += cfg.weight_offsets[offset_index];
                boffset += cfg.beta_offsets[offset_index];
                offset_index++;

                break;
            case MAXPOOL:
                output_w = l.out_h;
                output_h = l.out_w;

                TR = std::min(((OnChipIB_Height-l.size)/l.stride+1),Tr);//keep Kstride>=1
                TC = std::min(((OnChipIB_Width-l.size)/l.stride+1),Tc);
                TR = std::min(output_h,TR);
                TC = std::min(output_w,TC);
                TM = std::min(Tm,Tn);
                TM = std::min(l.c,TM);
                mLoops = (int)ceil(((float)l.c)/TM);

                YOLO2_FPGA(in_ptr[i],out_ptr[i],NULL,NULL,l.c,l.c,
                    l.size,l.stride,l.w,l.h, output_w, output_h, l.pad,0,0,TM,0,TR,TC, (mLoops + 2)*TM, mLoops*TM, (mLoops + 1)*TM, 1);

                break;
            case REORG:
                output_w = 26;
                output_h = 32*13;

                TR = std::min(((OnChipIB_Height-l.stride)/l.stride+1),Tr);//keep Kstride>=1
                TR = std::min(output_h,TR);
                TC = std::min(((OnChipIB_Width-l.stride)/l.stride+1),Tc);
                TC = std::min(output_w,TC);
                TM = std::min(Tm,Tn);
                TM = std::min(4,TM);
                mLoops = (int)ceil(((float)4)/TM);

                tmp_ptr_f0 = in_ptr[i];
                for(int k = 0; k<26*64; k++)
                    memcpy((IO_Dtype *)(region_buf + k*26), (IO_Dtype *)(tmp_ptr_f0 + k*32), 26*sizeof(IO_Dtype));
                reorg_cpu(region_buf, output_w, output_h, 4, 2, region_buf2);
                tmp_ptr_f0 = region_buf;
                memset(region_buf, 0,  13*16*256*sizeof(IO_Dtype));
                for(int k = 0; k<13*256; k++)
                    memcpy((IO_Dtype *)(tmp_ptr_f0 + k*16), (IO_Dtype *)(region_buf2 + k*13), 13*sizeof(IO_Dtype));
                memcpy(out_ptr[i], tmp_ptr_f0, 13*16*256*sizeof(IO_Dtype));

                break;
            case ROUTE:
                break;
            case REGION:
                tmp_ptr_f0 = in_ptr[i];
                for(int k = 0; k<13*425; k++)
                    for(int j = 0; j < 16; j++)
                    {
                        if(j < 13)
                            region_buf[k*13 + j] = tmp_ptr_f0[k*16 + j];
                    }
                forward_region_layer(l, region_buf);
                break;
            default:
                break;
        }
    }

    free(Memory_buf);
    free(region_buf);
    free(region_buf2);
}
