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
#include <type_traits>

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
    std::vector<int> weight_q; // per-layer weight Q
    std::vector<int> bias_q;   // per-layer bias Q
    std::vector<int> act_q;    // per-layer activation Q (iofm_Q)
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
        if (w.size() < expected_w) throw std::runtime_error("weights file too small");
        if (b.size() < expected_b) throw std::runtime_error("bias file too small");
        std::vector<IO_Dtype> wbuf(w.begin(), w.begin() + expected_w);
        std::vector<IO_Dtype> bbuf(b.begin(), b.begin() + expected_b);
        return {std::move(wbuf), std::move(bbuf), {}, {}, {}};
    } else {
        auto w = read_binary<int16_t>("weights/weights_reorg_int16.bin");
        auto b = read_binary<int16_t>("weights/bias_int16.bin");
        if (w.size() < expected_w) throw std::runtime_error("weights file too small");
        if (b.size() < expected_b) throw std::runtime_error("bias file too small");

        auto wQ = read_binary<int32_t>("weights/weight_int16_Q.bin");
        auto bQ = read_binary<int32_t>("weights/bias_int16_Q.bin");
        if (wQ.size() < static_cast<size_t>(conv_layers) || bQ.size() < static_cast<size_t>(conv_layers)) {
            throw std::runtime_error("Q tables too small for conv layers");
        }

        // Optional activation Q table (iofm)
        std::vector<int> act_q;
        try {
            act_q = read_binary<int32_t>("weights/iofm_Q.bin");
        } catch (...) {
            act_q.clear();
        }

        std::vector<IO_Dtype> wbuf(expected_w);
        std::vector<IO_Dtype> bbuf(expected_b);

        size_t w_file_off = 0;
        size_t w_out_off = 0;
        size_t b_file_off = 0;
        size_t b_out_off = 0;
        for (int li = 0; li < conv_layers; ++li) {
            const int wlen = cfg.weight_offsets[li];
            const int blen = cfg.beta_offsets[li];

            if (w_file_off + wlen > w.size()) throw std::runtime_error("int16 weight truncated at layer " + std::to_string(li));
            if (b_out_off + blen > bbuf.size()) throw std::runtime_error("int16 bias output buffer overflow at layer " + std::to_string(li));
            if (b_file_off + blen > b.size()) throw std::runtime_error("int16 bias truncated at layer " + std::to_string(li));

            std::copy_n(w.data() + w_file_off, wlen, wbuf.data() + w_out_off);
            std::copy_n(b.data() + b_file_off, blen, bbuf.data() + b_out_off);

            // Handle per-layer padding inserted during quantization for odd counts.
            const int wpad = (wlen & 0x1) ? 1 : 0;
            const int bpad = (blen & 0x1) ? 1 : 0;

            w_file_off += wlen + wpad;
            w_out_off  += wlen;
            b_file_off += blen + bpad;
            b_out_off  += blen;
        }
        return {std::move(wbuf), std::move(bbuf), std::move(wQ), std::move(bQ), std::move(act_q)};
    }
}

void yolov2_hls_ps(network *net, const float *input, Precision precision)
{
    const ModelConfig &cfg = yolo2_model_config();

#ifdef INT16_MODE
    if (precision == Precision::FP32) {
        throw std::runtime_error("FP32 precision requested while INT16_MODE is enabled. Rebuild without INT16_MODE for FP32.");
    }
#endif

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

    const int input_elems = 416*416*3;
    std::vector<IO_Dtype> input_q;
    const IO_Dtype *input_data = nullptr;
    if (precision == Precision::INT16) {
        if (wpack.act_q.empty()) {
            throw std::runtime_error("Activation Q table (iofm_Q.bin) is required for int16 inference.");
        }
        const int q_in = wpack.act_q.front();
        const float scale = std::ldexp(1.0f, q_in);
        input_q.resize(input_elems);
        for (int idx = 0; idx < input_elems; ++idx) {
            float v = input[idx] * scale;
            if (v > 32767.f) v = 32767.f;
            if (v < -32768.f) v = -32768.f;
            int64_t q = static_cast<int64_t>(std::llround(v));
            if (q > 32767) q = 32767;
            if (q < -32768) q = -32768;
            input_q[idx] = static_cast<IO_Dtype>(q);
        }
        input_data = input_q.data();
    } else {
        input_data = reinterpret_cast<const IO_Dtype *>(input);
    }

    memcpy(in_ptr[0], input_data, input_elems*sizeof(IO_Dtype));//416x416x3 input_pic

    const int region_len = 13*16*425;
    std::vector<IO_Dtype> region_buf(region_len, 0);
    std::vector<IO_Dtype> region_buf2(region_len, 0);

    int offset_index = 0;
    int woffset = 0;
    int boffset = 0;
    int TR,TC,TM,TN;
    int output_w,output_h;
    int mLoops;
    int current_Qa = (!wpack.act_q.empty()) ? wpack.act_q.front() : 0;
    int route24_q = 0;
    int pending_route_q = -1;

    for(int i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        switch(l.type)
        {
            case CONVOLUTIONAL: {
                output_w = (l.w - l.size + 2*l.pad)/l.stride + 1;
                output_h = (l.h - l.size + 2*l.pad)/l.stride + 1;

                TR = std::min(((OnChipIB_Height-l.size)/l.stride+1),Tr);//keep Kstride>=1
                TR = std::min(output_h,TR);
                TC = std::min(((OnChipIB_Width-l.size)/l.stride+1),Tc);
                TC = std::min(output_w,TC);
                TM = std::min(l.n,Tm);
                TN = std::min(l.c,Tn);
                mLoops = (int)ceil(((float)l.n)/TM);

                int Qw = 0, Qb = 0, Qa_in = 0, Qa_out = 0;
                if (precision == Precision::INT16) {
                    const size_t act_entries = wpack.act_q.size();
                    Qa_in = (offset_index < static_cast<int>(act_entries)) ? wpack.act_q[offset_index] : current_Qa;
                    Qa_out = (offset_index + 1 < static_cast<int>(act_entries)) ? wpack.act_q[offset_index + 1] : Qa_in;
                    Qw = (offset_index < static_cast<int>(wpack.weight_q.size())) ? wpack.weight_q[offset_index] : 0;
                    Qb = (offset_index < static_cast<int>(wpack.bias_q.size())) ? wpack.bias_q[offset_index] : 0;
                    if (pending_route_q >= 0) {
                        Qa_in = pending_route_q;
                    }
                }
                YOLO2_FPGA(in_ptr[i],out_ptr[i],Weight_buf+woffset,Beta_buf+boffset,
                    l.c,l.n,l.size,
                    l.stride,l.w,l.h,output_w, output_h, l.pad,l.activation==LEAKY?1:0,l.batch_normalize?1:0,
                    TM,TN,TR,TC, (mLoops + 1)*TM, mLoops*TM, (mLoops + 1)*TM, 0,
                    Qw, Qa_in, Qa_out, Qb);

                woffset += cfg.weight_offsets[offset_index];
                boffset += cfg.beta_offsets[offset_index];
                if (precision == Precision::INT16) {
                    current_Qa = Qa_out;
                    if (i == 24) {
                        route24_q = current_Qa; // save skip connection scale before reorg/route
                    }
                    pending_route_q = -1;
                }
                offset_index++;

                break;
            }
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
                    l.size,l.stride,l.w,l.h, output_w, output_h, l.pad,0,0,TM,0,TR,TC, (mLoops + 2)*TM, mLoops*TM, (mLoops + 1)*TM, 1,
                    0,0,0,0);

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
                    memcpy((IO_Dtype *)(region_buf.data() + k*26), (IO_Dtype *)(tmp_ptr_f0 + k*32), 26*sizeof(IO_Dtype));
                reorg_cpu(region_buf.data(), output_w, output_h, 4, 2, region_buf2.data());
                tmp_ptr_f0 = region_buf.data();
                memset(region_buf.data(), 0,  13*16*256*sizeof(IO_Dtype));
                for(int k = 0; k<13*256; k++)
                    memcpy((IO_Dtype *)(tmp_ptr_f0 + k*16), (IO_Dtype *)(region_buf2.data() + k*13), 13*sizeof(IO_Dtype));

                if (precision == Precision::INT16 && route24_q > 0) {
                    // Align the reorg branch scale with the skip connection branch before concatenation.
                    const int target_q = std::min(route24_q, current_Qa);
                    const int shift = current_Qa - target_q;
                    if (shift != 0) {
                        const int total = 13 * 16 * 256;
                        for (int idx = 0; idx < total; ++idx) {
                            int32_t v = static_cast<int32_t>(tmp_ptr_f0[idx]);
                            if (shift > 0) {
                                v >>= shift;
                            } else {
                                v <<= -shift;
                            }
                            if (v > 32767) v = 32767;
                            if (v < -32768) v = -32768;
                            tmp_ptr_f0[idx] = static_cast<IO_Dtype>(v);
                        }
                        current_Qa = target_q;
                    }
                    pending_route_q = current_Qa;
                }

                memcpy(out_ptr[i], tmp_ptr_f0, 13*16*256*sizeof(IO_Dtype));

                break;
            case ROUTE:
                break;
            case REGION: {
                tmp_ptr_f0 = in_ptr[i];
                for(int k = 0; k<13*425; k++)
                    for(int j = 0; j < 16; j++)
                    {
                        if(j < 13)
                            region_buf[k*13 + j] = tmp_ptr_f0[k*16 + j];
                    }
                std::vector<float> region_f(region_buf.size());
                if (precision == Precision::INT16 && !wpack.act_q.empty()) {
                    const int q_out = current_Qa;
                    const float scale = std::ldexp(1.0f, -q_out);
                    for (size_t t = 0; t < region_buf.size(); ++t) {
                        region_f[t] = static_cast<float>(region_buf[t]) * scale;
                    }
                } else {
                    for (size_t t = 0; t < region_buf.size(); ++t) {
                        region_f[t] = static_cast<float>(region_buf[t]);
                    }
                }
                forward_region_layer(l, region_f.data());
                break;
            }
            default:
                break;
        }
    }

    free(Memory_buf);
}
