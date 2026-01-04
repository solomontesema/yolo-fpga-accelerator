/**
 * YOLOv2 Inference Engine - Linux Implementation
 * 
 * Orchestrates layer-by-layer inference through the FPGA accelerator.
 * Ported from FreeRTOS - replaces vPortMalloc/vPortFree with malloc/free.
 * 
 * VERSION: 2.1 - Fixed region layer buffer overflow (13*425 -> 13*13*425)
 */

#define INFERENCE_VERSION "2.1"

#include "yolo2_inference.h"
#include "yolo2_accel_linux.h"
#include "yolo2_config.h"
#include "yolo2_network.h"
#include "dma_buffer_manager.h"
#include "yolo2_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

// Weight offsets (from model_config.cpp)
// Note: These are in elements (words), not bytes
// For INT16 mode, multiply by sizeof(int16_t) to get bytes
static const size_t weight_offsets[] = {
    864, 18432, 73728, 8192, 73728,
    294912, 32768, 294912, 1179648, 131072, 1179648, 131072,
    1179648, 4718592, 524288, 4718592, 524288, 4718592, 9437184,
    9437184, 32768, 11796480, 435200, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

static const size_t beta_offsets[] = {
    32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024,
    512, 1024, 512, 1024, 1024, 1024, 64, 1024, 425, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

#define NUM_WEIGHT_OFFSETS (sizeof(weight_offsets) / sizeof(weight_offsets[0]))
#define NUM_BETA_OFFSETS (sizeof(beta_offsets) / sizeof(beta_offsets[0]))

static uint32_t yolo2_get_layer_timeout_ms(void)
{
    const char *env = getenv("YOLO2_LAYER_TIMEOUT_MS");
    if (!env || env[0] == '\0') {
        return YOLO2_LAYER_TIMEOUT_MS;
    }

    char *end = NULL;
    unsigned long value = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || value == 0) {
        fprintf(stderr,
                "WARNING: Invalid YOLO2_LAYER_TIMEOUT_MS='%s', using default %u\n",
                env, (unsigned)YOLO2_LAYER_TIMEOUT_MS);
        return YOLO2_LAYER_TIMEOUT_MS;
    }

    if (value > UINT32_MAX) {
        return UINT32_MAX;
    }

    return (uint32_t)value;
}

static void yolo2_apply_q_shift_int16(int16_t *buf, size_t count, int shift)
{
    if (!buf || count == 0 || shift == 0) {
        return;
    }

    for (size_t idx = 0; idx < count; ++idx) {
        int32_t v = (int32_t)buf[idx];
        if (shift > 0) {
            v >>= shift;
        } else {
            v <<= -shift;
        }
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        buf[idx] = (int16_t)v;
    }
}

/**
 * Initialize inference context
 */
int yolo2_inference_init(yolo2_inference_context_t *ctx)
{
    memset(ctx, 0, sizeof(yolo2_inference_context_t));
    
    ctx->current_layer = 0;
    ctx->offset_index = 0;
    ctx->woffset = 0;
    ctx->boffset = 0;
    ctx->current_Qa = 0;
    ctx->route24_q = 0;
    ctx->pending_route_q = -1;
    ctx->region_output = NULL;
    ctx->region_output_size = 0;
    ctx->region_layer_idx = -1;
    
    return 0;
}

/**
 * Cleanup inference context
 */
void yolo2_inference_cleanup(yolo2_inference_context_t *ctx)
{
    if (ctx->weights_buf.ptr) {
        memory_free_ddr(&ctx->weights_buf);
    }
    if (ctx->bias_buf.ptr) {
        memory_free_ddr(&ctx->bias_buf);
    }
    if (ctx->inference_buf.ptr) {
        memory_free_ddr(&ctx->inference_buf);
    }
    if (ctx->weight_q) {
        free(ctx->weight_q);
    }
    if (ctx->bias_q) {
        free(ctx->bias_q);
    }
    if (ctx->act_q) {
        free(ctx->act_q);
    }
    if (ctx->region_output) {
        free(ctx->region_output);
    }
    
    memset(ctx, 0, sizeof(yolo2_inference_context_t));
}

/**
 * Process input image (quantize for INT16 mode)
 */
int yolo2_process_input_image(float *input_image, int16_t *output_buffer, 
                              int32_t q_in)
{
    // Calculate scale: 2^Q
    double scale;
    if (q_in >= 0 && q_in <= 31) {
        scale = (double)(1ULL << (unsigned int)q_in);
    } else if (q_in < 0 && q_in >= -31) {
        scale = 1.0 / (double)(1ULL << (unsigned int)(-q_in));
    } else {
        scale = 1.0;
    }
    
    for (int idx = 0; idx < INPUT_ELEMS; ++idx) {
        double v = input_image[idx] * scale;
        // Clamp to int16_t range
        if (v > 32767.0) v = 32767.0;
        if (v < -32768.0) v = -32768.0;
        // Round
        int64_t q = (int64_t)(v < 0 ? v - 0.5 : v + 0.5);
        if (q > 32767) q = 32767;
        if (q < -32768) q = -32768;
        output_buffer[idx] = (int16_t)q;
    }
    
    return 0;
}

/**
 * Execute single convolutional layer
 */
int yolo2_inference_conv_layer(yolo2_inference_context_t *ctx,
                               int layer_idx,
                               int ifm_num,
                               int ofm_num,
                               int ksize,
                               int kstride,
                               int input_w,
                               int input_h,
                               int output_w,
                               int output_h,
                               int padding,
                               int is_nl,
                               int is_bn,
                               int tm,
                               int tn,
                               int tr,
                               int tc,
                               int ofm_num_bound,
                               int mloopsxTM,
                               int mloops_a1xTM)
{
    // Check pointers
    if (!ctx->in_ptr[layer_idx] || !ctx->out_ptr[layer_idx]) {
        fprintf(stderr, "ERROR: Layer %d: Invalid pointers (in=%p, out=%p)\n", 
                layer_idx, (void*)ctx->in_ptr[layer_idx], (void*)ctx->out_ptr[layer_idx]);
        return -1;
    }
    
    // Bounds check for weight/bias offsets
    size_t weights_elems = ctx->weights_buf.size / sizeof(int16_t);
    size_t bias_elems = ctx->bias_buf.size / sizeof(int16_t);
    
    if (ctx->woffset >= weights_elems) {
        fprintf(stderr, "ERROR: Layer %d: woffset %zu >= weights size %zu\n", 
                layer_idx, ctx->woffset, weights_elems);
        return -1;
    }
    if (ctx->boffset >= bias_elems) {
        fprintf(stderr, "ERROR: Layer %d: boffset %zu >= bias size %zu\n", 
                layer_idx, ctx->boffset, bias_elems);
        return -1;
    }
    
    uint64_t input_addr = memory_get_phys_addr(ctx->in_ptr[layer_idx]);
    uint64_t output_addr = memory_get_phys_addr(ctx->out_ptr[layer_idx]);
    // woffset and boffset are in ELEMENTS (words), not bytes
    uint64_t weight_addr = memory_get_phys_addr((int16_t *)ctx->weights_buf.ptr + ctx->woffset);
    uint64_t beta_addr = memory_get_phys_addr((int16_t *)ctx->bias_buf.ptr + ctx->boffset);
    
    // Get Q values (INT16 mode)
    int32_t Qw = 0, Qa_in = 0, Qa_out = 0, Qb = 0;
    
    if (ctx->weight_q && ctx->offset_index < (int)ctx->weight_q_size) {
        Qw = ctx->weight_q[ctx->offset_index];
    }
    if (ctx->bias_q && ctx->offset_index < (int)ctx->bias_q_size) {
        Qb = ctx->bias_q[ctx->offset_index];
    }
    if (ctx->act_q && ctx->offset_index < (int)ctx->act_q_size) {
        Qa_in = ctx->act_q[ctx->offset_index];
    }
    if (ctx->act_q && (ctx->offset_index + 1) < (int)ctx->act_q_size) {
        Qa_out = ctx->act_q[ctx->offset_index + 1];
    } else if (ctx->act_q && ctx->offset_index < (int)ctx->act_q_size) {
        Qa_out = ctx->act_q[ctx->offset_index];
    }
    
    // Use pending route Q if set
    if (ctx->pending_route_q >= 0) {
        Qa_in = ctx->pending_route_q;
        ctx->pending_route_q = -1;
    }
    
    // Update current Q
    ctx->current_Qa = Qa_out;
    
    YOLO2_LOG_LAYER("    Layer %d: Qw=%d, Qb=%d, Qa_in=%d, Qa_out=%d\n",
                    layer_idx, Qw, Qb, Qa_in, Qa_out);
    
    // Debug: Print physical addresses being sent to accelerator
    if (yolo2_get_verbosity() >= 3) {
        printf("    [DEBUG] Physical addresses:\n");
        printf("      input_addr  = 0x%lx (virt=%p)\n", (unsigned long)input_addr, (void*)ctx->in_ptr[layer_idx]);
        printf("      output_addr = 0x%lx (virt=%p)\n", (unsigned long)output_addr, (void*)ctx->out_ptr[layer_idx]);
        printf("      weight_addr = 0x%lx (woffset=%zu)\n", (unsigned long)weight_addr, ctx->woffset);
        printf("      beta_addr   = 0x%lx (boffset=%zu)\n", (unsigned long)beta_addr, ctx->boffset);
    }
    
    // CRITICAL: Flush cache for all buffers before starting accelerator
    // The accelerator uses DMA to access these buffers, so they must be
    // flushed to memory before the accelerator can read/write them
    size_t input_size = input_w * input_h * ifm_num * sizeof(int16_t);
    size_t output_size = output_w * output_h * ofm_num * sizeof(int16_t);
    memory_flush_cache(ctx->in_ptr[layer_idx], input_size);
    memory_flush_cache(ctx->weights_buf.ptr, 
                       (ctx->woffset + (ifm_num * ofm_num * ksize * ksize)) * sizeof(int16_t));
    memory_flush_cache(ctx->bias_buf.ptr, 
                       (ctx->boffset + ofm_num) * sizeof(int16_t));
    
    // Execute layer
    int result = yolo2_execute_conv_layer(
        input_addr, output_addr, weight_addr, beta_addr,
        ifm_num, ofm_num, ksize, kstride,
        input_w, input_h, output_w, output_h, padding,
        is_nl, is_bn, tm, tn, tr, tc,
        ofm_num_bound, mloopsxTM, mloops_a1xTM,
        0, // layer_type = CONV
        Qw, Qa_in, Qa_out, Qb,
        yolo2_get_layer_timeout_ms()
    );
    
    if (result == YOLO2_SUCCESS) {
        // Save layer-24 output Q for later route/reorg concat alignment (route layer 28).
        if (layer_idx == 24) {
            ctx->route24_q = ctx->current_Qa;
            YOLO2_LOG_LAYER("    Stored route24_q=%d for reorg/route alignment\n", ctx->route24_q);
        }

        // Update offsets (in ELEMENTS, not bytes)
        if (ctx->offset_index < (int)NUM_WEIGHT_OFFSETS) {
            ctx->woffset += weight_offsets[ctx->offset_index];
        }
        if (ctx->offset_index < (int)NUM_BETA_OFFSETS) {
            ctx->boffset += beta_offsets[ctx->offset_index];
        }
        ctx->offset_index++;
    }
    
    return result;
}

/**
 * Execute maxpool layer
 */
int yolo2_inference_maxpool_layer(yolo2_inference_context_t *ctx,
                                  int layer_idx,
                                  int channels,
                                  int ksize,
                                  int kstride,
                                  int input_w,
                                  int input_h,
                                  int output_w,
                                  int output_h,
                                  int padding,
                                  int tm,
                                  int tr,
                                  int tc,
                                  int ofm_num_bound,
                                  int mloopsxTM,
                                  int mloops_a1xTM)
{
    if (!ctx->in_ptr[layer_idx] || !ctx->out_ptr[layer_idx]) {
        fprintf(stderr, "ERROR: Layer %d: Invalid pointers\n", layer_idx);
        return -1;
    }
    
    uint64_t input_addr = memory_get_phys_addr(ctx->in_ptr[layer_idx]);
    uint64_t output_addr = memory_get_phys_addr(ctx->out_ptr[layer_idx]);

    YOLO2_LOG_LAYER("    Maxpool %d: tm=%d tr=%d tc=%d ofm_num_bound=%d mLoopsxTM=%d mLoops_a1xTM=%d\n",
                    layer_idx, tm, tr, tc, ofm_num_bound, mloopsxTM, mloops_a1xTM);

    return yolo2_execute_maxpool_layer(
        input_addr, output_addr,
        channels, ksize, kstride,
        input_w, input_h, output_w, output_h, padding,
        tm, tr, tc, ofm_num_bound, mloopsxTM, mloops_a1xTM,
        yolo2_get_layer_timeout_ms()
    );
}

/**
 * Dequantize output
 */
int yolo2_dequantize_output(int16_t *input, float *output, size_t count, int32_t q_out)
{
    if (q_out <= 0) {
        const unsigned int shift = (unsigned int)(q_out < 0 ? -q_out : 0);
        const float scale = (float)(1ULL << shift);
        for (size_t i = 0; i < count; ++i) {
            output[i] = (float)input[i] * scale;
        }
    } else {
        const float scale = 1.0f / (float)(1ULL << (unsigned int)q_out);
        for (size_t i = 0; i < count; ++i) {
            output[i] = (float)input[i] * scale;
        }
    }
    
    return 0;
}

// Memory layout constants
#define ROUTE16_LEN (26*32*512)
#define CONV27_LEN (13*16*256)
#define CONV24_LEN (13*16*1024)
#define DETECTION_WORKSPACE (3*13*425)

/**
 * Generate IOFM offset pointers (memory layout)
 */
int yolo2_generate_iofm_offset(yolo2_inference_context_t *ctx) {
    if (!ctx || !ctx->inference_buf.ptr || !ctx->net) {
        fprintf(stderr, "ERROR: Invalid context for generate_iofm_offset\n");
        return -1;
    }
    
    int16_t *Memory_buf = (int16_t *)ctx->inference_buf.ptr;
    int16_t *Memory_top = Memory_buf + 512;
    int16_t *Memory_bottom = Memory_top + MEM_LEN;
    network_t *net = ctx->net;
    
    // Layers 0-17: Standard ping-pong
    for (int x = 0; x < 18 && x < net->n; x++) {
        int out_w = net->layers[x].out_w;
        int out_w_align_256b = (out_w >> 3) << 3;
        if (out_w & 0x7) {
            out_w_align_256b += 8;
        }
        
        if (x % 2 == 0) {
            ctx->in_ptr[x] = Memory_top;
            ctx->out_ptr[x] = Memory_bottom - net->layers[x].out_c * net->layers[x].out_h * out_w_align_256b;
        } else {
            ctx->in_ptr[x] = ctx->out_ptr[x-1];
            ctx->out_ptr[x] = Memory_top;
        }
    }
    
    // Layers 18-24: With route16_len offset
    for (int x = 18; x < 25 && x < net->n; x++) {
        int out_w = net->layers[x].out_w;
        int out_w_align_256b = (out_w >> 3) << 3;
        if (out_w & 0x7) {
            out_w_align_256b += 8;
        }
        
        if (x % 2 == 0) {
            ctx->in_ptr[x] = Memory_top;
            ctx->out_ptr[x] = Memory_bottom - ROUTE16_LEN - net->layers[x].out_c * net->layers[x].out_h * out_w_align_256b;
        } else {
            ctx->in_ptr[x] = ctx->out_ptr[x-1];
            ctx->out_ptr[x] = Memory_top;
        }
    }
    
    // Layer 26: Route layer input
    if (26 < net->n) {
        ctx->in_ptr[26] = Memory_bottom - ROUTE16_LEN;
        ctx->out_ptr[26] = Memory_top;
    }
    
    // Layer 27: After route concatenation
    if (27 < net->n) {
        ctx->in_ptr[27] = Memory_top;
        ctx->out_ptr[27] = Memory_bottom - ROUTE16_LEN - CONV24_LEN - CONV27_LEN;
    }
    
    // Layer 29: After layer 27
    if (29 < net->n) {
        ctx->in_ptr[29] = ctx->out_ptr[27];
        ctx->out_ptr[29] = Memory_top;
    }
    
    // Layer 30: Final conv before region
    if (30 < net->n) {
        ctx->in_ptr[30] = Memory_top;
        ctx->out_ptr[30] = Memory_bottom - (net->layers[30].outputs + DETECTION_WORKSPACE);
    }
    
    // Layer 31: Region layer
    if (31 < net->n) {
        ctx->in_ptr[31] = ctx->out_ptr[30];
        ctx->out_ptr[31] = NULL;
    }
    
    return 0;
}

// Reorg CPU implementation
static void reorg_cpu(int16_t *x, int w, int h, int c, int stride, int16_t *out) {
    int out_c = c / (stride * stride);
    
    for (int k = 0; k < c; ++k) {
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                int in_index = i + w * (j + h * k);
                int c2 = k % out_c;
                int offset = k / out_c;
                int w2 = i * stride + offset % stride;
                int h2 = j * stride + offset / stride;
                int out_index = w2 + w * stride * (h2 + h * stride * c2);
                out[in_index] = x[out_index];
            }
        }
    }
}

/**
 * Execute REORG layer on CPU
 */
int yolo2_execute_reorg_layer(yolo2_inference_context_t *ctx, int layer_idx, int stride) {
    if (!ctx || !ctx->net || layer_idx >= ctx->net->n) {
        fprintf(stderr, "ERROR: Invalid layer index for REORG\n");
        return -1;
    }
    
    int16_t *in_ptr = ctx->in_ptr[layer_idx];
    int16_t *out_ptr = ctx->out_ptr[layer_idx];
    
    if (!in_ptr || !out_ptr) {
        fprintf(stderr, "ERROR: REORG layer %d: Invalid pointers\n", layer_idx);
        return -1;
    }
    
    // Allocate temporary buffers
    int region_len = 13 * 16 * 256;
    int16_t *region_buf = (int16_t*)malloc(region_len * sizeof(int16_t));
    int16_t *region_buf2 = (int16_t*)malloc(region_len * sizeof(int16_t));
    
    if (!region_buf || !region_buf2) {
        fprintf(stderr, "ERROR: Failed to allocate reorg buffers\n");
        if (region_buf) free(region_buf);
        if (region_buf2) free(region_buf2);
        return -1;
    }
    
    // Copy from input to region_buf
    int16_t *tmp_ptr_f0 = in_ptr;
    for (int k = 0; k < 26 * 64; k++) {
        memcpy(region_buf + k * 26, tmp_ptr_f0 + k * 32, 26 * sizeof(int16_t));
    }
    
    // Perform reorg
    reorg_cpu(region_buf, 26, 32 * 13, 4, stride, region_buf2);
    
    // Copy back
    tmp_ptr_f0 = region_buf;
    memset(region_buf, 0, 13 * 16 * 256 * sizeof(int16_t));
    for (int k = 0; k < 13 * 256; k++) {
        memcpy(tmp_ptr_f0 + k * 16, region_buf2 + k * 13, 13 * sizeof(int16_t));
    }
    
    // Q alignment for route layer concatenation.
    // Keep in sync with `hls/models/yolov2/yolo2_model.cpp`: only the reorg branch is rescaled.
    if (ctx->route24_q > 0 && ctx->current_Qa > 0) {
        const int target_q = (ctx->route24_q < ctx->current_Qa) ? ctx->route24_q : ctx->current_Qa;
        const int shift = ctx->current_Qa - target_q;
        if (shift != 0) {
            YOLO2_LOG_LAYER("    Aligning Q scales: current_Qa=%d, route24_q=%d, target=%d, shift=%d\n",
                            ctx->current_Qa, ctx->route24_q, target_q, shift);
            yolo2_apply_q_shift_int16(tmp_ptr_f0, (size_t)(13 * 16 * 256), shift);
            ctx->current_Qa = target_q;
        }
        ctx->pending_route_q = ctx->current_Qa;
    }
    
    // Copy to output
    memcpy(out_ptr, tmp_ptr_f0, 13 * 16 * 256 * sizeof(int16_t));
    
    // Sync for device
    memory_flush_cache(out_ptr, 13 * 16 * 256 * sizeof(int16_t));
    
    free(region_buf);
    free(region_buf2);
    
    return 0;
}

/**
 * Execute ROUTE layer
 */
int yolo2_execute_route_layer(yolo2_inference_context_t *ctx, int layer_idx) {
    if (!ctx || !ctx->net || layer_idx >= ctx->net->n) {
        fprintf(stderr, "ERROR: Invalid layer index for ROUTE\n");
        return -1;
    }
    
    // Route layer 28 ("layers=-1,-4"): concatenates reorg output (27) with conv24 output (24).
    if (layer_idx == 28) {
        YOLO2_LOG_LAYER("    ROUTE layer 28: Concatenating layers 27 and 24\n");
    }
    
    return 0;
}

/**
 * Execute REGION layer
 */
int yolo2_execute_region_layer(yolo2_inference_context_t *ctx, int layer_idx) {
    if (!ctx || !ctx->net || layer_idx >= ctx->net->n) {
        fprintf(stderr, "ERROR: Invalid layer index for REGION\n");
        return -1;
    }
    
    int16_t *in_ptr = ctx->in_ptr[layer_idx];
    
    if (!in_ptr) {
        fprintf(stderr, "ERROR: REGION layer %d: Invalid input pointer\n", layer_idx);
        return -1;
    }
    
    // Convert format: input is 13x16x425 (padded), output is 13x13x425 (actual)
    int region_output_len = 13 * 13 * 425;  // 71825 elements
    int16_t *region_buf = (int16_t*)malloc(region_output_len * sizeof(int16_t));
    if (!region_buf) {
        fprintf(stderr, "ERROR: Failed to allocate region buffer\n");
        return -1;
    }
    memset(region_buf, 0, region_output_len * sizeof(int16_t));
    
    // Sync for CPU
    memory_invalidate_cache(in_ptr, 13 * 16 * 425 * sizeof(int16_t));
    
    // Convert: 13x16x425 -> 13x13x425
    // The data is arranged as 425 channels of 13x16 (padded from 13x13)
    int16_t *tmp_ptr_f0 = in_ptr;
    for (int k = 0; k < 13 * 425; k++) {
        for (int j = 0; j < 16; j++) {
            if (j < 13) {
                region_buf[k * 13 + j] = tmp_ptr_f0[k * 16 + j];
            }
        }
    }
    
    // Dequantize to float
    if (!ctx->region_output || ctx->region_output_size != (size_t)region_output_len) {
        free(ctx->region_output);
        ctx->region_output = (float*)malloc((size_t)region_output_len * sizeof(float));
        if (!ctx->region_output) {
            fprintf(stderr, "ERROR: Failed to allocate float buffer for region\n");
            ctx->region_output_size = 0;
            ctx->region_layer_idx = -1;
            free(region_buf);
            return -1;
        }
        ctx->region_output_size = (size_t)region_output_len;
    }

    float *region_f = ctx->region_output;
    
    if (ctx->act_q && ctx->act_q_size > 0) {
        const int q_out = ctx->current_Qa;
        float scale;
        if (q_out <= 0) {
            const unsigned int shift = (unsigned int)(q_out < 0 ? -q_out : 0);
            scale = (float)(1ULL << shift);
        } else {
            scale = 1.0f / (float)(1ULL << (unsigned int)q_out);
        }
        
        YOLO2_LOG_INFO("    Dequantizing region output with current_Qa=%d (scale=%.6f)\n", q_out, scale);
        for (int t = 0; t < region_output_len; ++t) {
            region_f[t] = (float)region_buf[t] * scale;
        }
    } else {
        for (int t = 0; t < region_output_len; ++t) {
            region_f[t] = (float)region_buf[t];
        }
    }
    
    // Store output
    ctx->region_layer_idx = layer_idx;
    
    YOLO2_LOG_INFO("    REGION layer output dequantized: %d elements\n", region_output_len);
    
    free(region_buf);
    
    return 0;
}

/**
 * Get region layer output
 */
float* yolo2_get_region_output(yolo2_inference_context_t *ctx, int layer_idx, size_t *output_size) {
    if (!ctx) {
        return NULL;
    }
    
    if (ctx->region_output && ctx->region_layer_idx == layer_idx) {
        if (output_size) {
            *output_size = ctx->region_output_size;
        }
        return ctx->region_output;
    }
    
    return NULL;
}

/**
 * Run complete inference pipeline
 */
int yolo2_run_inference(yolo2_inference_context_t *ctx, float *input_image) {
    if (!ctx || !ctx->net || !input_image) {
        fprintf(stderr, "ERROR: Invalid context or input image\n");
        return -1;
    }
    
    network_t *net = ctx->net;
    int TR, TC, TM, TN;
    int output_w, output_h;
    int mLoops;
    
    YOLO2_LOG_INFO("\n[Inference Engine v%s]\n", INFERENCE_VERSION);
    YOLO2_LOG_INFO("Starting inference through %d layers...\n", net->n);
    
    // Generate memory layout
    if (yolo2_generate_iofm_offset(ctx) != 0) {
        fprintf(stderr, "ERROR: Failed to generate IOFM offsets\n");
        return -1;
    }
    
    // Quantize and copy input image
    if (ctx->act_q && ctx->act_q_size > 0) {
        const int q_in = ctx->act_q[0];
        ctx->current_Qa = q_in;
        YOLO2_LOG_INFO("Quantizing input with Q=%d\n", q_in);
        yolo2_process_input_image(input_image, ctx->in_ptr[0], q_in);
        memory_flush_cache(ctx->in_ptr[0], INPUT_ELEMS * sizeof(int16_t));
    } else {
        fprintf(stderr, "ERROR: FP32 mode not supported in this implementation\n");
        return -1;
    }
    
    // Reset offsets
    ctx->offset_index = 0;
    ctx->woffset = 0;
    ctx->boffset = 0;
    ctx->route24_q = 0;
    ctx->pending_route_q = -1;
    
    // Run through all layers
    for (int i = 0; i < net->n; ++i) {
        layer_t *l = &net->layers[i];
        
        YOLO2_LOG_LAYER("  Processing Layer %d (Type: %d)...\n", i, l->type);
        
        switch (l->type) {
            case LAYER_CONVOLUTIONAL: {
                output_w = (l->w - l->size + 2 * l->pad) / l->stride + 1;
                output_h = (l->h - l->size + 2 * l->pad) / l->stride + 1;
                
                TR = ((OnChipIB_Height - l->size) / l->stride + 1) < Tr ? 
                     ((OnChipIB_Height - l->size) / l->stride + 1) : Tr;
                TR = output_h < TR ? output_h : TR;
                TC = ((OnChipIB_Width - l->size) / l->stride + 1) < Tc ? 
                     ((OnChipIB_Width - l->size) / l->stride + 1) : Tc;
                TC = output_w < TC ? output_w : TC;
                TM = l->filters < Tm ? l->filters : Tm;
                TN = l->c < Tn ? l->c : Tn;
                mLoops = (int)ceil(((float)l->filters) / TM);
                
                int result = yolo2_inference_conv_layer(ctx, i,
                    l->c, l->filters, l->size, l->stride,
                    l->w, l->h, output_w, output_h, l->pad,
                    (l->activation == ACT_LEAKY) ? 1 : 0,
                    l->batch_normalize ? 1 : 0,
                    TM, TN, TR, TC,
                    (mLoops + 1) * TM, mLoops * TM, (mLoops + 1) * TM);
                
                if (result != 0) {
                    fprintf(stderr, "ERROR: Conv layer %d failed\n", i);
                    return -1;
                }
                
                memory_invalidate_cache(ctx->out_ptr[i], output_w * output_h * l->filters * sizeof(int16_t));
                break;
            }
            case LAYER_MAXPOOL: {
                output_w = l->out_w;
                output_h = l->out_h;
                
                TR = ((OnChipIB_Height - l->size) / l->stride + 1) < Tr ? 
                     ((OnChipIB_Height - l->size) / l->stride + 1) : Tr;
                TC = ((OnChipIB_Width - l->size) / l->stride + 1) < Tc ? 
                     ((OnChipIB_Width - l->size) / l->stride + 1) : Tc;
                TR = output_h < TR ? output_h : TR;
                TC = output_w < TC ? output_w : TC;
                TM = Tm < Tn ? Tm : Tn;
                TM = l->c < TM ? l->c : TM;
                mLoops = (int)ceil(((float)l->c) / TM);
                
                int result = yolo2_inference_maxpool_layer(ctx, i,
                    l->c, l->size, l->stride,
                    l->w, l->h, output_w, output_h, l->pad,
                    TM, TR, TC,
                    (mLoops + 2) * TM,    // OFM_num_bound
                    mLoops * TM,          // mLoopsxTM
                    (mLoops + 1) * TM);   // mLoops_a1xTM
                
                if (result != 0) {
                    fprintf(stderr, "ERROR: Maxpool layer %d failed\n", i);
                    return -1;
                }
                
                memory_invalidate_cache(ctx->out_ptr[i], output_w * output_h * l->c * sizeof(int16_t));
                break;
            }
            case LAYER_REORG: {
                int result = yolo2_execute_reorg_layer(ctx, i, l->stride);
                if (result != 0) {
                    fprintf(stderr, "ERROR: Reorg layer %d failed\n", i);
                    return -1;
                }
                break;
            }
            case LAYER_ROUTE: {
                int result = yolo2_execute_route_layer(ctx, i);
                if (result != 0) {
                    fprintf(stderr, "ERROR: Route layer %d failed\n", i);
                    return -1;
                }
                break;
            }
            case LAYER_REGION: {
                int result = yolo2_execute_region_layer(ctx, i);
                if (result != 0) {
                    fprintf(stderr, "ERROR: Region layer %d failed\n", i);
                    return -1;
                }
                break;
            }
            default:
                YOLO2_LOG_LAYER("    Layer %d: UNKNOWN type %d (skipping)\n", i, l->type);
                break;
        }
    }
    
    YOLO2_LOG_INFO("\nInference completed successfully!\n");
    return 0;
}
