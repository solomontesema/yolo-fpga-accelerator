/**
 * YOLOv2 Inference Engine - Linux Implementation
 * 
 * Orchestrates layer-by-layer inference through the FPGA accelerator.
 */

#ifndef YOLO2_INFERENCE_H
#define YOLO2_INFERENCE_H

#include <stdint.h>
#include "dma_buffer_manager.h"
#include "yolo2_network.h"

/**
 * Inference context structure
 * Contains all state needed for running inference
 */
typedef struct {
    // Memory buffers
    memory_buffer_t weights_buf;
    memory_buffer_t bias_buf;
    memory_buffer_t inference_buf;
    
    // Q values (INT16 quantization mode)
    int32_t *weight_q;
    int32_t *bias_q;
    int32_t *act_q;
    size_t weight_q_size;
    size_t bias_q_size;
    size_t act_q_size;
    
    // Layer tracking
    int current_layer;
    int offset_index;
    int woffset;
    int boffset;
    
    // Q value tracking (INT16 mode)
    int current_Qa;
    int route24_q;
    int pending_route_q;
    
    // Memory layout pointers
    int16_t *in_ptr[32];
    int16_t *out_ptr[32];
    
    // Network structure
    network_t *net;
    
    // Region layer output (dequantized, for post-processing)
    float *region_output;
    size_t region_output_size;
    int region_layer_idx;
} yolo2_inference_context_t;

/**
 * Initialize inference context
 * 
 * ctx: Context to initialize
 * Returns: 0 on success, -1 on error
 */
int yolo2_inference_init(yolo2_inference_context_t *ctx);

/**
 * Cleanup inference context
 * Frees all allocated resources
 */
void yolo2_inference_cleanup(yolo2_inference_context_t *ctx);

/**
 * Process input image (quantize for INT16 mode)
 * 
 * input_image: Float image data [0-1] normalized
 * output_buffer: INT16 quantized output
 * q_in: Input quantization Q value
 */
int yolo2_process_input_image(float *input_image, int16_t *output_buffer, 
                              int32_t q_in);

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
                               int mloops_a1xTM);

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
                                  int mloops_a1xTM);

/**
 * Dequantize output (INT16 mode)
 */
int yolo2_dequantize_output(int16_t *input, float *output, size_t count, int32_t q_out);

/**
 * Generate IOFM offset pointers (memory layout)
 */
int yolo2_generate_iofm_offset(yolo2_inference_context_t *ctx);

/**
 * Execute REORG layer on CPU
 */
int yolo2_execute_reorg_layer(yolo2_inference_context_t *ctx, int layer_idx, int stride);

/**
 * Execute ROUTE layer (memory pointer management)
 */
int yolo2_execute_route_layer(yolo2_inference_context_t *ctx, int layer_idx);

/**
 * Execute REGION layer (post-processing)
 */
int yolo2_execute_region_layer(yolo2_inference_context_t *ctx, int layer_idx);

/**
 * Run complete inference pipeline
 * 
 * ctx: Inference context (must be initialized with weights/network)
 * input_image: Float input image [0-1] normalized
 * Returns: 0 on success, -1 on error
 */
int yolo2_run_inference(yolo2_inference_context_t *ctx, float *input_image);

/**
 * Get region layer output (for post-processing)
 */
float* yolo2_get_region_output(yolo2_inference_context_t *ctx, int layer_idx, size_t *output_size);

#endif /* YOLO2_INFERENCE_H */
