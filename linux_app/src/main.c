/**
 * YOLOv2 FPGA Accelerator - Linux Main Application
 * 
 * This application runs object detection using the custom HLS accelerator
 * on the AMD/Xilinx KV260 board.
 * 
 * Usage: sudo ./yolo2_linux [options]
 *   -i <image>    Input image path (default: /home/ubuntu/test_images/dog.jpg)
 *   -w <dir>      Weights directory (default: /home/ubuntu/weights)
 *   -c <config>   Network config file (default: /home/ubuntu/config/yolov2.cfg)
 *   -l <labels>   Labels file (default: /home/ubuntu/config/coco.names)
 *   -t <thresh>   Detection threshold (default: 0.24)
 *   -n <nms>      NMS threshold (default: 0.45)
 *   -h            Show this help
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <errno.h>

#include "yolo2_config.h"
#include "yolo2_accel_linux.h"
#include "dma_buffer_manager.h"
#include "yolo2_inference.h"
#include "yolo2_network.h"
#include "yolo2_image_loader.h"
#include "yolo2_postprocess.h"
#include "yolo2_labels.h"
#include "file_loader.h"
#include "yolo2_log.h"

// Default paths
static char weights_dir[512] = "/home/ubuntu/weights";
static char config_path[512] = "/home/ubuntu/config/yolov2.cfg";
static char labels_path[512] = "/home/ubuntu/config/coco.names";
static char image_path[512] = "/home/ubuntu/test_images/dog.jpg";
static float det_thresh = 0.24f;
static float nms_thresh = 0.45f;

static void print_usage(const char *prog_name) {
    printf("YOLOv2 FPGA Accelerator - Linux Application\n");
    printf("\n");
    printf("Usage: sudo %s [options]\n", prog_name);
    printf("\n");
    printf("Options:\n");
    printf("  -i <image>    Input image path (default: %s)\n", image_path);
    printf("  -w <dir>      Weights directory (default: %s)\n", weights_dir);
    printf("  -c <config>   Network config file (default: %s)\n", config_path);
    printf("  -l <labels>   Labels file (default: %s)\n", labels_path);
    printf("  -t <thresh>   Detection threshold (default: %.2f)\n", det_thresh);
    printf("  -n <nms>      NMS threshold (default: %.2f)\n", nms_thresh);
    printf("  -v <level>    Verbosity 0..3 (overrides YOLO2_VERBOSE)\n");
    printf("  -h            Show this help\n");
    printf("\n");
    printf("Notes:\n");
    printf("  - Must run with sudo for /dev/mem access\n");
    printf("  - Requires udmabuf kernel module for DMA buffers\n");
    printf("  - Requires FPGA bitstream to be loaded\n");
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static int dump_float_array_text(const char *path, const float *data, size_t count)
{
    if (!path || !path[0] || !data) {
        return -1;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open dump file %s: %s\n", path, strerror(errno));
        return -1;
    }

    for (size_t i = 0; i < count; ++i) {
        // One value per line for easy diffing.
        fprintf(fp, "%.9g\n", data[i]);
    }

    fclose(fp);
    YOLO2_LOG_INFO("  Dumped %zu floats to %s\n", count, path);
    return 0;
}

int main(int argc, char *argv[]) {
    int opt;
    int result;
    double start_time, end_time;
    
    // Parse command line arguments
    while ((opt = getopt(argc, argv, "i:w:c:l:t:n:v:h")) != -1) {
        switch (opt) {
            case 'i':
                strncpy(image_path, optarg, sizeof(image_path) - 1);
                break;
            case 'w':
                strncpy(weights_dir, optarg, sizeof(weights_dir) - 1);
                break;
            case 'c':
                strncpy(config_path, optarg, sizeof(config_path) - 1);
                break;
            case 'l':
                strncpy(labels_path, optarg, sizeof(labels_path) - 1);
                break;
            case 't':
                det_thresh = atof(optarg);
                break;
            case 'n':
                nms_thresh = atof(optarg);
                break;
            case 'v':
                yolo2_set_verbosity(atoi(optarg));
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }
    
    YOLO2_LOG_INFO("\n");
    YOLO2_LOG_INFO("========================================\n");
    YOLO2_LOG_INFO("YOLOv2 FPGA Accelerator - Linux\n");
    YOLO2_LOG_INFO("========================================\n");
    YOLO2_LOG_INFO("\n");
    YOLO2_LOG_INFO("Configuration:\n");
    YOLO2_LOG_INFO("  Image:      %s\n", image_path);
    YOLO2_LOG_INFO("  Weights:    %s\n", weights_dir);
    YOLO2_LOG_INFO("  Config:     %s\n", config_path);
    YOLO2_LOG_INFO("  Labels:     %s\n", labels_path);
    YOLO2_LOG_INFO("  Threshold:  %.2f\n", det_thresh);
    YOLO2_LOG_INFO("  NMS:        %.2f\n", nms_thresh);
    YOLO2_LOG_INFO("  Verbosity:  %d\n", yolo2_get_verbosity());
    YOLO2_LOG_INFO("\n");
    
    // Build weight file paths
    char weights_file[512], bias_file[512];
    char weight_q_file[512], bias_q_file[512], iofm_q_file[512];
    snprintf(weights_file, sizeof(weights_file), "%s/weights_reorg_int16.bin", weights_dir);
    snprintf(bias_file, sizeof(bias_file), "%s/bias_int16.bin", weights_dir);
    snprintf(weight_q_file, sizeof(weight_q_file), "%s/weight_int16_Q.bin", weights_dir);
    snprintf(bias_q_file, sizeof(bias_q_file), "%s/bias_int16_Q.bin", weights_dir);
    snprintf(iofm_q_file, sizeof(iofm_q_file), "%s/iofm_Q.bin", weights_dir);
    
    yolo2_inference_context_t ctx;
    void *weights_data = NULL, *bias_data = NULL;
    size_t weights_size = 0, bias_size = 0;
    float *input_image = NULL;
    char **labels = NULL;
    int num_labels = 0;
    
    // Initialize inference context
    yolo2_inference_init(&ctx);
    
    // Step 1: Initialize accelerator driver
    YOLO2_LOG_INFO("[1/8] Initializing accelerator driver...\n");
    result = yolo2_accel_init();
    if (result != YOLO2_SUCCESS) {
        fprintf(stderr, "ERROR: Accelerator initialization failed: %d\n", result);
        goto cleanup;
    }
    YOLO2_LOG_INFO("      Accelerator driver initialized OK\n\n");
    
    // Step 2: Initialize DMA buffer manager
    YOLO2_LOG_INFO("[2/8] Initializing DMA buffer manager...\n");
    result = dma_buffer_init();
    if (result != 0) {
        fprintf(stderr, "ERROR: DMA buffer initialization failed\n");
        goto cleanup;
    }
    YOLO2_LOG_INFO("      DMA buffer manager initialized OK\n\n");
    
    // Step 3: Load weights from filesystem
    YOLO2_LOG_INFO("[3/8] Loading weights...\n");
    result = load_weights(weights_file, &weights_data, &weights_size);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to load weights from %s\n", weights_file);
        goto cleanup;
    }
    
    result = load_bias(bias_file, &bias_data, &bias_size);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to load bias from %s\n", bias_file);
        goto cleanup;
    }
    YOLO2_LOG_INFO("      Weights: %zu bytes, Bias: %zu bytes\n\n", weights_size, bias_size);
    
    // Step 4: Load Q values (INT16 mode)
    YOLO2_LOG_INFO("[4/8] Loading Q values...\n");
    result = load_q_values(weight_q_file, &ctx.weight_q, &ctx.weight_q_size);
    if (result != 0) {
        YOLO2_LOG_INFO("      WARNING: Weight Q values not found (using defaults)\n");
    }
    
    result = load_q_values(bias_q_file, &ctx.bias_q, &ctx.bias_q_size);
    if (result != 0) {
        YOLO2_LOG_INFO("      WARNING: Bias Q values not found (using defaults)\n");
    }
    
    result = load_q_values(iofm_q_file, &ctx.act_q, &ctx.act_q_size);
    if (result != 0) {
        YOLO2_LOG_INFO("      WARNING: Activation Q values not found (using defaults)\n");
    }
    
    if (ctx.act_q && ctx.act_q_size > 0) {
        ctx.current_Qa = ctx.act_q[0];
        YOLO2_LOG_INFO("      Q values loaded OK\n");
    }
    YOLO2_LOG_INFO("\n");
    
    // Step 5: Allocate DMA buffers
    YOLO2_LOG_INFO("[5/8] Allocating DMA buffers...\n");
    
    result = memory_allocate_weights(weights_size, &ctx.weights_buf);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to allocate weights buffer\n");
        goto cleanup;
    }
    
    result = memory_allocate_bias(bias_size, &ctx.bias_buf);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to allocate bias buffer\n");
        goto cleanup;
    }
    
    result = memory_allocate_inference_buffer(&ctx.inference_buf);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to allocate inference buffer\n");
        goto cleanup;
    }
    
    // Copy weights and bias to DMA buffers
    // Use chunked copy for large uncached DMA buffers to avoid bus errors
    YOLO2_LOG_INFO("      Copying weights to DMA buffers...\n");
    {
        const size_t chunk_size = 4096;  // Copy 4KB at a time
        size_t offset = 0;
        char *src = (char*)weights_data;
        volatile char *dst = (volatile char*)ctx.weights_buf.ptr;
        
        while (offset < weights_size) {
            size_t copy_len = (weights_size - offset > chunk_size) ? chunk_size : (weights_size - offset);
            for (size_t i = 0; i < copy_len; i++) {
                dst[offset + i] = src[offset + i];
            }
            offset += copy_len;
            
            // Progress indicator for large copy
            if ((offset % (10 * 1024 * 1024)) == 0) {
                YOLO2_LOG_INFO("        %zu MB copied...\n", offset / (1024 * 1024));
            }
        }
        __sync_synchronize();
        YOLO2_LOG_INFO("      Weights copied (%zu bytes)\n", weights_size);
    }
    
    // Copy bias (smaller, can use direct copy)
    {
        volatile char *dst = (volatile char*)ctx.bias_buf.ptr;
        char *src = (char*)bias_data;
        for (size_t i = 0; i < bias_size; i++) {
            dst[i] = src[i];
        }
        __sync_synchronize();
        YOLO2_LOG_INFO("      Bias copied (%zu bytes)\n", bias_size);
    }
    
    // Sync for device
    memory_flush_cache(ctx.weights_buf.ptr, weights_size);
    memory_flush_cache(ctx.bias_buf.ptr, bias_size);
    
    // Free temporary buffers
    free(weights_data);
    weights_data = NULL;
    free(bias_data);
    bias_data = NULL;
    
    YOLO2_LOG_INFO("      DMA buffers allocated OK\n\n");
    
    // Step 6: Parse network configuration
    YOLO2_LOG_INFO("[6/8] Parsing network configuration...\n");
    ctx.net = yolo2_parse_network_cfg(config_path);
    if (!ctx.net) {
        fprintf(stderr, "ERROR: Failed to parse network configuration\n");
        goto cleanup;
    }
    YOLO2_LOG_INFO("\n");
    
    // Step 7: Load input image
    YOLO2_LOG_INFO("[7/8] Loading input image...\n");
    input_image = (float*)malloc(INPUT_ELEMS * sizeof(float));
    if (!input_image) {
        fprintf(stderr, "ERROR: Failed to allocate input image buffer\n");
        goto cleanup;
    }
    
    result = yolo2_load_image(image_path, input_image);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to load image from %s\n", image_path);
        goto cleanup;
    }
    YOLO2_LOG_INFO("\n");
    
    // Debug: Test memory access pattern
    if (yolo2_get_verbosity() >= 3) {
        printf("\n[DEBUG] Testing memory write/read...\n");
        // Write test pattern to first few elements of inference buffer
        int16_t *test_buf = (int16_t *)ctx.inference_buf.ptr;
        uint64_t test_phys = ctx.inference_buf.phys_addr;
        
        printf("  Inference buffer: virt=%p, phys=0x%lx\n", 
               (void*)test_buf, (unsigned long)test_phys);
        
        // Write pattern
        for (int i = 0; i < 16; i++) {
            test_buf[i] = 0x1234 + i;
        }
        __sync_synchronize();
        
        // Read back
        printf("  Written: ");
        for (int i = 0; i < 8; i++) {
            printf("0x%04x ", (unsigned)test_buf[i]);
        }
        printf("\n");
        
        // Check if input data was written correctly
        int16_t *in_data = ctx.in_ptr[0];
        printf("  Input buffer ptr: %p (should be ~%p + 1024)\n", 
               (void*)in_data, (void*)test_buf);
    }
    
    // Step 8: Run inference
    YOLO2_LOG_INFO("\n[8/8] Running inference...\n");
    start_time = get_time_ms();
    
    result = yolo2_run_inference(&ctx, input_image);
    
    end_time = get_time_ms();
    
    if (result != 0) {
        fprintf(stderr, "ERROR: Inference failed\n");
        goto cleanup;
    }
    
    YOLO2_LOG_INFO("\nInference time: %.2f ms\n", end_time - start_time);
    
	    // Post-processing
	    if (ctx.region_output && ctx.region_layer_idx >= 0) {
	        layer_t *region_layer = &ctx.net->layers[ctx.region_layer_idx];
	        
	        YOLO2_LOG_INFO("\nRunning post-processing...\n");
        
        // Debug: Check region output values
        if (yolo2_get_verbosity() >= 3) {
            float min_val = ctx.region_output[0], max_val = ctx.region_output[0];
            float sum = 0;
            for (size_t i = 0; i < ctx.region_output_size; i++) {
                if (ctx.region_output[i] < min_val) min_val = ctx.region_output[i];
                if (ctx.region_output[i] > max_val) max_val = ctx.region_output[i];
                sum += ctx.region_output[i];
            }
            printf("  Region output stats: min=%.6f, max=%.6f, mean=%.6f\n",
                   min_val, max_val, sum / ctx.region_output_size);

            // Print first few values for debugging
            printf("  First 10 values: ");
            for (int i = 0; i < 10 && i < (int)ctx.region_output_size; i++) {
                printf("%.4f ", ctx.region_output[i]);
            }
            printf("\n");
        }

	        // Dump region outputs by default for easy CPU vs HW comparison.
	        // You can override the output paths via env vars or disable dumps entirely.
	        const char *disable_dumps = getenv("YOLO2_NO_DUMP");
	        const int do_dump = !(disable_dumps && disable_dumps[0] && disable_dumps[0] != '0');

	        // - `YOLO2_DUMP_REGION_RAW`: raw dequantized conv30 output (pre-sigmoid/softmax)
	        const char *dump_raw = getenv("YOLO2_DUMP_REGION_RAW");
	        const char *dump_raw_path = (dump_raw && dump_raw[0]) ? dump_raw : "yolov2_region_raw_hw.txt";
	        if (do_dump) {
	            dump_float_array_text(dump_raw_path, ctx.region_output, ctx.region_output_size);
	        }
	        
	        // Allocate output buffer
	        float *region_output_processed = (float*)malloc(ctx.region_output_size * sizeof(float));
	        if (!region_output_processed) {
            fprintf(stderr, "ERROR: Failed to allocate processed region output\n");
            goto cleanup;
        }
        
        // Forward region layer
	        result = yolo2_forward_region_layer(region_layer, ctx.region_output, region_output_processed);
	        if (result != 0) {
	            fprintf(stderr, "ERROR: Forward region layer failed\n");
	            free(region_output_processed);
	            goto cleanup;
	        }

	        // - `YOLO2_DUMP_REGION`: region output after sigmoid/softmax (what post-processing consumes)
	        const char *dump_processed = getenv("YOLO2_DUMP_REGION");
	        const char *dump_processed_path = (dump_processed && dump_processed[0]) ? dump_processed : "yolov2_region_proc_hw.txt";
	        if (do_dump) {
	            dump_float_array_text(dump_processed_path, region_output_processed, ctx.region_output_size);
	        }
        
        // Load labels
        yolo2_load_labels(labels_path, &labels, &num_labels);
        
        // Get detections
        const int max_dets = 1000;
        yolo2_detection_t *dets = (yolo2_detection_t*)malloc(max_dets * sizeof(yolo2_detection_t));
        if (!dets) {
            fprintf(stderr, "ERROR: Failed to allocate detections array\n");
            free(region_output_processed);
            goto cleanup;
        }
        
        // Use network input size for detection (letterbox will be corrected)
        int num_dets = yolo2_get_region_detections(region_layer, region_output_processed,
                                                   INPUT_WIDTH, INPUT_HEIGHT,
                                                   INPUT_WIDTH, INPUT_HEIGHT,
                                                   det_thresh, dets, max_dets);
        
        if (num_dets > 0) {
            // Apply NMS
            yolo2_do_nms_sort(dets, num_dets, region_layer->classes, nms_thresh);
            
            // Print detections
            yolo2_print_detections(dets, num_dets, det_thresh, 
                                   (const char**)labels, num_labels);
        } else {
            printf("\nNo detections found above threshold %.2f\n", det_thresh);
        }
        
        // Cleanup
        yolo2_free_detections(dets, num_dets);
        free(dets);
        free(region_output_processed);
    } else {
        fprintf(stderr, "WARNING: Region layer output not available for post-processing\n");
    }
    
    YOLO2_LOG_INFO("\nInference completed successfully!\n");
    result = 0;
    
cleanup:
    // Cleanup
    if (input_image) free(input_image);
    if (weights_data) free(weights_data);
    if (bias_data) free(bias_data);
    if (labels) yolo2_free_labels(labels, num_labels);
    if (ctx.net) yolo2_free_network(ctx.net);
    
    yolo2_inference_cleanup(&ctx);
    dma_buffer_cleanup();
    yolo2_accel_cleanup();
    
    YOLO2_LOG_INFO("\n========================================\n");
    YOLO2_LOG_INFO("Application finished\n");
    YOLO2_LOG_INFO("========================================\n\n");
    
    return result;
}
