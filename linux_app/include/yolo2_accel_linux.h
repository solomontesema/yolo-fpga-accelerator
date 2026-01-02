/**
 * YOLOv2 FPGA Accelerator - Linux Driver Header
 * 
 * Provides userspace access to the HLS-generated accelerator via /dev/mem.
 * The accelerator control registers are mapped at 0xA0000000.
 */

#ifndef YOLO2_ACCEL_LINUX_H
#define YOLO2_ACCEL_LINUX_H

#include <stdint.h>

/**
 * Initialize accelerator driver
 * Maps control registers and GPIO peripherals via /dev/mem
 * 
 * Returns: YOLO2_SUCCESS on success, error code on failure
 */
int yolo2_accel_init(void);

/**
 * Cleanup accelerator driver
 * Unmaps all memory regions and closes /dev/mem
 */
void yolo2_accel_cleanup(void);

/**
 * Set Q values via AXI GPIO (INT16 quantization mode)
 * 
 * qw:     Weight quantization shift
 * qa_in:  Input activation quantization shift
 * qa_out: Output activation quantization shift
 * qb:     Bias quantization shift
 */
void yolo2_set_q_values(int32_t qw, int32_t qa_in, int32_t qa_out, int32_t qb);

/**
 * Check if accelerator is currently processing
 * Returns: 1 if busy, 0 if idle
 */
int yolo2_is_busy(void);

/**
 * Check if accelerator has completed
 * Returns: 1 if done, 0 if not done
 */
int yolo2_is_done(void);

/**
 * Wait for accelerator completion (polling)
 * 
 * timeout_ms: Maximum wait time in milliseconds (0 = infinite)
 * Returns: YOLO2_SUCCESS on completion, YOLO2_TIMEOUT on timeout
 */
int yolo2_wait_for_completion(uint32_t timeout_ms);

/**
 * Get accelerator status register value
 * Returns: Raw status register value
 */
uint32_t yolo2_get_status(void);

/**
 * Execute convolutional layer on accelerator
 * 
 * All buffer addresses must be physical addresses accessible by the accelerator.
 * The accelerator performs DMA directly to/from DDR via AXI HP ports.
 */
int yolo2_execute_conv_layer(
    uint64_t input_addr,      // Physical address of input buffer
    uint64_t output_addr,     // Physical address of output buffer
    uint64_t weight_addr,     // Physical address of weight buffer
    uint64_t beta_addr,       // Physical address of beta/bias buffer
    int ifm_num,              // Input feature map channels
    int ofm_num,              // Output feature map channels
    int ksize,                // Kernel size
    int kstride,              // Kernel stride
    int input_w,              // Input width
    int input_h,              // Input height
    int output_w,             // Output width
    int output_h,             // Output height
    int padding,              // Padding size
    int is_nl,                // Non-linear activation (1=leaky ReLU, 0=none)
    int is_bn,                // Batch normalization (1=enabled, 0=disabled)
    int tm,                   // Tile M (output channels)
    int tn,                   // Tile N (input channels)
    int tr,                   // Tile R (rows)
    int tc,                   // Tile C (columns)
    int ofm_num_bound,        // OFM bound for tiling
    int mloopsxTM,            // mLoops * TM
    int mloops_a1xTM,         // (mLoops+1) * TM
    int layer_type,           // Layer type (0=conv)
    int qw,                   // Weight Q value
    int qa_in,                // Input activation Q value
    int qa_out,               // Output activation Q value
    int qb,                   // Bias Q value
    uint32_t timeout_ms       // Timeout in milliseconds
);

/**
 * Execute maxpool layer on accelerator
 */
int yolo2_execute_maxpool_layer(
    uint64_t input_addr,      // Physical address of input buffer
    uint64_t output_addr,     // Physical address of output buffer
    int channels,             // Number of channels
    int ksize,                // Pool size
    int kstride,              // Pool stride
    int input_w,              // Input width
    int input_h,              // Input height
    int output_w,             // Output width
    int output_h,             // Output height
    int padding,              // Padding size
    int tm,                   // Tile M
    int tr,                   // Tile R
    int tc,                   // Tile C
    int ofm_num_bound,        // OFM bound (maxpool pipeline flush)
    int mloopsxTM,            // mLoops * TM
    int mloops_a1xTM,         // (mLoops+1) * TM
    uint32_t timeout_ms       // Timeout in milliseconds
);

/**
 * Read register value (for debugging)
 * offset: Register offset from base address
 */
uint32_t yolo2_read_reg(uint32_t offset);

/**
 * Write register value (for debugging)
 * offset: Register offset from base address
 * value: Value to write
 */
void yolo2_write_reg(uint32_t offset, uint32_t value);

#endif /* YOLO2_ACCEL_LINUX_H */
