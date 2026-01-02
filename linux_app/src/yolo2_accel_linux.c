/**
 * YOLOv2 FPGA Accelerator - Linux Driver Implementation
 * 
 * Provides userspace access to the HLS-generated accelerator via /dev/mem.
 * This is the standard approach for FPGA accelerators on Zynq UltraScale+.
 */

#include "yolo2_accel_linux.h"
#include "yolo2_config.h"
#include "yolo2_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <errno.h>

// Memory-mapped register pointers
static volatile uint32_t *ctrl_regs = NULL;
static volatile uint32_t *gpio_qw = NULL;
static volatile uint32_t *gpio_qa_in = NULL;
static volatile uint32_t *gpio_qa_out = NULL;
static volatile uint32_t *gpio_qb = NULL;

// File descriptor for /dev/mem
static int mem_fd = -1;

// Initialization flag
static int initialized = 0;

/**
 * Helper: Map physical address to virtual address via /dev/mem
 */
static void* map_physical(off_t phys_addr, size_t size)
{
    void *mapped = mmap(NULL, size, 
                       PROT_READ | PROT_WRITE, 
                       MAP_SHARED, 
                       mem_fd, 
                       phys_addr);
    
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed for 0x%lx: %s\n", 
                (unsigned long)phys_addr, strerror(errno));
        return NULL;
    }
    
    return mapped;
}

/**
 * Helper: Unmap memory region
 */
static void unmap_region(volatile void *addr, size_t size)
{
    if (addr && addr != MAP_FAILED) {
        munmap((void*)addr, size);
    }
}

/**
 * Initialize accelerator driver
 */
int yolo2_accel_init(void)
{
    if (initialized) {
        return YOLO2_SUCCESS;
    }
    
    YOLO2_LOG_INFO("Initializing YOLOv2 accelerator driver...\n");
    
    // Open /dev/mem
    mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        fprintf(stderr, "ERROR: Cannot open /dev/mem: %s\n", strerror(errno));
        fprintf(stderr, "       Run with sudo or ensure proper permissions\n");
        return YOLO2_MMAP_ERROR;
    }
    
    // Map accelerator control registers
    ctrl_regs = (volatile uint32_t*)map_physical(YOLO2_CTRL_BASE, YOLO2_CTRL_SIZE);
    if (!ctrl_regs) {
        fprintf(stderr, "ERROR: Failed to map control registers at 0x%lx\n", 
                (unsigned long)YOLO2_CTRL_BASE);
        close(mem_fd);
        mem_fd = -1;
        return YOLO2_MMAP_ERROR;
    }
    
    // Map Q value GPIOs
    gpio_qw = (volatile uint32_t*)map_physical(AXI_GPIO_QW_BASE, AXI_GPIO_SIZE);
    gpio_qa_in = (volatile uint32_t*)map_physical(AXI_GPIO_QA_IN_BASE, AXI_GPIO_SIZE);
    gpio_qa_out = (volatile uint32_t*)map_physical(AXI_GPIO_QA_OUT_BASE, AXI_GPIO_SIZE);
    gpio_qb = (volatile uint32_t*)map_physical(AXI_GPIO_QB_BASE, AXI_GPIO_SIZE);
    
    if (!gpio_qw || !gpio_qa_in || !gpio_qa_out || !gpio_qb) {
        fprintf(stderr, "ERROR: Failed to map GPIO registers\n");
        yolo2_accel_cleanup();
        return YOLO2_MMAP_ERROR;
    }
    
    // Initialize Q values to 0
    gpio_qw[GPIO_DATA_OFFSET / 4] = 0;
    gpio_qa_in[GPIO_DATA_OFFSET / 4] = 0;
    gpio_qa_out[GPIO_DATA_OFFSET / 4] = 0;
    gpio_qb[GPIO_DATA_OFFSET / 4] = 0;
    
    // Check accelerator status
    uint32_t status = ctrl_regs[CTRL_AP_CTRL / 4];
    YOLO2_LOG_INFO("  Accelerator status: 0x%02x", status);
    if (status & CTRL_AP_IDLE) YOLO2_LOG_INFO(" [IDLE]");
    if (status & CTRL_AP_DONE) YOLO2_LOG_INFO(" [DONE]");
    if (status & CTRL_AP_READY) YOLO2_LOG_INFO(" [READY]");
    YOLO2_LOG_INFO("\n");
    
    initialized = 1;
    YOLO2_LOG_INFO("  Accelerator driver initialized successfully\n");
    
    return YOLO2_SUCCESS;
}

/**
 * Cleanup accelerator driver
 */
void yolo2_accel_cleanup(void)
{
    if (ctrl_regs) {
        unmap_region(ctrl_regs, YOLO2_CTRL_SIZE);
        ctrl_regs = NULL;
    }
    if (gpio_qw) {
        unmap_region(gpio_qw, AXI_GPIO_SIZE);
        gpio_qw = NULL;
    }
    if (gpio_qa_in) {
        unmap_region(gpio_qa_in, AXI_GPIO_SIZE);
        gpio_qa_in = NULL;
    }
    if (gpio_qa_out) {
        unmap_region(gpio_qa_out, AXI_GPIO_SIZE);
        gpio_qa_out = NULL;
    }
    if (gpio_qb) {
        unmap_region(gpio_qb, AXI_GPIO_SIZE);
        gpio_qb = NULL;
    }
    
    if (mem_fd >= 0) {
        close(mem_fd);
        mem_fd = -1;
    }
    
    initialized = 0;
}

/**
 * Set Q values via AXI GPIO
 */
void yolo2_set_q_values(int32_t qw, int32_t qa_in, int32_t qa_out, int32_t qb)
{
    if (!initialized) return;
    
    YOLO2_LOG_DEBUG("    [DEBUG] Setting Q values via GPIO: Qw=%d, Qa_in=%d, Qa_out=%d, Qb=%d\n",
                    qw, qa_in, qa_out, qb);
    
    gpio_qw[GPIO_DATA_OFFSET / 4] = (uint32_t)qw;
    gpio_qa_in[GPIO_DATA_OFFSET / 4] = (uint32_t)qa_in;
    gpio_qa_out[GPIO_DATA_OFFSET / 4] = (uint32_t)qa_out;
    gpio_qb[GPIO_DATA_OFFSET / 4] = (uint32_t)qb;
    __sync_synchronize();
}

/**
 * Check if accelerator is busy
 */
int yolo2_is_busy(void)
{
    if (!initialized || !ctrl_regs) return 0;
    uint32_t status = ctrl_regs[CTRL_AP_CTRL / 4];
    return !(status & CTRL_AP_DONE);
}

/**
 * Check if accelerator is done
 */
int yolo2_is_done(void)
{
    if (!initialized || !ctrl_regs) return 1;
    uint32_t status = ctrl_regs[CTRL_AP_CTRL / 4];
    return (status & CTRL_AP_DONE) != 0;
}

/**
 * Get current time in milliseconds
 */
static uint64_t get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/**
 * Wait for accelerator completion
 */
int yolo2_wait_for_completion(uint32_t timeout_ms)
{
    if (!initialized || !ctrl_regs) return YOLO2_INIT_ERROR;
    
    uint64_t start_time = get_time_ms();
    
    while (!yolo2_is_done()) {
        if (timeout_ms > 0) {
            if ((get_time_ms() - start_time) > timeout_ms) {
                fprintf(stderr, "ERROR: Accelerator timeout after %u ms\n", timeout_ms);
                return YOLO2_TIMEOUT;
            }
        }
        // Small delay to reduce CPU usage
        usleep(100);
    }
    
    return YOLO2_SUCCESS;
}

/**
 * Get accelerator status
 */
uint32_t yolo2_get_status(void)
{
    if (!initialized || !ctrl_regs) return 0;
    return ctrl_regs[CTRL_AP_CTRL / 4];
}

/**
 * Read register value
 */
uint32_t yolo2_read_reg(uint32_t offset)
{
    if (!initialized || !ctrl_regs) return 0;
    return ctrl_regs[offset / 4];
}

/**
 * Write register value
 */
void yolo2_write_reg(uint32_t offset, uint32_t value)
{
    if (!initialized || !ctrl_regs) return;
    ctrl_regs[offset / 4] = value;
}

/**
 * Wait for accelerator completion with IDLE-based detection
 * 
 * HLS accelerator status bits:
 * - ap_done (bit 1): May be read-clear in some configurations
 * - ap_idle (bit 2): Set when accelerator is idle (reliable)
 * - ap_ready (bit 3): Set when ready for next input
 * 
 * Strategy: Wait for IDLE after we've started (IDLE goes low during operation)
 */
static int wait_for_idle(uint32_t timeout_ms)
{
    uint64_t start_time = get_time_ms();
    uint32_t status;
    int was_running = 0;
    uint32_t last_status = 0;
    int status_change_count = 0;
    
    // First, wait for accelerator to leave IDLE (start running)
    // Give it a few ms to start
    for (int i = 0; i < 100; i++) {
        status = ctrl_regs[CTRL_AP_CTRL / 4];
        // Read DONE bit to clear it (clear-on-read)
        if (status & CTRL_AP_DONE) {
            // DONE is set - read it to clear, then check if IDLE
            status = ctrl_regs[CTRL_AP_CTRL / 4]; // Re-read after clearing DONE
        }
        if (!(status & CTRL_AP_IDLE)) {
            was_running = 1;
            break;
        }
        usleep(100);
    }
    
    // If it never left IDLE, check if DONE is set (completed instantly)
    if (!was_running) {
        status = ctrl_regs[CTRL_AP_CTRL / 4];
        if ((status & CTRL_AP_DONE) || (status & CTRL_AP_READY)) {
            // Completed before we could see it running
            // Clear DONE by reading it
            status = ctrl_regs[CTRL_AP_CTRL / 4];
            YOLO2_LOG_DEBUG("    [DEBUG] Accelerator completed instantly (status=0x%02x)\n", status);
            return YOLO2_SUCCESS;
        }
    }
    
    // Now wait for IDLE to return (operation complete)
    // Also check for DONE bit (clear-on-read)
    while (1) {
        status = ctrl_regs[CTRL_AP_CTRL / 4];
        
        // Check for DONE bit (clear-on-read, so reading it clears it)
        if (status & CTRL_AP_DONE) {
            // Re-read to clear DONE, then check IDLE
            status = ctrl_regs[CTRL_AP_CTRL / 4];
            if (status & CTRL_AP_IDLE) {
                return YOLO2_SUCCESS;
            }
        }
        
        if (status & CTRL_AP_IDLE) {
            // Back to idle - operation complete
            return YOLO2_SUCCESS;
        }
        
        // Debug: Print status changes and periodic updates
        uint64_t elapsed = get_time_ms() - start_time;
        if (status != last_status) {
            if (status_change_count < 10) { // Limit debug output
                if (yolo2_get_verbosity() >= 3) {
                    printf("    [DEBUG] Status changed: 0x%02x -> 0x%02x (elapsed: %lu ms)",
                           last_status, status, elapsed);
                    if (status & CTRL_AP_START) printf(" START");
                    if (status & CTRL_AP_DONE) printf(" DONE");
                    if (status & CTRL_AP_IDLE) printf(" IDLE");
                    if (status & CTRL_AP_READY) printf(" READY");
                    printf("\n");
                }
            }
            last_status = status;
            status_change_count++;
        } else if (elapsed > 0 && (elapsed % 1000 == 0) && status_change_count < 15) {
            // Print periodic status every second
            if (yolo2_get_verbosity() >= 3) {
                printf("    [DEBUG] Still waiting... status=0x%02x (elapsed: %lu ms)", status, elapsed);
                if (status & CTRL_AP_START) printf(" START");
                if (status & CTRL_AP_DONE) printf(" DONE");
                if (status & CTRL_AP_IDLE) printf(" IDLE");
                if (status & CTRL_AP_READY) printf(" READY");
                printf("\n");
            }
            status_change_count++;
        }
        
        if (timeout_ms > 0 && elapsed > timeout_ms) {
            fprintf(stderr, "ERROR: Accelerator timeout after %u ms (status=0x%02x)\n", 
                    timeout_ms, status);
            fprintf(stderr, "       Status bits: START=%d DONE=%d IDLE=%d READY=%d\n",
                    (status & CTRL_AP_START) ? 1 : 0,
                    (status & CTRL_AP_DONE) ? 1 : 0,
                    (status & CTRL_AP_IDLE) ? 1 : 0,
                    (status & CTRL_AP_READY) ? 1 : 0);
            fprintf(stderr, "       This may indicate:\n");
            fprintf(stderr, "       - Accelerator is stuck in hardware\n");
            fprintf(stderr, "       - DMA transfer issue (buffers not flushed?)\n");
            fprintf(stderr, "       - Invalid layer parameters\n");
            fprintf(stderr, "       - Cache coherency issue\n");
            
            // Try to reset the accelerator by clearing START bit
            // Note: This may not work if hardware is truly stuck, but worth trying
            if (status & CTRL_AP_START) {
                fprintf(stderr, "       Attempting to clear START bit...\n");
                // Write 0 to clear START (though this may not work if hardware is stuck)
                ctrl_regs[CTRL_AP_CTRL / 4] = 0;
                __sync_synchronize();
                usleep(1000);
                status = ctrl_regs[CTRL_AP_CTRL / 4];
                fprintf(stderr, "       Status after clear attempt: 0x%02x\n", status);
            }
            
            return YOLO2_TIMEOUT;
        }
        
        usleep(1000); // 1ms delay for better visibility
    }
}

static int validate_conv_params(
    int ifm_num,
    int ofm_num,
    int ksize,
    int kstride,
    int input_w,
    int input_h,
    int output_w,
    int output_h,
    int padding,
    int tm,
    int tn,
    int tr,
    int tc)
{
    // Keep these checks aligned with the HLS `assert()` constraints in `hls/models/yolov2/yolo2_accel.cpp`.
    // When violated, the HLS core may never return to IDLE, so fail fast in software.
    if (ifm_num <= 0 || ifm_num > 2048) return 0;
    if (ofm_num <= 0 || ofm_num > 2048) return 0;
    if (ksize <= 0 || ksize > 3) return 0;
    if (kstride <= 0 || kstride > 2) return 0;
    if (input_w <= 0 || input_w > 1024) return 0;
    if (input_h <= 0 || input_h > 1024) return 0;
    if (output_w <= 0 || output_w > 1024) return 0;
    if (output_h <= 0 || output_h > 1024) return 0;
    if (padding < 0 || padding > 4) return 0;
    if (tm <= 0 || tm > Tm) return 0;
    if (tn < 0 || tn > Tn) return 0;
    if (tr <= 0 || tr > Tr) return 0;
    if (tc <= 0 || tc > Tc) return 0;
    return 1;
}

/**
 * Execute convolutional layer
 */
int yolo2_execute_conv_layer(
    uint64_t input_addr,
    uint64_t output_addr,
    uint64_t weight_addr,
    uint64_t beta_addr,
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
    int mloops_a1xTM,
    int layer_type,
    int qw,
    int qa_in,
    int qa_out,
    int qb,
    uint32_t timeout_ms
)
{
    if (!initialized || !ctrl_regs) {
        fprintf(stderr, "ERROR: Accelerator not initialized\n");
        return YOLO2_INIT_ERROR;
    }

    if (!validate_conv_params(ifm_num, ofm_num, ksize, kstride,
                              input_w, input_h, output_w, output_h, padding,
                              tm, tn, tr, tc)) {
        fprintf(stderr,
                "ERROR: Invalid layer parameters for HLS core: "
                "IFM=%d OFM=%d K=%d S=%d IN=%dx%d OUT=%dx%d PAD=%d "
                "TM=%d TN=%d TR=%d TC=%d (max TM=%d TN=%d TR=%d TC=%d)\n",
                ifm_num, ofm_num, ksize, kstride,
                input_w, input_h, output_w, output_h, padding,
                tm, tn, tr, tc, Tm, Tn, Tr, Tc);
        return YOLO2_ERROR;
    }
    
    // Set Q values first (INT16 mode)
    if (qw != 0 || qa_in != 0 || qa_out != 0 || qb != 0) {
        yolo2_set_q_values(qw, qa_in, qa_out, qb);
    }
    
    // Wait for accelerator to be idle before starting
    // Also clear any previous DONE/READY bits (clear-on-read)
    uint32_t status = ctrl_regs[CTRL_AP_CTRL / 4];
    if (status & CTRL_AP_DONE) {
        // Clear DONE by reading it again
        status = ctrl_regs[CTRL_AP_CTRL / 4];
    }
    if (status & CTRL_AP_READY) {
        // Clear READY by reading it again
        status = ctrl_regs[CTRL_AP_CTRL / 4];
    }
    
    if (!(status & CTRL_AP_IDLE)) {
        YOLO2_LOG_DEBUG("    [DEBUG] Waiting for IDLE before start (current status=0x%02x)...\n", status);
        if (wait_for_idle(1000) != YOLO2_SUCCESS) {
            fprintf(stderr, "ERROR: Accelerator not ready for new layer (status=0x%02x)\n", 
                    ctrl_regs[CTRL_AP_CTRL / 4]);
            return YOLO2_TIMEOUT;
        }
        // Clear any DONE/READY bits after waiting
        status = ctrl_regs[CTRL_AP_CTRL / 4];
        if (status & (CTRL_AP_DONE | CTRL_AP_READY)) {
            status = ctrl_regs[CTRL_AP_CTRL / 4]; // Clear by reading
        }
    }
    
    // Debug: Show what we're writing to the registers
    if (yolo2_get_verbosity() >= 3) {
        printf("    [DEBUG] Writing to control registers:\n");
        printf("      Input  @0x%02x: 0x%016lx\n", CTRL_INPUT_OFFSET, (unsigned long)input_addr);
        printf("      Output @0x%02x: 0x%016lx\n", CTRL_OUTPUT_OFFSET, (unsigned long)output_addr);
        printf("      Weight @0x%02x: 0x%016lx\n", CTRL_WEIGHT_OFFSET, (unsigned long)weight_addr);
        printf("      Beta   @0x%02x: 0x%016lx\n", CTRL_BETA_OFFSET, (unsigned long)beta_addr);
    }
    
    // Write 64-bit addresses (split into low/high 32-bit)
    ctrl_regs[CTRL_INPUT_OFFSET / 4] = (uint32_t)(input_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_INPUT_OFFSET / 4 + 1] = (uint32_t)(input_addr >> 32);
    ctrl_regs[CTRL_OUTPUT_OFFSET / 4] = (uint32_t)(output_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_OUTPUT_OFFSET / 4 + 1] = (uint32_t)(output_addr >> 32);
    ctrl_regs[CTRL_WEIGHT_OFFSET / 4] = (uint32_t)(weight_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_WEIGHT_OFFSET / 4 + 1] = (uint32_t)(weight_addr >> 32);
    ctrl_regs[CTRL_BETA_OFFSET / 4] = (uint32_t)(beta_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_BETA_OFFSET / 4 + 1] = (uint32_t)(beta_addr >> 32);
    
    // Verify by reading back
    __sync_synchronize();
    uint32_t input_lo = ctrl_regs[CTRL_INPUT_OFFSET / 4];
    uint32_t input_hi = ctrl_regs[CTRL_INPUT_OFFSET / 4 + 1];
    YOLO2_LOG_DEBUG("      Read back Input: 0x%08x%08x\n", input_hi, input_lo);
    
    // Write layer parameters
    ctrl_regs[CTRL_IFM_NUM_OFFSET / 4] = (uint32_t)ifm_num;
    ctrl_regs[CTRL_OFM_NUM_OFFSET / 4] = (uint32_t)ofm_num;
    ctrl_regs[CTRL_KSIZE_OFFSET / 4] = (uint32_t)ksize;
    ctrl_regs[CTRL_KSTRIDE_OFFSET / 4] = (uint32_t)kstride;
    ctrl_regs[CTRL_INPUT_W_OFFSET / 4] = (uint32_t)input_w;
    ctrl_regs[CTRL_INPUT_H_OFFSET / 4] = (uint32_t)input_h;
    ctrl_regs[CTRL_OUTPUT_W_OFFSET / 4] = (uint32_t)output_w;
    ctrl_regs[CTRL_OUTPUT_H_OFFSET / 4] = (uint32_t)output_h;
    ctrl_regs[CTRL_PADDING_OFFSET / 4] = (uint32_t)padding;
    ctrl_regs[CTRL_ISNL_OFFSET / 4] = (uint32_t)is_nl;
    ctrl_regs[CTRL_ISBN_OFFSET / 4] = (uint32_t)is_bn;
    ctrl_regs[CTRL_TM_OFFSET / 4] = (uint32_t)tm;
    ctrl_regs[CTRL_TN_OFFSET / 4] = (uint32_t)tn;
    ctrl_regs[CTRL_TR_OFFSET / 4] = (uint32_t)tr;
    ctrl_regs[CTRL_TC_OFFSET / 4] = (uint32_t)tc;
    ctrl_regs[CTRL_OFM_NUM_BOUND_OFFSET / 4] = (uint32_t)ofm_num_bound;
    ctrl_regs[CTRL_MLOOPSXTM_OFFSET / 4] = (uint32_t)mloopsxTM;
    ctrl_regs[CTRL_MLOOPS_A1XTM_OFFSET / 4] = (uint32_t)mloops_a1xTM;
    ctrl_regs[CTRL_LAYER_TYPE_OFFSET / 4] = (uint32_t)layer_type;
    
    // NOTE: Q values are set via AXI GPIO (yolo2_set_q_values), NOT control registers
    // The HLS IP does not have Q registers in CTRL_BUS
    
    // Memory barrier to ensure all writes complete
    __sync_synchronize();
    
    // CRITICAL: Flush cache for all DMA buffers before starting accelerator
    // The accelerator uses DMA to read input/weight/beta and write output
    // These buffers must be flushed to memory before accelerator can access them
    // Note: Even with O_SYNC mapping, explicit cache maintenance may be needed
    // for proper DMA coherency on ARM64 systems
    
    // Start accelerator
    ctrl_regs[CTRL_AP_CTRL / 4] = CTRL_AP_START;
    
    // Memory barrier after start (ensures START write is visible to accelerator)
    __sync_synchronize();
    
    // Small delay to allow accelerator to see the START signal
    usleep(10);
    
    // Verify accelerator actually started (status should show START bit)
    status = ctrl_regs[CTRL_AP_CTRL / 4];
    if (!(status & CTRL_AP_START)) {
        fprintf(stderr, "ERROR: Accelerator did not start (status=0x%02x)\n", status);
        return YOLO2_ERROR;
    }
    
    // Wait for completion using IDLE-based detection
    return wait_for_idle(timeout_ms);
}

/**
 * Execute maxpool layer
 */
int yolo2_execute_maxpool_layer(
    uint64_t input_addr,
    uint64_t output_addr,
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
    int mloops_a1xTM,
    uint32_t timeout_ms
)
{
    if (!initialized || !ctrl_regs) {
        fprintf(stderr, "ERROR: Accelerator not initialized\n");
        return YOLO2_INIT_ERROR;
    }
    
    // Wait for accelerator to be idle before starting
    uint32_t status = ctrl_regs[CTRL_AP_CTRL / 4];
    if (!(status & CTRL_AP_IDLE)) {
        if (wait_for_idle(1000) != YOLO2_SUCCESS) {
            return YOLO2_TIMEOUT;
        }
    }
    
    // Write addresses
    ctrl_regs[CTRL_INPUT_OFFSET / 4] = (uint32_t)(input_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_INPUT_OFFSET / 4 + 1] = (uint32_t)(input_addr >> 32);
    ctrl_regs[CTRL_OUTPUT_OFFSET / 4] = (uint32_t)(output_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_OUTPUT_OFFSET / 4 + 1] = (uint32_t)(output_addr >> 32);
    ctrl_regs[CTRL_WEIGHT_OFFSET / 4] = 0;  // Not used for maxpool
    ctrl_regs[CTRL_WEIGHT_OFFSET / 4 + 1] = 0;
    ctrl_regs[CTRL_BETA_OFFSET / 4] = 0;    // Not used for maxpool
    ctrl_regs[CTRL_BETA_OFFSET / 4 + 1] = 0;
    
    // Write layer parameters
    ctrl_regs[CTRL_IFM_NUM_OFFSET / 4] = (uint32_t)channels;
    ctrl_regs[CTRL_OFM_NUM_OFFSET / 4] = (uint32_t)channels;
    ctrl_regs[CTRL_KSIZE_OFFSET / 4] = (uint32_t)ksize;
    ctrl_regs[CTRL_KSTRIDE_OFFSET / 4] = (uint32_t)kstride;
    ctrl_regs[CTRL_INPUT_W_OFFSET / 4] = (uint32_t)input_w;
    ctrl_regs[CTRL_INPUT_H_OFFSET / 4] = (uint32_t)input_h;
    ctrl_regs[CTRL_OUTPUT_W_OFFSET / 4] = (uint32_t)output_w;
    ctrl_regs[CTRL_OUTPUT_H_OFFSET / 4] = (uint32_t)output_h;
    ctrl_regs[CTRL_PADDING_OFFSET / 4] = (uint32_t)padding;
    ctrl_regs[CTRL_ISNL_OFFSET / 4] = 0;
    ctrl_regs[CTRL_ISBN_OFFSET / 4] = 0;
    ctrl_regs[CTRL_TM_OFFSET / 4] = (uint32_t)tm;
    ctrl_regs[CTRL_TN_OFFSET / 4] = 0;
    ctrl_regs[CTRL_TR_OFFSET / 4] = (uint32_t)tr;
    ctrl_regs[CTRL_TC_OFFSET / 4] = (uint32_t)tc;
    ctrl_regs[CTRL_OFM_NUM_BOUND_OFFSET / 4] = (uint32_t)ofm_num_bound;
    ctrl_regs[CTRL_MLOOPSXTM_OFFSET / 4] = (uint32_t)mloopsxTM;
    ctrl_regs[CTRL_MLOOPS_A1XTM_OFFSET / 4] = (uint32_t)mloops_a1xTM;
    ctrl_regs[CTRL_LAYER_TYPE_OFFSET / 4] = 1;  // MAXPOOL = 1
    
    // Memory barrier
    __sync_synchronize();
    
    // Start accelerator
    ctrl_regs[CTRL_AP_CTRL / 4] = CTRL_AP_START;
    
    // Memory barrier after start
    __sync_synchronize();
    
    // Wait for completion using IDLE-based detection
    return wait_for_idle(timeout_ms);
}
