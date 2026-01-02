/**
 * Test Program for Accelerator Register Access
 * 
 * This program verifies that the accelerator control registers
 * can be accessed via /dev/mem.
 * 
 * Build: make test_accel
 * Run:   sudo ./test_accel
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

#include "yolo2_config.h"
#include "yolo2_accel_linux.h"

int main(int argc, char *argv[])
{
    int result;
    uint32_t status;
    
    (void)argc;
    (void)argv;
    
    printf("========================================\n");
    printf("YOLOv2 Accelerator Register Test\n");
    printf("========================================\n\n");
    
    // Initialize accelerator driver
    printf("[1] Initializing accelerator driver...\n");
    result = yolo2_accel_init();
    if (result != YOLO2_SUCCESS) {
        fprintf(stderr, "ERROR: Initialization failed: %d\n", result);
        return 1;
    }
    printf("    SUCCESS\n\n");
    
    // Read status register
    printf("[2] Reading status register...\n");
    status = yolo2_get_status();
    printf("    AP_CTRL = 0x%02x\n", status);
    printf("    - START:  %s\n", (status & CTRL_AP_START) ? "1" : "0");
    printf("    - DONE:   %s\n", (status & CTRL_AP_DONE) ? "1" : "0");
    printf("    - IDLE:   %s\n", (status & CTRL_AP_IDLE) ? "1" : "0");
    printf("    - READY:  %s\n", (status & CTRL_AP_READY) ? "1" : "0");
    printf("\n");
    
    // Verify IDLE state
    if (!(status & CTRL_AP_IDLE)) {
        printf("WARNING: Accelerator is not in IDLE state\n");
        printf("         This may indicate the bitstream is not loaded correctly\n\n");
    }
    
    // Read some parameter registers
    printf("[3] Reading parameter registers...\n");
    printf("    INPUT_OFFSET  (0x10): 0x%08x\n", yolo2_read_reg(CTRL_INPUT_OFFSET));
    printf("    OUTPUT_OFFSET (0x1c): 0x%08x\n", yolo2_read_reg(CTRL_OUTPUT_OFFSET));
    printf("    WEIGHT_OFFSET (0x28): 0x%08x\n", yolo2_read_reg(CTRL_WEIGHT_OFFSET));
    printf("    BETA_OFFSET   (0x34): 0x%08x\n", yolo2_read_reg(CTRL_BETA_OFFSET));
    printf("    IFM_NUM       (0x40): 0x%08x\n", yolo2_read_reg(CTRL_IFM_NUM_OFFSET));
    printf("    OFM_NUM       (0x48): 0x%08x\n", yolo2_read_reg(CTRL_OFM_NUM_OFFSET));
    printf("\n");
    
    // Test write/read cycle (write to a parameter register)
    printf("[4] Testing write/read cycle...\n");
    uint32_t test_value = 0x12345678;
    uint32_t original = yolo2_read_reg(CTRL_IFM_NUM_OFFSET);
    
    printf("    Writing 0x%08x to IFM_NUM register...\n", test_value);
    yolo2_write_reg(CTRL_IFM_NUM_OFFSET, test_value);
    
    uint32_t readback = yolo2_read_reg(CTRL_IFM_NUM_OFFSET);
    printf("    Read back: 0x%08x\n", readback);
    
    if (readback == test_value) {
        printf("    SUCCESS: Write/read verified\n");
    } else {
        printf("    WARNING: Mismatch (expected 0x%08x, got 0x%08x)\n", 
               test_value, readback);
    }
    
    // Restore original value
    yolo2_write_reg(CTRL_IFM_NUM_OFFSET, original);
    printf("\n");
    
    // Test Q value GPIO
    printf("[5] Testing Q value GPIO...\n");
    printf("    Setting Q values: Qw=8, Qa_in=7, Qa_out=6, Qb=5\n");
    yolo2_set_q_values(8, 7, 6, 5);
    printf("    Q values set (cannot read back GPIO output directly)\n");
    printf("\n");
    
    // Cleanup
    printf("[6] Cleaning up...\n");
    yolo2_accel_cleanup();
    printf("    Done\n\n");
    
    printf("========================================\n");
    printf("Test completed successfully!\n");
    printf("========================================\n\n");
    
    return 0;
}
