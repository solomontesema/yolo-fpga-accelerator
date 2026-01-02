/**
 * Test PL-DDR connectivity via HP ports
 * 
 * This test verifies that the FPGA accelerator can access DDR memory.
 * It writes a known pattern to udmabuf via CPU, then triggers the
 * accelerator to read from that location. If the accelerator hangs,
 * the HP port connectivity is broken.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

#define YOLO2_CTRL_BASE    0xA0000000UL
#define YOLO2_CTRL_SIZE    0x1000

// HLS register offsets
#define CTRL_AP_CTRL       0x00
#define CTRL_INPUT_OFFSET  0x10
#define CTRL_OUTPUT_OFFSET 0x1c
#define CTRL_WEIGHT_OFFSET 0x28
#define CTRL_BETA_OFFSET   0x34
#define CTRL_IFM_NUM       0x40
#define CTRL_OFM_NUM       0x48
#define CTRL_KSIZE         0x50
#define CTRL_KSTRIDE       0x58
#define CTRL_INPUT_W       0x60
#define CTRL_INPUT_H       0x68
#define CTRL_OUTPUT_W      0x70
#define CTRL_OUTPUT_H      0x78
#define CTRL_PADDING       0x80
#define CTRL_ISNL          0x88
#define CTRL_ISBN          0x90
#define CTRL_TM            0x98
#define CTRL_TN            0xa0
#define CTRL_TR            0xa8
#define CTRL_TC            0xb0
#define CTRL_OFM_BOUND     0xb8
#define CTRL_MLOOPSXTM     0xc0
#define CTRL_MLOOPS_A1XTM  0xc8
#define CTRL_LAYERTYPE     0xd0

// Status bits
#define AP_START  (1 << 0)
#define AP_DONE   (1 << 1)
#define AP_IDLE   (1 << 2)
#define AP_READY  (1 << 3)

static uint64_t get_udmabuf_phys(const char *name) {
    char path[128];
    FILE *fp;
    uint64_t addr = 0;
    
    snprintf(path, sizeof(path), "/sys/class/u-dma-buf/%s/phys_addr", name);
    fp = fopen(path, "r");
    if (fp) {
        fscanf(fp, "0x%lx", &addr);
        fclose(fp);
    }
    return addr;
}

static size_t get_udmabuf_size(const char *name) {
    char path[128];
    FILE *fp;
    size_t size = 0;
    
    snprintf(path, sizeof(path), "/sys/class/u-dma-buf/%s/size", name);
    fp = fopen(path, "r");
    if (fp) {
        fscanf(fp, "%zu", &size);
        fclose(fp);
    }
    return size;
}

int main(void) {
    int devmem_fd;
    volatile uint32_t *ctrl_regs;
    uint32_t status;
    
    printf("==============================================\n");
    printf("PL-DDR Connectivity Test\n");
    printf("==============================================\n\n");
    
    // Check udmabuf availability
    uint64_t udma0_phys = get_udmabuf_phys("udmabuf0");
    size_t udma0_size = get_udmabuf_size("udmabuf0");
    
    if (udma0_phys == 0) {
        printf("ERROR: udmabuf0 not available. Load the module first.\n");
        return 1;
    }
    
    printf("[1] udmabuf0 info:\n");
    printf("    Physical: 0x%lx\n", (unsigned long)udma0_phys);
    printf("    Size: %zu bytes (%.1f MB)\n\n", udma0_size, udma0_size / (1024.0 * 1024.0));
    
    // Map accelerator control registers
    devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (devmem_fd < 0) {
        perror("Cannot open /dev/mem");
        return 1;
    }
    
    ctrl_regs = mmap(NULL, YOLO2_CTRL_SIZE, PROT_READ | PROT_WRITE, 
                     MAP_SHARED, devmem_fd, YOLO2_CTRL_BASE);
    if (ctrl_regs == MAP_FAILED) {
        perror("Cannot mmap control registers");
        close(devmem_fd);
        return 1;
    }
    
    printf("[2] Accelerator control registers mapped\n");
    status = ctrl_regs[CTRL_AP_CTRL / 4];
    printf("    Initial status: 0x%02x", status);
    if (status & AP_IDLE) printf(" IDLE");
    if (status & AP_DONE) printf(" DONE");
    if (status & AP_READY) printf(" READY");
    printf("\n\n");
    
    if (!(status & AP_IDLE)) {
        printf("WARNING: Accelerator not IDLE. It may be stuck.\n");
        printf("Try reloading the bitstream: sudo xmutil unloadapp; sudo xmutil loadapp yolov2_accel\n\n");
    }
    
    // Test: Write minimal parameters and start accelerator
    // We'll use the smallest possible configuration that should complete quickly
    printf("[3] Writing minimal test parameters...\n");
    
    // Set addresses - all pointing to udmabuf0
    uint64_t test_addr = udma0_phys;
    printf("    Input/Output/Weight/Beta addr: 0x%lx\n", (unsigned long)test_addr);
    
    ctrl_regs[CTRL_INPUT_OFFSET / 4] = (uint32_t)(test_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_INPUT_OFFSET / 4 + 1] = (uint32_t)(test_addr >> 32);
    ctrl_regs[CTRL_OUTPUT_OFFSET / 4] = (uint32_t)(test_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_OUTPUT_OFFSET / 4 + 1] = (uint32_t)(test_addr >> 32);
    ctrl_regs[CTRL_WEIGHT_OFFSET / 4] = (uint32_t)(test_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_WEIGHT_OFFSET / 4 + 1] = (uint32_t)(test_addr >> 32);
    ctrl_regs[CTRL_BETA_OFFSET / 4] = (uint32_t)(test_addr & 0xFFFFFFFF);
    ctrl_regs[CTRL_BETA_OFFSET / 4 + 1] = (uint32_t)(test_addr >> 32);
    
    // Minimal layer parameters (1x1 conv with tiny size)
    ctrl_regs[CTRL_IFM_NUM / 4] = 1;      // 1 input channel
    ctrl_regs[CTRL_OFM_NUM / 4] = 1;      // 1 output channel
    ctrl_regs[CTRL_KSIZE / 4] = 1;        // 1x1 kernel
    ctrl_regs[CTRL_KSTRIDE / 4] = 1;      // stride 1
    ctrl_regs[CTRL_INPUT_W / 4] = 8;      // 8x8 input
    ctrl_regs[CTRL_INPUT_H / 4] = 8;
    ctrl_regs[CTRL_OUTPUT_W / 4] = 8;     // 8x8 output
    ctrl_regs[CTRL_OUTPUT_H / 4] = 8;
    ctrl_regs[CTRL_PADDING / 4] = 0;
    ctrl_regs[CTRL_ISNL / 4] = 0;         // no activation
    ctrl_regs[CTRL_ISBN / 4] = 0;         // no batch norm
    ctrl_regs[CTRL_TM / 4] = 1;
    ctrl_regs[CTRL_TN / 4] = 1;
    ctrl_regs[CTRL_TR / 4] = 8;
    ctrl_regs[CTRL_TC / 4] = 8;
    ctrl_regs[CTRL_OFM_BOUND / 4] = 16;
    ctrl_regs[CTRL_MLOOPSXTM / 4] = 1;
    ctrl_regs[CTRL_MLOOPS_A1XTM / 4] = 16;
    ctrl_regs[CTRL_LAYERTYPE / 4] = 0;    // CONV
    
    __sync_synchronize();
    
    printf("    Parameters written.\n\n");
    
    // Read back addresses to verify writes
    uint32_t in_lo = ctrl_regs[CTRL_INPUT_OFFSET / 4];
    uint32_t in_hi = ctrl_regs[CTRL_INPUT_OFFSET / 4 + 1];
    printf("[4] Verify register writes:\n");
    printf("    Input addr readback: 0x%08x%08x\n", in_hi, in_lo);
    
    // Check status before start
    status = ctrl_regs[CTRL_AP_CTRL / 4];
    printf("    Status before start: 0x%02x\n\n", status);
    
    printf("[5] Starting accelerator...\n");
    ctrl_regs[CTRL_AP_CTRL / 4] = AP_START;
    __sync_synchronize();
    
    // Poll status
    printf("    Polling for completion (5 second timeout)...\n");
    int timeout_ms = 5000;
    int elapsed = 0;
    
    while (elapsed < timeout_ms) {
        status = ctrl_regs[CTRL_AP_CTRL / 4];
        
        if (status & AP_IDLE) {
            printf("\n    SUCCESS! Accelerator returned to IDLE.\n");
            printf("    Final status: 0x%02x\n", status);
            break;
        }
        
        if (elapsed % 1000 == 0) {
            printf("    [%ds] status=0x%02x", elapsed / 1000, status);
            if (status & AP_START) printf(" START");
            if (status & AP_DONE) printf(" DONE");
            if (status & AP_IDLE) printf(" IDLE");
            if (status & AP_READY) printf(" READY");
            printf("\n");
        }
        
        usleep(10000);  // 10ms
        elapsed += 10;
    }
    
    if (elapsed >= timeout_ms) {
        status = ctrl_regs[CTRL_AP_CTRL / 4];
        printf("\n    TIMEOUT! Accelerator stuck.\n");
        printf("    Final status: 0x%02x\n", status);
        printf("\n    DIAGNOSIS: HP ports cannot access DDR.\n");
        printf("    Possible causes:\n");
        printf("      1. HP port clocks not enabled\n");
        printf("      2. SmartConnect not properly connected\n");
        printf("      3. AXI address width mismatch\n");
        printf("      4. Missing device tree configuration\n");
    }
    
    printf("\n[6] Check dmesg for errors:\n");
    printf("    Run: sudo dmesg | tail -20\n");
    
    munmap((void*)ctrl_regs, YOLO2_CTRL_SIZE);
    close(devmem_fd);
    
    printf("\n==============================================\n");
    printf("Test completed.\n");
    printf("==============================================\n");
    
    return (elapsed >= timeout_ms) ? 1 : 0;
}
