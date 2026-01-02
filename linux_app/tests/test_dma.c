/**
 * Test Program for DMA Buffer Allocation
 * 
 * This program verifies that DMA buffers can be allocated
 * using udmabuf and physical addresses obtained.
 * 
 * Build: make test_dma
 * Run:   sudo ./test_dma
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "yolo2_config.h"
#include "dma_buffer_manager.h"

int main(int argc, char *argv[])
{
    int result;
    
    (void)argc;
    (void)argv;
    
    printf("========================================\n");
    printf("DMA Buffer Allocation Test\n");
    printf("========================================\n\n");
    
    // Initialize DMA buffer manager
    printf("[1] Initializing DMA buffer manager...\n");
    result = dma_buffer_init();
    if (result != 0) {
        fprintf(stderr, "ERROR: DMA buffer initialization failed\n");
        fprintf(stderr, "       Is udmabuf kernel module loaded?\n");
        fprintf(stderr, "       Try: sudo modprobe u-dma-buf\n");
        return 1;
    }
    printf("    SUCCESS\n\n");
    
    // Test small allocation
    printf("[2] Testing small buffer allocation (1MB)...\n");
    dma_buffer_t small_buf;
    result = dma_buffer_alloc(1024 * 1024, &small_buf);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to allocate small buffer\n");
        goto cleanup;
    }
    printf("    Virtual address:  %p\n", small_buf.virt_addr);
    printf("    Physical address: 0x%lx\n", (unsigned long)small_buf.phys_addr);
    printf("    Size:             %zu bytes\n", small_buf.size);
    printf("    Device:           %s\n", small_buf.device_name);
    printf("\n");
    
    // Test write/read
    printf("[3] Testing write/read cycle...\n");
    uint32_t *ptr = (uint32_t*)small_buf.virt_addr;
    printf("    Writing test pattern...\n");
    for (int i = 0; i < 256; i++) {
        ptr[i] = 0xDEAD0000 | i;
    }
    
    // Sync for device
    dma_buffer_sync_for_device(&small_buf, 0, 0);
    
    // Sync for CPU
    dma_buffer_sync_for_cpu(&small_buf, 0, 0);
    
    printf("    Verifying...\n");
    int errors = 0;
    for (int i = 0; i < 256; i++) {
        if (ptr[i] != (uint32_t)(0xDEAD0000 | i)) {
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("    SUCCESS: All 256 values verified\n");
    } else {
        printf("    ERROR: %d mismatches found\n", errors);
    }
    printf("\n");
    
    // Free small buffer
    printf("[4] Freeing small buffer...\n");
    dma_buffer_free(&small_buf);
    printf("    Done\n\n");
    
    // Test memory_buffer_t interface (compatibility layer)
    printf("[5] Testing memory_buffer_t interface...\n");
    memory_buffer_t mem_buf;
    result = memory_allocate_ddr(2 * 1024 * 1024, MEMORY_ALIGNMENT, &mem_buf);
    if (result != 0) {
        fprintf(stderr, "ERROR: memory_allocate_ddr failed\n");
        goto cleanup;
    }
    printf("    Pointer:          %p\n", mem_buf.ptr);
    printf("    Physical address: 0x%lx\n", (unsigned long)mem_buf.phys_addr);
    printf("    Size:             %zu bytes\n", mem_buf.size);
    printf("\n");
    
    // Test memory_get_phys_addr
    printf("[6] Testing memory_get_phys_addr...\n");
    void *offset_ptr = (char*)mem_buf.ptr + 4096;
    uint64_t offset_phys = memory_get_phys_addr(offset_ptr);
    printf("    Base physical:    0x%lx\n", (unsigned long)mem_buf.phys_addr);
    printf("    Offset +4096:     0x%lx\n", (unsigned long)offset_phys);
    printf("    Expected:         0x%lx\n", (unsigned long)(mem_buf.phys_addr + 4096));
    
    if (offset_phys == mem_buf.phys_addr + 4096) {
        printf("    SUCCESS: Physical address calculation correct\n");
    } else {
        printf("    ERROR: Physical address mismatch\n");
    }
    printf("\n");
    
    // Free memory buffer
    printf("[7] Freeing memory buffer...\n");
    memory_free_ddr(&mem_buf);
    printf("    Done\n\n");
    
    // Test inference buffer allocation
    printf("[8] Testing inference buffer allocation...\n");
    printf("    Required size: %zu bytes (%.1f MB)\n", 
           (MEM_LEN + 512 * 2) * sizeof(int16_t),
           (MEM_LEN + 512 * 2) * sizeof(int16_t) / (1024.0 * 1024.0));
    
    memory_buffer_t inf_buf;
    result = memory_allocate_inference_buffer(&inf_buf);
    if (result != 0) {
        fprintf(stderr, "WARNING: Inference buffer allocation failed\n");
        fprintf(stderr, "         May need larger udmabuf device\n");
    } else {
        printf("    SUCCESS: Allocated at phys 0x%lx\n", (unsigned long)inf_buf.phys_addr);
        memory_free_ddr(&inf_buf);
    }
    printf("\n");
    
cleanup:
    // Cleanup
    printf("[9] Cleaning up...\n");
    dma_buffer_cleanup();
    printf("    Done\n\n");
    
    printf("========================================\n");
    printf("Test completed!\n");
    printf("========================================\n\n");
    
    return 0;
}
