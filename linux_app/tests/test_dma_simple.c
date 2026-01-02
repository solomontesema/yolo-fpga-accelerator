/**
 * Simple DMA Buffer Test - Debug version
 * 
 * Minimal test to debug udmabuf access issues
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

int main(void)
{
    int fd;
    void *mapped;
    size_t size = 4096;  // Start with just 1 page
    
    printf("Simple udmabuf test\n");
    printf("===================\n\n");
    
    // Try opening udmabuf1 (smallest, 1MB)
    printf("[1] Opening /dev/udmabuf1...\n");
    fd = open("/dev/udmabuf1", O_RDWR);
    if (fd < 0) {
        printf("    Failed with O_RDWR: %s\n", strerror(errno));
        
        // Try with O_SYNC
        fd = open("/dev/udmabuf1", O_RDWR | O_SYNC);
        if (fd < 0) {
            printf("    Failed with O_RDWR|O_SYNC: %s\n", strerror(errno));
            return 1;
        }
        printf("    Opened with O_SYNC\n");
    } else {
        printf("    Opened OK (fd=%d)\n", fd);
    }
    
    // Try mmap with different flag combinations
    printf("\n[2] Trying mmap (size=%zu)...\n", size);
    
    // Method 1: MAP_SHARED
    printf("    Method 1: MAP_SHARED... ");
    mapped = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        printf("FAILED: %s\n", strerror(errno));
    } else {
        printf("OK at %p\n", mapped);
        
        // Try to access it
        printf("    Trying read... ");
        fflush(stdout);
        volatile uint32_t val = *(volatile uint32_t*)mapped;
        printf("OK (val=0x%08x)\n", val);
        
        printf("    Trying write... ");
        fflush(stdout);
        *(volatile uint32_t*)mapped = 0x12345678;
        printf("OK\n");
        
        printf("    Trying readback... ");
        fflush(stdout);
        val = *(volatile uint32_t*)mapped;
        printf("OK (val=0x%08x)\n", val);
        
        if (val == 0x12345678) {
            printf("    SUCCESS: Write/read verified!\n");
        } else {
            printf("    WARNING: Value mismatch\n");
        }
        
        munmap(mapped, size);
    }
    
    close(fd);
    
    // Also try with O_SYNC from the start
    printf("\n[3] Testing with O_SYNC from start...\n");
    fd = open("/dev/udmabuf1", O_RDWR | O_SYNC);
    if (fd < 0) {
        printf("    Open failed: %s\n", strerror(errno));
        return 1;
    }
    
    mapped = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        printf("    mmap failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    
    printf("    Mapped at %p\n", mapped);
    printf("    Writing test pattern... ");
    fflush(stdout);
    
    volatile uint32_t *ptr = (volatile uint32_t*)mapped;
    for (int i = 0; i < 16; i++) {
        ptr[i] = 0xCAFE0000 | i;
    }
    printf("OK\n");
    
    printf("    Reading back... ");
    fflush(stdout);
    int errors = 0;
    for (int i = 0; i < 16; i++) {
        if (ptr[i] != (0xCAFE0000 | i)) {
            errors++;
        }
    }
    printf("OK (%d errors)\n", errors);
    
    munmap(mapped, size);
    close(fd);
    
    printf("\n===================\n");
    printf("Test complete!\n");
    
    return 0;
}
