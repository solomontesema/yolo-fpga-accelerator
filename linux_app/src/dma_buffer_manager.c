/**
 * DMA Buffer Manager - Linux Implementation
 * 
 * Uses udmabuf kernel module for physically contiguous DMA buffers.
 * 
 * udmabuf provides device files like /dev/udmabuf0, /dev/udmabuf1, etc.
 * Each device can be mmapped to get a user-space pointer.
 * Physical address is obtained via sysfs.
 * 
 * Alternative: CMA (Contiguous Memory Allocator) via /dev/dma_heap
 */

#include "dma_buffer_manager.h"
#include "yolo2_config.h"
#include "yolo2_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <dirent.h>
#include <errno.h>

// Maximum number of tracked DMA buffers
#define MAX_DMA_BUFFERS 16

// Buffer tracking for physical address lookup
static struct {
    dma_buffer_t buffers[MAX_DMA_BUFFERS];
    int count;
    int initialized;
} dma_ctx = {0};

/**
 * Get physical address from udmabuf sysfs
 */
static uint64_t get_udmabuf_phys_addr(const char *device_name)
{
    char sysfs_path[256];
    char buf[64];
    FILE *fp;
    uint64_t phys_addr = 0;
    
    // Read physical address from sysfs
    snprintf(sysfs_path, sizeof(sysfs_path), 
             "/sys/class/u-dma-buf/%s/phys_addr", device_name);
    
    fp = fopen(sysfs_path, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot read %s: %s\n", sysfs_path, strerror(errno));
        return 0;
    }
    
    if (fgets(buf, sizeof(buf), fp)) {
        phys_addr = strtoull(buf, NULL, 16);
    }
    fclose(fp);
    
    return phys_addr;
}

/**
 * Get buffer size from udmabuf sysfs
 */
static size_t get_udmabuf_size(const char *device_name)
{
    char sysfs_path[256];
    char buf[64];
    FILE *fp;
    size_t size = 0;
    
    snprintf(sysfs_path, sizeof(sysfs_path), 
             "/sys/class/u-dma-buf/%s/size", device_name);
    
    fp = fopen(sysfs_path, "r");
    if (!fp) {
        return 0;
    }
    
    if (fgets(buf, sizeof(buf), fp)) {
        size = strtoul(buf, NULL, 10);
    }
    fclose(fp);
    
    return size;
}

/**
 * Find available udmabuf device with sufficient size
 */
static int find_udmabuf_device(size_t required_size, char *device_name, size_t name_len)
{
    DIR *dir;
    struct dirent *entry;
    
    dir = opendir("/sys/class/u-dma-buf");
    if (!dir) {
        fprintf(stderr, "ERROR: udmabuf not available (no /sys/class/u-dma-buf)\n");
        fprintf(stderr, "       Install udmabuf: https://github.com/ikwzm/udmabuf\n");
        return -1;
    }
    
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        if (strncmp(entry->d_name, "udmabuf", 7) != 0) continue;
        
        size_t buf_size = get_udmabuf_size(entry->d_name);
        if (buf_size >= required_size) {
            // Check if not already in use
            int in_use = 0;
            for (int i = 0; i < dma_ctx.count; i++) {
                if (strcmp(dma_ctx.buffers[i].device_name, entry->d_name) == 0) {
                    in_use = 1;
                    break;
                }
            }
            
            if (!in_use) {
                snprintf(device_name, name_len, "%s", entry->d_name);
                closedir(dir);
                return 0;
            }
        }
    }
    
    closedir(dir);
    fprintf(stderr, "ERROR: No udmabuf device with %zu bytes available\n", required_size);
    return -1;
}

/**
 * Initialize DMA buffer manager
 */
int dma_buffer_init(void)
{
    if (dma_ctx.initialized) {
        return 0;
    }
    
    YOLO2_LOG_INFO("Initializing DMA buffer manager...\n");
    
    // Check if udmabuf is available
    DIR *dir = opendir("/sys/class/u-dma-buf");
    if (!dir) {
        fprintf(stderr, "ERROR: udmabuf kernel module not loaded\n");
        fprintf(stderr, "       Load module: sudo modprobe u-dma-buf\n");
        fprintf(stderr, "       Or install: https://github.com/ikwzm/udmabuf\n");
        return -1;
    }
    
    // Count available devices
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "udmabuf", 7) == 0) {
            size_t size = get_udmabuf_size(entry->d_name);
            uint64_t phys = get_udmabuf_phys_addr(entry->d_name);
            YOLO2_LOG_DEBUG("  Found %s: size=%zu bytes, phys=0x%lx\n",
                            entry->d_name, size, (unsigned long)phys);
            count++;
        }
    }
    closedir(dir);
    
    if (count == 0) {
        fprintf(stderr, "ERROR: No udmabuf devices found\n");
        fprintf(stderr, "       Create devices via device tree or module parameter\n");
        return -1;
    }
    
    memset(&dma_ctx, 0, sizeof(dma_ctx));
    dma_ctx.initialized = 1;
    
    YOLO2_LOG_INFO("  DMA buffer manager initialized (%d devices available)\n", count);
    return 0;
}

/**
 * Cleanup DMA buffer manager
 */
void dma_buffer_cleanup(void)
{
    for (int i = 0; i < dma_ctx.count; i++) {
        if (dma_ctx.buffers[i].virt_addr) {
            dma_buffer_free(&dma_ctx.buffers[i]);
        }
    }
    memset(&dma_ctx, 0, sizeof(dma_ctx));
}

/**
 * Set udmabuf sync_mode via sysfs
 */
static void set_udmabuf_sync_mode(const char *device_name, int mode)
{
    char sysfs_path[256];
    FILE *fp;
    
    snprintf(sysfs_path, sizeof(sysfs_path), 
             "/sys/class/u-dma-buf/%s/sync_mode", device_name);
    
    fp = fopen(sysfs_path, "w");
    if (fp) {
        fprintf(fp, "%d", mode);
        fclose(fp);
    }
}

/**
 * Allocate DMA buffer
 */
int dma_buffer_alloc(size_t size, dma_buffer_t *buffer)
{
    char device_name[64];
    char device_path[128];
    int fd;
    void *mapped;
    
    if (!dma_ctx.initialized) {
        if (dma_buffer_init() != 0) {
            return -1;
        }
    }
    
    if (dma_ctx.count >= MAX_DMA_BUFFERS) {
        fprintf(stderr, "ERROR: Maximum DMA buffers exceeded\n");
        return -1;
    }
    
    // Align size to page boundary
    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);
    
    // Find available udmabuf device
    if (find_udmabuf_device(aligned_size, device_name, sizeof(device_name)) != 0) {
        return -1;
    }
    
    // Set sync_mode to 1 (for proper DMA coherency on ARM64)
    set_udmabuf_sync_mode(device_name, 1);
    
    // Open device with O_SYNC for uncached access
    snprintf(device_path, sizeof(device_path), "/dev/%s", device_name);
    fd = open(device_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Cannot open %s: %s\n", device_path, strerror(errno));
        return -1;
    }
    
    // Get physical address
    uint64_t phys_addr = get_udmabuf_phys_addr(device_name);
    if (phys_addr == 0) {
        fprintf(stderr, "ERROR: Cannot get physical address for %s\n", device_name);
        close(fd);
        return -1;
    }
    
    // Map to user space
    mapped = mmap(NULL, aligned_size, 
                  PROT_READ | PROT_WRITE, 
                  MAP_SHARED, 
                  fd, 0);
    
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed for %s: %s\n", device_path, strerror(errno));
        close(fd);
        return -1;
    }
    
    // Gentle zero-init: only first page to verify access works
    // Full zero-init can be slow and cause issues with large uncached buffers
    volatile uint32_t *test_ptr = (volatile uint32_t*)mapped;
    test_ptr[0] = 0;  // Test write
    (void)test_ptr[0]; // Test read
    
    // Fill buffer structure
    buffer->virt_addr = mapped;
    buffer->phys_addr = phys_addr;
    buffer->size = aligned_size;
    buffer->fd = fd;
    strncpy(buffer->device_name, device_name, sizeof(buffer->device_name) - 1);
    
    // Track buffer
    memcpy(&dma_ctx.buffers[dma_ctx.count], buffer, sizeof(dma_buffer_t));
    dma_ctx.count++;
    
    YOLO2_LOG_DEBUG("  Allocated DMA buffer: %s, size=%zu, phys=0x%lx, virt=%p\n",
                    device_name, aligned_size, (unsigned long)phys_addr, mapped);
    
    return 0;
}

/**
 * Free DMA buffer
 */
void dma_buffer_free(dma_buffer_t *buffer)
{
    if (!buffer || !buffer->virt_addr) {
        return;
    }
    
    munmap(buffer->virt_addr, buffer->size);
    close(buffer->fd);
    
    // Remove from tracking
    for (int i = 0; i < dma_ctx.count; i++) {
        if (dma_ctx.buffers[i].virt_addr == buffer->virt_addr) {
            // Shift remaining entries
            for (int j = i; j < dma_ctx.count - 1; j++) {
                dma_ctx.buffers[j] = dma_ctx.buffers[j + 1];
            }
            dma_ctx.count--;
            break;
        }
    }
    
    memset(buffer, 0, sizeof(dma_buffer_t));
}

/**
 * Sync buffer for device
 * With O_SYNC mapping, this is mostly a no-op but we add a memory barrier
 */
void dma_buffer_sync_for_device(dma_buffer_t *buffer, size_t offset, size_t size)
{
    (void)buffer;
    (void)offset;
    (void)size;
    __sync_synchronize();
}

/**
 * Sync buffer for CPU
 */
void dma_buffer_sync_for_cpu(dma_buffer_t *buffer, size_t offset, size_t size)
{
    (void)buffer;
    (void)offset;
    (void)size;
    __sync_synchronize();
}

/**
 * Get physical address at offset
 */
uint64_t dma_buffer_get_phys(dma_buffer_t *buffer, size_t offset)
{
    if (!buffer) return 0;
    return buffer->phys_addr + offset;
}

/*===========================================================================
 * Compatibility layer for memory_manager.h interface
 *===========================================================================*/

// Global tracking for memory_buffer_t allocations
static struct {
    memory_buffer_t buffers[MAX_DMA_BUFFERS];
    dma_buffer_t dma_buffers[MAX_DMA_BUFFERS];
    int count;
} mem_ctx = {0};

/**
 * Allocate memory buffer (wrapper)
 */
int memory_allocate_ddr(size_t size, size_t alignment, memory_buffer_t *buffer)
{
    (void)alignment;  // udmabuf handles alignment
    
    if (mem_ctx.count >= MAX_DMA_BUFFERS) {
        fprintf(stderr, "ERROR: Maximum memory buffers exceeded\n");
        return -1;
    }
    
    dma_buffer_t dma_buf;
    if (dma_buffer_alloc(size, &dma_buf) != 0) {
        return -1;
    }
    
    buffer->ptr = dma_buf.virt_addr;
    buffer->size = dma_buf.size;
    buffer->phys_addr = dma_buf.phys_addr;
    
    // Track for cleanup
    mem_ctx.buffers[mem_ctx.count] = *buffer;
    mem_ctx.dma_buffers[mem_ctx.count] = dma_buf;
    mem_ctx.count++;
    
    return 0;
}

/**
 * Free memory buffer
 */
void memory_free_ddr(memory_buffer_t *buffer)
{
    if (!buffer || !buffer->ptr) return;
    
    for (int i = 0; i < mem_ctx.count; i++) {
        if (mem_ctx.buffers[i].ptr == buffer->ptr) {
            dma_buffer_free(&mem_ctx.dma_buffers[i]);
            
            // Shift remaining
            for (int j = i; j < mem_ctx.count - 1; j++) {
                mem_ctx.buffers[j] = mem_ctx.buffers[j + 1];
                mem_ctx.dma_buffers[j] = mem_ctx.dma_buffers[j + 1];
            }
            mem_ctx.count--;
            break;
        }
    }
    
    memset(buffer, 0, sizeof(memory_buffer_t));
}

/**
 * Allocate weights buffer
 */
int memory_allocate_weights(size_t size, memory_buffer_t *buffer)
{
    return memory_allocate_ddr(size, MEMORY_ALIGNMENT, buffer);
}

/**
 * Allocate bias buffer
 */
int memory_allocate_bias(size_t size, memory_buffer_t *buffer)
{
    return memory_allocate_ddr(size, MEMORY_ALIGNMENT, buffer);
}

/**
 * Allocate inference buffer
 */
int memory_allocate_inference_buffer(memory_buffer_t *buffer)
{
    // mem_len + 512*2 for overflow protection, INT16 = 2 bytes per element
    size_t mem_size = (MEM_LEN + 512 * 2) * sizeof(int16_t);
    return memory_allocate_ddr(mem_size, MEMORY_ALIGNMENT, buffer);
}

/**
 * Get physical address
 */
uint64_t memory_get_phys_addr(void *virt_addr)
{
    // Search tracked buffers
    for (int i = 0; i < mem_ctx.count; i++) {
        void *start = mem_ctx.buffers[i].ptr;
        void *end = (char*)start + mem_ctx.buffers[i].size;
        
        if (virt_addr >= start && virt_addr < end) {
            size_t offset = (char*)virt_addr - (char*)start;
            return mem_ctx.buffers[i].phys_addr + offset;
        }
    }
    
    // Not found - print debug info
    fprintf(stderr, "WARNING: memory_get_phys_addr: address %p not in tracked buffers\n", virt_addr);
    fprintf(stderr, "  Tracked buffers (%d):\n", mem_ctx.count);
    for (int i = 0; i < mem_ctx.count; i++) {
        void *start = mem_ctx.buffers[i].ptr;
        void *end = (char*)start + mem_ctx.buffers[i].size;
        fprintf(stderr, "    [%d] %p - %p (size=%zu)\n", i, start, end, mem_ctx.buffers[i].size);
    }
    return (uint64_t)(uintptr_t)virt_addr;
}

/**
 * Flush cache (no-op for uncached DMA buffers)
 */
void memory_flush_cache(void *addr, size_t size)
{
    (void)addr;
    (void)size;
    __sync_synchronize();
}

/**
 * Invalidate cache (no-op for uncached DMA buffers)
 */
void memory_invalidate_cache(void *addr, size_t size)
{
    (void)addr;
    (void)size;
    __sync_synchronize();
}
