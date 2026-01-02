/**
 * DMA Buffer Manager - Linux Implementation
 * 
 * Allocates physically contiguous DMA buffers using udmabuf kernel module.
 * The accelerator requires physical addresses for DMA operations.
 * 
 * udmabuf provides:
 * - Physically contiguous memory allocation
 * - Physical address retrieval
 * - Uncached/write-combined mappings for DMA coherency
 */

#ifndef DMA_BUFFER_MANAGER_H
#define DMA_BUFFER_MANAGER_H

#include <stdint.h>
#include <stddef.h>

/**
 * DMA buffer structure
 * Contains both virtual and physical addresses
 */
typedef struct {
    void *virt_addr;        // Virtual address (for CPU access)
    uint64_t phys_addr;     // Physical address (for accelerator DMA)
    size_t size;            // Buffer size in bytes
    int fd;                 // File descriptor for udmabuf device
    char device_name[64];   // Device name (e.g., "udmabuf0")
} dma_buffer_t;

/**
 * Initialize DMA buffer manager
 * Checks for udmabuf availability
 * 
 * Returns: 0 on success, -1 on error
 */
int dma_buffer_init(void);

/**
 * Cleanup DMA buffer manager
 */
void dma_buffer_cleanup(void);

/**
 * Allocate DMA buffer using udmabuf
 * 
 * size: Requested size in bytes
 * buffer: Output buffer structure (filled on success)
 * 
 * Returns: 0 on success, -1 on error
 */
int dma_buffer_alloc(size_t size, dma_buffer_t *buffer);

/**
 * Free DMA buffer
 * 
 * buffer: Buffer to free
 */
void dma_buffer_free(dma_buffer_t *buffer);

/**
 * Sync buffer for DMA (CPU -> Device)
 * Call before accelerator reads from buffer
 * 
 * buffer: Buffer to sync
 * offset: Offset within buffer
 * size: Size to sync (0 = entire buffer)
 */
void dma_buffer_sync_for_device(dma_buffer_t *buffer, size_t offset, size_t size);

/**
 * Sync buffer for CPU (Device -> CPU)
 * Call after accelerator writes to buffer
 * 
 * buffer: Buffer to sync
 * offset: Offset within buffer
 * size: Size to sync (0 = entire buffer)
 */
void dma_buffer_sync_for_cpu(dma_buffer_t *buffer, size_t offset, size_t size);

/**
 * Get physical address for a virtual address within a buffer
 * 
 * buffer: DMA buffer
 * offset: Offset within buffer
 * 
 * Returns: Physical address
 */
uint64_t dma_buffer_get_phys(dma_buffer_t *buffer, size_t offset);

/**
 * Memory buffer structure (compatible with FreeRTOS version)
 */
typedef struct {
    void *ptr;
    size_t size;
    uint64_t phys_addr;
} memory_buffer_t;

/**
 * Allocate memory buffer (wrapper around dma_buffer_alloc)
 * For compatibility with FreeRTOS memory_manager interface
 */
int memory_allocate_ddr(size_t size, size_t alignment, memory_buffer_t *buffer);

/**
 * Free memory buffer
 */
void memory_free_ddr(memory_buffer_t *buffer);

/**
 * Allocate memory for weights
 */
int memory_allocate_weights(size_t size, memory_buffer_t *buffer);

/**
 * Allocate memory for bias
 */
int memory_allocate_bias(size_t size, memory_buffer_t *buffer);

/**
 * Allocate main inference memory buffer
 */
int memory_allocate_inference_buffer(memory_buffer_t *buffer);

/**
 * Get physical address from virtual address
 */
uint64_t memory_get_phys_addr(void *virt_addr);

/**
 * Flush cache for memory region (no-op for uncached DMA buffers)
 */
void memory_flush_cache(void *addr, size_t size);

/**
 * Invalidate cache for memory region (no-op for uncached DMA buffers)
 */
void memory_invalidate_cache(void *addr, size_t size);

#endif /* DMA_BUFFER_MANAGER_H */
