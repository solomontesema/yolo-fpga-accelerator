/**
 * File Loader - Binary file loading
 */

#ifndef FILE_LOADER_H
#define FILE_LOADER_H

#include <stdint.h>
#include <stddef.h>

/**
 * Load binary file into memory
 * 
 * path: Path to file
 * buffer: Pointer to receive allocated buffer (caller must free)
 * size: Pointer to receive file size
 * 
 * Returns: 0 on success, -1 on error
 */
int load_binary_file(const char *path, void **buffer, size_t *size);

/**
 * Load weights file
 */
int load_weights(const char *path, void **buffer, size_t *size);

/**
 * Load bias file
 */
int load_bias(const char *path, void **buffer, size_t *size);

/**
 * Load Q values (INT16 mode)
 */
int load_q_values(const char *path, int32_t **buffer, size_t *count);

#endif /* FILE_LOADER_H */
