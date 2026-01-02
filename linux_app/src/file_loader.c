/**
 * File Loader - Binary file loading for weights and Q values
 */

#include "file_loader.h"
#include "yolo2_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Load binary file into memory
 */
int load_binary_file(const char *path, void **buffer, size_t *size) {
    FILE *file;
    long file_size;
    void *data;
    size_t bytes_read;
    
    file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", path);
        return -1;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (file_size <= 0) {
        fprintf(stderr, "ERROR: File is empty or invalid: %s\n", path);
        fclose(file);
        return -1;
    }
    
    // Allocate buffer
    data = malloc(file_size);
    if (!data) {
        fprintf(stderr, "ERROR: Failed to allocate %ld bytes for %s\n", file_size, path);
        fclose(file);
        return -1;
    }
    
    // Read file
    bytes_read = fread(data, 1, file_size, file);
    fclose(file);
    
    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, "ERROR: Read %zu of %ld bytes from %s\n", bytes_read, file_size, path);
        free(data);
        return -1;
    }
    
    *buffer = data;
    *size = file_size;
    
    YOLO2_LOG_INFO("  Loaded %s: %zu bytes\n", path, *size);
    
    return 0;
}

/**
 * Load weights file
 */
int load_weights(const char *path, void **buffer, size_t *size) {
    return load_binary_file(path, buffer, size);
}

/**
 * Load bias file
 */
int load_bias(const char *path, void **buffer, size_t *size) {
    return load_binary_file(path, buffer, size);
}

/**
 * Load Q values (INT16 mode)
 */
int load_q_values(const char *path, int32_t **buffer, size_t *count) {
    void *data = NULL;
    size_t bytes = 0;
    
    int result = load_binary_file(path, &data, &bytes);
    if (result == 0) {
        *buffer = (int32_t*)data;
        *count = bytes / sizeof(int32_t);
        YOLO2_LOG_INFO("    (%zu Q values)\n", *count);
    }
    
    return result;
}
