/**
 * YOLOv2 Network Structure and Configuration Parser
 * 
 * Linux port - Replaces FATFS with standard file I/O
 */

#ifndef YOLO2_NETWORK_H
#define YOLO2_NETWORK_H

#include <stdint.h>
#include <stdbool.h>

// Layer types
typedef enum {
    LAYER_NET = 0,
    LAYER_CONVOLUTIONAL = 1,
    LAYER_MAXPOOL = 2,
    LAYER_REORG = 3,
    LAYER_ROUTE = 4,
    LAYER_REGION = 5,
    LAYER_UNKNOWN = 99
} layer_type_t;

// Activation types
typedef enum {
    ACT_LINEAR = 0,
    ACT_LEAKY = 1
} activation_t;

// Layer structure (simplified for YOLOv2)
typedef struct {
    layer_type_t type;
    int batch_normalize;  // 1 if batch norm, 0 otherwise
    int filters;          // Number of output filters (for conv)
    int size;             // Kernel size
    int stride;
    int pad;
    activation_t activation;
    
    // Computed dimensions
    int h, w, c;          // Input dimensions
    int out_h, out_w, out_c;  // Output dimensions
    int outputs;           // Total output elements
    
    // Route layer specific
    int n;                // Number of input layers for route
    int *input_layers;    // Array of layer indices to route from
    int *input_sizes;     // Array of sizes for each input
    
    // Region layer specific
    int classes;
    int coords;
    int num;
    int softmax;          // 1 if softmax enabled (YOLOv2 region uses this)
    float thresh;
    float nms;
} layer_t;

// Network structure
typedef struct {
    int n;                // Number of layers
    layer_t *layers;      // Array of layers
    int w, h, c;          // Input dimensions
    int inputs;           // Total input elements
} network_t;

/**
 * Parse network configuration from .cfg file
 * 
 * cfg_path: Path to configuration file
 * Returns: Parsed network structure, or NULL on error
 */
network_t* yolo2_parse_network_cfg(const char *cfg_path);

/**
 * Free network structure
 */
void yolo2_free_network(network_t *net);

/**
 * Get output layer
 */
layer_t* yolo2_get_network_output_layer(network_t *net);

#endif /* YOLO2_NETWORK_H */
