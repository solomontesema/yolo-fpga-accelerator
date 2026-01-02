/**
 * YOLOv2 Network Configuration Parser - Linux Implementation
 * 
 * Parses Darknet-style .cfg files and creates network structure.
 * Uses standard C file I/O.
 */

#include "yolo2_network.h"
#include "yolo2_config.h"
#include "yolo2_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Maximum line length and options per section
#define MAX_LINE_LEN 256
#define MAX_OPTIONS 64

// Key-value option
typedef struct {
    char key[64];
    char val[128];
} option_t;

// Section with options
typedef struct {
    char type[64];
    option_t options[MAX_OPTIONS];
    int num_options;
} section_t;

// Strip whitespace from string
static void strip(char *s) {
    char *p = s;
    int l = strlen(p);

    // Strip trailing
    while (l > 0 && (p[l-1] == ' ' || p[l-1] == '\t' || p[l-1] == '\n' || p[l-1] == '\r')) {
        p[--l] = 0;
    }
    // Strip leading
    while (*p && (*p == ' ' || *p == '\t')) {
        ++p;
        --l;
    }
    memmove(s, p, l + 1);
}

// Find option value in section
static const char* option_find(section_t *s, const char *key) {
    for (int i = 0; i < s->num_options; i++) {
        if (strcmp(s->options[i].key, key) == 0) {
            return s->options[i].val;
        }
    }
    return NULL;
}

// Get int option with default
static int option_find_int(section_t *s, const char *key, int def) {
    const char *v = option_find(s, key);
    if (v) return atoi(v);
    return def;
}

// Get float option with default  
static float option_find_float(section_t *s, const char *key, float def) {
    const char *v = option_find(s, key);
    if (v) return atof(v);
    return def;
}

// Get string option with default
static const char* option_find_str(section_t *s, const char *key, const char *def) {
    const char *v = option_find(s, key);
    if (v) return v;
    return def;
}

// Parse a line into key=value
static int parse_option(const char *line, option_t *opt) {
    char *eq = strchr(line, '=');
    if (!eq) return 0;
    
    int key_len = eq - line;
    if (key_len <= 0 || key_len >= 64) return 0;
    
    strncpy(opt->key, line, key_len);
    opt->key[key_len] = '\0';
    strip(opt->key);
    
    strncpy(opt->val, eq + 1, 127);
    opt->val[127] = '\0';
    strip(opt->val);
    
    return 1;
}

// Case-insensitive string compare
static int str_eq(const char *a, const char *b) {
    while (*a && *b) {
        char ca = (*a >= 'A' && *a <= 'Z') ? (*a + 32) : *a;
        char cb = (*b >= 'A' && *b <= 'Z') ? (*b + 32) : *b;
        if (ca != cb) return 0;
        a++; b++;
    }
    return (*a == 0 && *b == 0);
}

// Make a layer from section
static void make_convolutional_layer(layer_t *l, section_t *s, int h, int w, int c) {
    l->type = LAYER_CONVOLUTIONAL;
    l->h = h;
    l->w = w;
    l->c = c;
    
    l->filters = option_find_int(s, "filters", 1);
    l->size = option_find_int(s, "size", 1);
    l->stride = option_find_int(s, "stride", 1);
    l->pad = option_find_int(s, "pad", 0);
    l->batch_normalize = option_find_int(s, "batch_normalize", 0);
    
    const char *act = option_find_str(s, "activation", "linear");
    if (str_eq(act, "leaky")) {
        l->activation = ACT_LEAKY;
    } else {
        l->activation = ACT_LINEAR;
    }
    
    // If pad=1 and size=3, actual padding is size/2
    int padding = l->pad ? l->size / 2 : 0;
    
    l->out_h = (l->h + 2*padding - l->size) / l->stride + 1;
    l->out_w = (l->w + 2*padding - l->size) / l->stride + 1;
    l->out_c = l->filters;
    l->outputs = l->out_h * l->out_w * l->out_c;
    
    // Store actual padding for accelerator
    l->pad = padding;
}

static void make_maxpool_layer(layer_t *l, section_t *s, int h, int w, int c) {
    l->type = LAYER_MAXPOOL;
    l->h = h;
    l->w = w;
    l->c = c;
    
    l->size = option_find_int(s, "size", 2);
    l->stride = option_find_int(s, "stride", 2);
    l->pad = option_find_int(s, "padding", 0);
    
    l->out_h = (l->h - l->size) / l->stride + 1;
    l->out_w = (l->w - l->size) / l->stride + 1;
    l->out_c = l->c;
    l->outputs = l->out_h * l->out_w * l->out_c;
}

static void make_reorg_layer(layer_t *l, section_t *s, int h, int w, int c) {
    l->type = LAYER_REORG;
    l->h = h;
    l->w = w;
    l->c = c;
    
    l->stride = option_find_int(s, "stride", 2);
    
    l->out_h = l->h / l->stride;
    l->out_w = l->w / l->stride;
    l->out_c = l->c * l->stride * l->stride;
    l->outputs = l->out_h * l->out_w * l->out_c;
}

static void make_route_layer(layer_t *l, section_t *s, network_t *net, int idx) {
    l->type = LAYER_ROUTE;
    
    // Parse layers parameter (e.g., "layers=-1" or "layers=-9,-4")
    const char *layers_str = option_find_str(s, "layers", "-1");
    
    // Count how many layers
    int count = 1;
    for (const char *p = layers_str; *p; p++) {
        if (*p == ',') count++;
    }
    
    l->n = count;
    l->input_layers = (int*)malloc(count * sizeof(int));
    
    // Parse each layer index
    char buf[256];
    strncpy(buf, layers_str, 255);
    buf[255] = '\0';
    
    char *token = strtok(buf, ",");
    int i = 0;
    int total_c = 0;
    int out_h = 0, out_w = 0;
    
    while (token && i < count) {
        int layer_idx = atoi(token);
        if (layer_idx < 0) {
            layer_idx = idx + layer_idx;  // Relative index
        }
        l->input_layers[i] = layer_idx;
        
        if (layer_idx >= 0 && layer_idx < idx) {
            total_c += net->layers[layer_idx].out_c;
            out_h = net->layers[layer_idx].out_h;
            out_w = net->layers[layer_idx].out_w;
        }
        
        token = strtok(NULL, ",");
        i++;
    }
    
    l->h = out_h;
    l->w = out_w;
    l->c = total_c;
    l->out_h = out_h;
    l->out_w = out_w;
    l->out_c = total_c;
    l->outputs = l->out_h * l->out_w * l->out_c;
}

static void make_region_layer(layer_t *l, section_t *s, int h, int w, int c) {
    l->type = LAYER_REGION;
    l->h = h;
    l->w = w;
    l->c = c;
    
    l->classes = option_find_int(s, "classes", 20);
    l->coords = option_find_int(s, "coords", 4);
    l->num = option_find_int(s, "num", 5);
    l->softmax = option_find_int(s, "softmax", 0);
    l->thresh = option_find_float(s, "thresh", 0.5f);
    l->nms = option_find_float(s, "nms", 0.45f);
    
    l->out_h = h;
    l->out_w = w;
    l->out_c = l->num * (l->classes + l->coords + 1);
    l->outputs = l->out_h * l->out_w * l->out_c;
}

/**
 * Parse network configuration file
 */
network_t* yolo2_parse_network_cfg(const char *cfg_path) {
    FILE *file;
    char line[MAX_LINE_LEN];
    
    YOLO2_LOG_INFO("Parsing network configuration: %s\n", cfg_path);
    
    file = fopen(cfg_path, "r");
    if (!file) {
        fprintf(stderr, "ERROR: Cannot open config file: %s\n", cfg_path);
        return NULL;
    }
    
    // First pass: count sections (layers)
    int section_count = 0;
    while (fgets(line, MAX_LINE_LEN, file)) {
        strip(line);
        if (line[0] == '[') {
            section_count++;
        }
    }
    rewind(file);
    
    if (section_count <= 1) {
        fprintf(stderr, "ERROR: No layers found in config\n");
        fclose(file);
        return NULL;
    }
    
    // Allocate sections array
    section_t *sections = (section_t*)calloc(section_count, sizeof(section_t));
    if (!sections) {
        fclose(file);
        return NULL;
    }
    
    // Second pass: read all sections
    int cur_section = -1;
    while (fgets(line, MAX_LINE_LEN, file)) {
        strip(line);
        
        // Skip empty lines and comments
        if (line[0] == '\0' || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        // New section
        if (line[0] == '[') {
            cur_section++;
            if (cur_section < section_count) {
                // Extract section type (remove brackets)
                char *end = strchr(line, ']');
                if (end) *end = '\0';
                strncpy(sections[cur_section].type, line + 1, 63);
                sections[cur_section].num_options = 0;
            }
            continue;
        }
        
        // Option line
        if (cur_section >= 0 && cur_section < section_count) {
            section_t *s = &sections[cur_section];
            if (s->num_options < MAX_OPTIONS) {
                if (parse_option(line, &s->options[s->num_options])) {
                    s->num_options++;
                }
            }
        }
    }
    fclose(file);
    
    // Count actual layers (excluding [net])
    int layer_count = 0;
    for (int i = 0; i < section_count; i++) {
        const char *t = sections[i].type;
        if (str_eq(t, "convolutional") || str_eq(t, "conv") ||
            str_eq(t, "maxpool") || str_eq(t, "max") ||
            str_eq(t, "reorg") || str_eq(t, "route") ||
            str_eq(t, "region") || str_eq(t, "yolo")) {
            layer_count++;
        }
    }
    
    YOLO2_LOG_INFO("  Found %d sections, %d layers\n", section_count, layer_count);
    
    if (layer_count == 0) {
        fprintf(stderr, "ERROR: No valid layers found\n");
        free(sections);
        return NULL;
    }
    
    // Allocate network
    network_t *net = (network_t*)calloc(1, sizeof(network_t));
    if (!net) {
        free(sections);
        return NULL;
    }
    
    net->n = layer_count;
    net->layers = (layer_t*)calloc(layer_count, sizeof(layer_t));
    if (!net->layers) {
        free(net);
        free(sections);
        return NULL;
    }
    
    // Default network dimensions
    net->w = INPUT_WIDTH;
    net->h = INPUT_HEIGHT;
    net->c = INPUT_CHANNELS;
    
    // Process [net] section first
    for (int i = 0; i < section_count; i++) {
        if (str_eq(sections[i].type, "net") || str_eq(sections[i].type, "network")) {
            net->w = option_find_int(&sections[i], "width", INPUT_WIDTH);
            net->h = option_find_int(&sections[i], "height", INPUT_HEIGHT);
            net->c = option_find_int(&sections[i], "channels", INPUT_CHANNELS);
            break;
        }
    }
    net->inputs = net->w * net->h * net->c;
    
    // Build layers
    int h = net->h, w = net->w, c = net->c;
    int layer_idx = 0;
    
    for (int i = 0; i < section_count; i++) {
        section_t *s = &sections[i];
        const char *t = s->type;
        
        if (str_eq(t, "convolutional") || str_eq(t, "conv")) {
            make_convolutional_layer(&net->layers[layer_idx], s, h, w, c);
            h = net->layers[layer_idx].out_h;
            w = net->layers[layer_idx].out_w;
            c = net->layers[layer_idx].out_c;
            YOLO2_LOG_LAYER("    Layer %2d: conv      %3dx%3dx%4d -> %3dx%3dx%4d\n",
                            layer_idx, net->layers[layer_idx].h, net->layers[layer_idx].w, net->layers[layer_idx].c,
                            h, w, c);
            layer_idx++;
        }
        else if (str_eq(t, "maxpool") || str_eq(t, "max")) {
            make_maxpool_layer(&net->layers[layer_idx], s, h, w, c);
            h = net->layers[layer_idx].out_h;
            w = net->layers[layer_idx].out_w;
            c = net->layers[layer_idx].out_c;
            YOLO2_LOG_LAYER("    Layer %2d: maxpool   %3dx%3dx%4d -> %3dx%3dx%4d\n",
                            layer_idx, net->layers[layer_idx].h, net->layers[layer_idx].w, net->layers[layer_idx].c,
                            h, w, c);
            layer_idx++;
        }
        else if (str_eq(t, "reorg")) {
            make_reorg_layer(&net->layers[layer_idx], s, h, w, c);
            h = net->layers[layer_idx].out_h;
            w = net->layers[layer_idx].out_w;
            c = net->layers[layer_idx].out_c;
            YOLO2_LOG_LAYER("    Layer %2d: reorg     %3dx%3dx%4d -> %3dx%3dx%4d\n",
                            layer_idx, net->layers[layer_idx].h, net->layers[layer_idx].w, net->layers[layer_idx].c,
                            h, w, c);
            layer_idx++;
        }
        else if (str_eq(t, "route")) {
            make_route_layer(&net->layers[layer_idx], s, net, layer_idx);
            h = net->layers[layer_idx].out_h;
            w = net->layers[layer_idx].out_w;
            c = net->layers[layer_idx].out_c;
            YOLO2_LOG_LAYER("    Layer %2d: route     -> %3dx%3dx%4d\n",
                            layer_idx, h, w, c);
            layer_idx++;
        }
        else if (str_eq(t, "region") || str_eq(t, "yolo")) {
            make_region_layer(&net->layers[layer_idx], s, h, w, c);
            h = net->layers[layer_idx].out_h;
            w = net->layers[layer_idx].out_w;
            c = net->layers[layer_idx].out_c;
            YOLO2_LOG_LAYER("    Layer %2d: region    %3dx%3dx%4d (%d classes)\n",
                            layer_idx, h, w, net->layers[layer_idx].out_c, net->layers[layer_idx].classes);
            layer_idx++;
        }
    }
    
    free(sections);
    
    net->n = layer_idx;
    YOLO2_LOG_INFO("  Parsed network: %d layers, input %dx%dx%d\n", net->n, net->w, net->h, net->c);
    
    return net;
}

/**
 * Free network structure
 */
void yolo2_free_network(network_t *net) {
    if (!net) return;
    
    if (net->layers) {
        for (int i = 0; i < net->n; ++i) {
            if (net->layers[i].input_layers) {
                free(net->layers[i].input_layers);
            }
        }
        free(net->layers);
    }
    free(net);
}

/**
 * Get output layer
 */
layer_t* yolo2_get_network_output_layer(network_t *net) {
    if (!net || net->n == 0) return NULL;
    return &net->layers[net->n - 1];
}
