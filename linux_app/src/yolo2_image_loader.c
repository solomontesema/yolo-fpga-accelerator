/**
 * YOLOv2 Image Loader - Linux Implementation
 * 
 * Uses stb_image for loading JPEG/PNG images.
 */

#include "yolo2_image_loader.h"
#include "yolo2_config.h"
#include "yolo2_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// stb_image implementation will be in a separate compilation unit
// Here we just use the declarations
#define STBI_NO_THREAD_LOCALS
#include "stb_image.h"

static void yolo2_fill_chw(float *data, int w, int h, int c, float value)
{
    const size_t plane = (size_t)w * (size_t)h;
    for (int ch = 0; ch < c; ++ch) {
        float *dst = data + (size_t)ch * plane;
        for (size_t idx = 0; idx < plane; ++idx) {
            dst[idx] = value;
        }
    }
}

/**
 * Load image from file
 */
int yolo2_load_image(const char *image_path, float *output_buffer) {
    int width, height, channels;
    unsigned char *data;
    
    YOLO2_LOG_INFO("Loading image: %s\n", image_path);
    
    // Load image
    data = stbi_load(image_path, &width, &height, &channels, 3);  // Force RGB
    if (!data) {
        fprintf(stderr, "ERROR: Failed to load image: %s\n", image_path);
        fprintf(stderr, "       stb_image error: %s\n", stbi_failure_reason());
        return -1;
    }
    
    YOLO2_LOG_INFO("  Original size: %dx%dx%d\n", width, height, channels);
    
    // Convert to float [0-1] normalized, CHW layout to match CPU reference.
    float *temp_image = (float*)malloc((size_t)width * (size_t)height * 3 * sizeof(float));
    if (!temp_image || !output_buffer) {
        fprintf(stderr, "ERROR: Failed to allocate temporary image buffer\n");
        stbi_image_free(data);
        return -1;
    }
    
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < width; ++i) {
                const int dst_index = i + width * j + width * height * k;
                const int src_index = k + 3 * i + 3 * width * j;
                temp_image[dst_index] = (float)data[src_index] / 255.0f;
            }
        }
    }
    
    stbi_image_free(data);
    
    // Letterbox resize to network input size
    int result = yolo2_letterbox_image(temp_image, width, height, 3,
                                       output_buffer, INPUT_WIDTH, INPUT_HEIGHT);
    
    free(temp_image);
    
    if (result == 0) {
        YOLO2_LOG_INFO("  Resized to: %dx%dx%d (letterbox)\n", INPUT_WIDTH, INPUT_HEIGHT, 3);
    }
    
    return result;
}

/**
 * Load image with original dimensions
 */
int yolo2_load_image_raw(const char *image_path, float **output_buffer, 
                         int *width, int *height, int *channels) {
    unsigned char *data;
    
    data = stbi_load(image_path, width, height, channels, 3);
    if (!data) {
        fprintf(stderr, "ERROR: Failed to load image: %s\n", image_path);
        return -1;
    }
    
    const int w = *width;
    const int h = *height;
    const size_t size = (size_t)w * (size_t)h * 3;
    *output_buffer = (float*)malloc(size * sizeof(float));
    if (!*output_buffer) {
        stbi_image_free(data);
        return -1;
    }
    
    for (int k = 0; k < 3; ++k) {
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                const int dst_index = i + w * j + w * h * k;
                const int src_index = k + 3 * i + 3 * w * j;
                (*output_buffer)[dst_index] = (float)data[src_index] / 255.0f;
            }
        }
    }
    
    stbi_image_free(data);
    *channels = 3;
    
    return 0;
}

/**
 * Letterbox resize
 */
int yolo2_letterbox_image(float *input, int in_w, int in_h, int in_c,
                          float *output, int out_w, int out_h) {
    if (!input || !output || in_w <= 0 || in_h <= 0 || in_c <= 0 || out_w <= 0 || out_h <= 0) {
        return -1;
    }

    int new_w = in_w;
    int new_h = in_h;
    if (((float)out_w / (float)in_w) < ((float)out_h / (float)in_h)) {
        new_w = out_w;
        new_h = (in_h * out_w) / in_w;
    } else {
        new_h = out_h;
        new_w = (in_w * out_h) / in_h;
    }

    float *resized = (float *)malloc((size_t)new_w * (size_t)new_h * (size_t)in_c * sizeof(float));
    if (!resized) {
        return -1;
    }

    const int resize_rc = yolo2_resize_image(input, in_w, in_h, in_c, resized, new_w, new_h);
    if (resize_rc != 0) {
        free(resized);
        return resize_rc;
    }

    yolo2_fill_chw(output, out_w, out_h, in_c, 0.5f);

    const int dx = (out_w - new_w) / 2;
    const int dy = (out_h - new_h) / 2;
    const size_t out_plane = (size_t)out_w * (size_t)out_h;
    const size_t resized_plane = (size_t)new_w * (size_t)new_h;

    for (int k = 0; k < in_c; ++k) {
        const float *src = resized + (size_t)k * resized_plane;
        float *dst = output + (size_t)k * out_plane;
        for (int y = 0; y < new_h; ++y) {
            memcpy(dst + (size_t)(y + dy) * (size_t)out_w + (size_t)dx,
                   src + (size_t)y * (size_t)new_w,
                   (size_t)new_w * sizeof(float));
        }
    }

    free(resized);
    return 0;
}

/**
 * Bilinear resize
 */
int yolo2_resize_image(float *input, int in_w, int in_h, int in_c,
                       float *output, int out_w, int out_h) {
    if (!input || !output || in_w <= 0 || in_h <= 0 || in_c <= 0 || out_w <= 0 || out_h <= 0) {
        return -1;
    }

    if (out_w == 1 || out_h == 1) {
        // Degenerate resize: pick the top-left pixel for simplicity.
        const size_t out_plane = (size_t)out_w * (size_t)out_h;
        for (int k = 0; k < in_c; ++k) {
            const float v = input[(size_t)k * (size_t)in_w * (size_t)in_h];
            for (size_t idx = 0; idx < out_plane; ++idx) {
                output[(size_t)k * out_plane + idx] = v;
            }
        }
        return 0;
    }

    const float w_scale = (float)(in_w - 1) / (float)(out_w - 1);
    const float h_scale = (float)(in_h - 1) / (float)(out_h - 1);
    const size_t part_plane = (size_t)out_w * (size_t)in_h;

    float *part = (float *)malloc(part_plane * (size_t)in_c * sizeof(float));
    if (!part) {
        return -1;
    }

    for (int k = 0; k < in_c; ++k) {
        const float *in = input + (size_t)k * (size_t)in_w * (size_t)in_h;
        float *mid = part + (size_t)k * part_plane;
        for (int r = 0; r < in_h; ++r) {
            for (int c = 0; c < out_w; ++c) {
                float val;
                if (c == out_w - 1 || in_w == 1) {
                    val = in[(size_t)r * (size_t)in_w + (size_t)(in_w - 1)];
                } else {
                    const float sx = (float)c * w_scale;
                    const int ix = (int)sx;
                    const float dx = sx - (float)ix;
                    const float v0 = in[(size_t)r * (size_t)in_w + (size_t)ix];
                    const float v1 = in[(size_t)r * (size_t)in_w + (size_t)(ix + 1)];
                    val = (1.0f - dx) * v0 + dx * v1;
                }
                mid[(size_t)r * (size_t)out_w + (size_t)c] = val;
            }
        }
    }

    const size_t out_plane = (size_t)out_w * (size_t)out_h;
    for (int k = 0; k < in_c; ++k) {
        const float *mid = part + (size_t)k * part_plane;
        float *out = output + (size_t)k * out_plane;
        for (int r = 0; r < out_h; ++r) {
            const float sy = (float)r * h_scale;
            const int iy = (int)sy;
            const float dy = sy - (float)iy;
            for (int c = 0; c < out_w; ++c) {
                float val = (1.0f - dy) * mid[(size_t)iy * (size_t)out_w + (size_t)c];
                if (!(r == out_h - 1 || in_h == 1)) {
                    val += dy * mid[(size_t)(iy + 1) * (size_t)out_w + (size_t)c];
                }
                out[(size_t)r * (size_t)out_w + (size_t)c] = val;
            }
        }
    }

    free(part);
    return 0;
}
