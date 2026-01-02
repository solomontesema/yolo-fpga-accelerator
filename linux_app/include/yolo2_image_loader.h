/**
 * YOLOv2 Image Loader - Linux Implementation
 * 
 * Uses stb_image for JPEG/PNG loading and provides letterbox resize.
 */

#ifndef YOLO2_IMAGE_LOADER_H
#define YOLO2_IMAGE_LOADER_H

#include <stdint.h>

/**
 * Load image from file and convert to float array (416x416x3)
 * 
 * image_path: Path to image file (JPEG, PNG, etc.)
 * output_buffer: Pre-allocated buffer for normalized [0-1] float data
 *                Size must be INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS.
 *                Layout matches the CPU reference: CHW (channel-major).
 * 
 * Returns: 0 on success, -1 on error
 */
int yolo2_load_image(const char *image_path, float *output_buffer);

/**
 * Load image with original dimensions
 * 
 * image_path: Path to image file
 * output_buffer: Pointer to receive allocated buffer
 * width: Pointer to receive image width
 * height: Pointer to receive image height
 * channels: Pointer to receive image channels
 * 
 * Returns: 0 on success, -1 on error
 * Note: Caller must free output_buffer
 */
int yolo2_load_image_raw(const char *image_path, float **output_buffer, 
                         int *width, int *height, int *channels);

/**
 * Letterbox resize (maintains aspect ratio, pads with gray)
 * 
 * input: Input image data (float, CHW format)
 * in_w, in_h, in_c: Input dimensions
 * output: Output buffer (pre-allocated)
 * out_w, out_h: Output dimensions
 * 
 * Returns: 0 on success
 */
int yolo2_letterbox_image(float *input, int in_w, int in_h, int in_c,
                          float *output, int out_w, int out_h);

/**
 * Bilinear resize (CHW), matches CPU reference `resize_image()`.
 */
int yolo2_resize_image(float *input, int in_w, int in_h, int in_c,
                       float *output, int out_w, int out_h);

#endif /* YOLO2_IMAGE_LOADER_H */
