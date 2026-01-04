/**
 * YOLOv2 Linux App - Simple headless drawing + image output helpers
 *
 * Draws bounding boxes into an RGB24 frame buffer and writes PNG files.
 */

#ifndef YOLO2_DRAW_H
#define YOLO2_DRAW_H

#include <stdint.h>

#include "yolo2_postprocess.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Draw a rectangle border into an RGB24 buffer (interleaved RGB).
 *
 * Coordinates are inclusive pixel corners, clamped to the image bounds.
 */
void yolo2_draw_rect_rgb24(
    uint8_t *rgb,
    int width,
    int height,
    int x0,
    int y0,
    int x1,
    int y1,
    int thickness,
    uint8_t r,
    uint8_t g,
    uint8_t b);

/**
 * Draw YOLO detections (above thresh) into an RGB24 buffer.
 * Uses a simple fixed color palette and renders a small text label per box.
 *
 * Returns: number of boxes drawn.
 */
int yolo2_draw_detections_rgb24(
    uint8_t *rgb,
    int width,
    int height,
    const yolo2_detection_t *dets,
    int num_dets,
    float thresh,
    const char **labels,
    int num_labels);

/**
 * Write an RGB24 buffer as a PNG file.
 *
 * Returns: 0 on success, -1 on error.
 */
int yolo2_write_png_rgb24(const char *path, const uint8_t *rgb, int width, int height);

#ifdef __cplusplus
}
#endif

#endif /* YOLO2_DRAW_H */
