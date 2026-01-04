/**
 * YOLOv2 Linux App - Minimal V4L2 camera capture helpers
 *
 * Supports MJPEG (decoded via stb_image) and YUYV (software convert) capture.
 */

#ifndef YOLO2_V4L2_H
#define YOLO2_V4L2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    YOLO2_V4L2_FMT_MJPEG = 0,
    YOLO2_V4L2_FMT_YUYV = 1,
} yolo2_v4l2_format_t;

typedef struct {
    void *start;
    size_t length;
} yolo2_v4l2_mmap_buf_t;

typedef struct {
    int fd;
    int width;
    int height;
    int fps;
    uint32_t pixfmt;                // actual V4L2 pixfmt (e.g., V4L2_PIX_FMT_MJPEG)
    yolo2_v4l2_mmap_buf_t *buffers; // mmap'd V4L2 buffers
    unsigned int num_buffers;
} yolo2_v4l2_camera_t;

typedef struct {
    const uint8_t *data;
    size_t size;
    unsigned int index;
} yolo2_v4l2_frame_t;

const char *yolo2_v4l2_pixfmt_name(uint32_t pixfmt);

int yolo2_v4l2_open(
    yolo2_v4l2_camera_t *cam,
    const char *device,
    int width,
    int height,
    int fps,
    yolo2_v4l2_format_t requested_format);

int yolo2_v4l2_start(yolo2_v4l2_camera_t *cam);
int yolo2_v4l2_stop(yolo2_v4l2_camera_t *cam);
void yolo2_v4l2_close(yolo2_v4l2_camera_t *cam);

int yolo2_v4l2_dequeue(yolo2_v4l2_camera_t *cam, yolo2_v4l2_frame_t *frame);
int yolo2_v4l2_enqueue(yolo2_v4l2_camera_t *cam, const yolo2_v4l2_frame_t *frame);

int yolo2_decode_mjpeg_to_rgb24(const uint8_t *mjpeg, size_t mjpeg_size, uint8_t *rgb, int width, int height);
void yolo2_yuyv_to_rgb24(const uint8_t *yuyv, uint8_t *rgb, int width, int height);

#ifdef __cplusplus
}
#endif

#endif /* YOLO2_V4L2_H */

