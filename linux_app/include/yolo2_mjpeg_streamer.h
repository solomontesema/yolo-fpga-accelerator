/**
 * YOLOv2 Linux App - MJPEG streaming helper (threaded)
 *
 * Runs an MJPEG-over-HTTP server in a background thread and continuously serves
 * the latest RGB frame at a fixed rate. This keeps VLC clients alive even when
 * inference is slow.
 */

#ifndef YOLO2_MJPEG_STREAMER_H
#define YOLO2_MJPEG_STREAMER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yolo2_mjpeg_streamer yolo2_mjpeg_streamer_t;

int yolo2_mjpeg_streamer_start(
    yolo2_mjpeg_streamer_t **out,
    const char *bind_addr,
    int port,
    int fps,
    int jpeg_quality);

void yolo2_mjpeg_streamer_stop(yolo2_mjpeg_streamer_t *s);

// Copy a new RGB24 frame into the streamer (safe to call from the inference thread).
// Returns 0 on success, -1 on error.
int yolo2_mjpeg_streamer_update_rgb24(yolo2_mjpeg_streamer_t *s, const uint8_t *rgb, int width, int height);

#ifdef __cplusplus
}
#endif

#endif /* YOLO2_MJPEG_STREAMER_H */

