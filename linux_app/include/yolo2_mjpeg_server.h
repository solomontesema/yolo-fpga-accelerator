/**
 * YOLOv2 Linux App - MJPEG over HTTP streaming (headless)
 *
 * This is meant for a simple "stream to PC" workflow:
 *   - Run `yolo2_linux` on KV260 with `--stream-mjpeg 8080`
 *   - Open `http://<kv260-ip>:8080/` in VLC
 *
 * The server is intentionally minimal: single client, best-effort delivery.
 */

#ifndef YOLO2_MJPEG_SERVER_H
#define YOLO2_MJPEG_SERVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int listen_fd;
    int client_fd;
    int port;
    char bind_addr[64];
} yolo2_mjpeg_server_t;

int yolo2_mjpeg_server_start(yolo2_mjpeg_server_t *srv, const char *bind_addr, int port);
void yolo2_mjpeg_server_stop(yolo2_mjpeg_server_t *srv);

// Non-blocking accept (no-op if already connected). Returns 1 if a client is connected.
int yolo2_mjpeg_server_poll_accept(yolo2_mjpeg_server_t *srv);

// Encode `rgb24` to JPEG and send it as the next multipart chunk (best-effort).
// Returns: 0 on success, -1 on fatal error. If no client is connected, returns 0.
int yolo2_mjpeg_server_send_rgb24(
    yolo2_mjpeg_server_t *srv,
    const uint8_t *rgb,
    int width,
    int height,
    int jpeg_quality);

#ifdef __cplusplus
}
#endif

#endif /* YOLO2_MJPEG_SERVER_H */

