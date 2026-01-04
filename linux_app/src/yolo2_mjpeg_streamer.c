/**
 * YOLOv2 Linux App - MJPEG streaming helper (threaded)
 */

#include "yolo2_mjpeg_streamer.h"

#include "yolo2_log.h"
#include "yolo2_mjpeg_server.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct yolo2_mjpeg_streamer {
    yolo2_mjpeg_server_t server;

    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;

    int stop;
    int started; // -1 failed, 0 starting, 1 running

    char bind_addr[64];
    int port;
    int fps;
    int jpeg_quality;

    uint8_t *rgb;
    size_t rgb_cap;
    int width;
    int height;
    int has_frame;
};

static int clamp_int(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void sleep_ms(int ms)
{
    if (ms <= 0) ms = 1;
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}

static void *stream_thread(void *arg)
{
    yolo2_mjpeg_streamer_t *s = (yolo2_mjpeg_streamer_t *)arg;

    if (yolo2_mjpeg_server_start(&s->server, s->bind_addr, s->port) != 0) {
        pthread_mutex_lock(&s->mu);
        s->started = -1;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
        return NULL;
    }

    pthread_mutex_lock(&s->mu);
    s->started = 1;
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);

    const int interval_ms = (s->fps > 0) ? clamp_int(1000 / s->fps, 50, 1000) : 250;
    uint8_t *local = NULL;
    size_t local_cap = 0;
    int local_w = 0;
    int local_h = 0;
    int local_has = 0;

    for (;;) {
        pthread_mutex_lock(&s->mu);
        const int stop = s->stop;
        const int has_frame = s->has_frame;
        const int w = s->width;
        const int h = s->height;
        const size_t bytes = has_frame ? ((size_t)w * (size_t)h * 3u) : 0;

        if (stop) {
            pthread_mutex_unlock(&s->mu);
            break;
        }

        if (has_frame && bytes > 0 && s->rgb) {
            if (bytes > local_cap) {
                uint8_t *p = (uint8_t *)realloc(local, bytes);
                if (p) {
                    local = p;
                    local_cap = bytes;
                }
            }
            if (local && local_cap >= bytes) {
                memcpy(local, s->rgb, bytes);
                local_w = w;
                local_h = h;
                local_has = 1;
            } else {
                local_has = 0;
            }
        }

        pthread_mutex_unlock(&s->mu);

        if (local_has) {
            (void)yolo2_mjpeg_server_send_rgb24(&s->server, local, local_w, local_h, s->jpeg_quality);
        } else {
            (void)yolo2_mjpeg_server_poll_accept(&s->server);
        }

        sleep_ms(interval_ms);
    }

    free(local);
    yolo2_mjpeg_server_stop(&s->server);
    return NULL;
}

int yolo2_mjpeg_streamer_start(
    yolo2_mjpeg_streamer_t **out,
    const char *bind_addr,
    int port,
    int fps,
    int jpeg_quality)
{
    if (!out) return -1;
    *out = NULL;

    if (port <= 0 || port > 65535) return -1;

    yolo2_mjpeg_streamer_t *s = (yolo2_mjpeg_streamer_t *)calloc(1, sizeof(*s));
    if (!s) return -1;

    s->server.listen_fd = -1;
    s->server.client_fd = -1;
    s->stop = 0;
    s->started = 0;
    s->port = port;
    s->fps = clamp_int(fps > 0 ? fps : 4, 1, 30);
    s->jpeg_quality = clamp_int(jpeg_quality, 1, 100);
    snprintf(s->bind_addr, sizeof(s->bind_addr), "%s", (bind_addr && bind_addr[0]) ? bind_addr : "0.0.0.0");

    if (pthread_mutex_init(&s->mu, NULL) != 0) {
        free(s);
        return -1;
    }
    if (pthread_cond_init(&s->cv, NULL) != 0) {
        pthread_mutex_destroy(&s->mu);
        free(s);
        return -1;
    }

    if (pthread_create(&s->thread, NULL, stream_thread, s) != 0) {
        pthread_cond_destroy(&s->cv);
        pthread_mutex_destroy(&s->mu);
        free(s);
        return -1;
    }

    // Wait for server startup result.
    pthread_mutex_lock(&s->mu);
    while (s->started == 0) {
        pthread_cond_wait(&s->cv, &s->mu);
    }
    const int started = s->started;
    pthread_mutex_unlock(&s->mu);

    if (started != 1) {
        yolo2_mjpeg_streamer_stop(s);
        return -1;
    }

    YOLO2_LOG_INFO("MJPEG stream: http://<kv260-ip>:%d/ (bind %s, send %dfps)\n", s->port, s->bind_addr, s->fps);

    *out = s;
    return 0;
}

void yolo2_mjpeg_streamer_stop(yolo2_mjpeg_streamer_t *s)
{
    if (!s) return;

    pthread_mutex_lock(&s->mu);
    s->stop = 1;
    pthread_mutex_unlock(&s->mu);

    (void)pthread_join(s->thread, NULL);

    free(s->rgb);
    s->rgb = NULL;
    s->rgb_cap = 0;

    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->mu);

    free(s);
}

int yolo2_mjpeg_streamer_update_rgb24(yolo2_mjpeg_streamer_t *s, const uint8_t *rgb, int width, int height)
{
    if (!s || !rgb || width <= 0 || height <= 0) return -1;

    const size_t bytes = (size_t)width * (size_t)height * 3u;

    pthread_mutex_lock(&s->mu);
    if (bytes > s->rgb_cap) {
        uint8_t *p = (uint8_t *)realloc(s->rgb, bytes);
        if (!p) {
            pthread_mutex_unlock(&s->mu);
            return -1;
        }
        s->rgb = p;
        s->rgb_cap = bytes;
    }

    memcpy(s->rgb, rgb, bytes);
    s->width = width;
    s->height = height;
    s->has_frame = 1;
    pthread_mutex_unlock(&s->mu);

    return 0;
}

