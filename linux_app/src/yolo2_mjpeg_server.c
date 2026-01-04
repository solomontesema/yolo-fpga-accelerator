/**
 * YOLOv2 Linux App - MJPEG over HTTP streaming (headless)
 */

#include "yolo2_mjpeg_server.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "stb_image_write.h"

typedef struct {
    uint8_t *data;
    size_t size;
    size_t cap;
} yolo2_mem_buf_t;

static void mem_buf_reset(yolo2_mem_buf_t *b)
{
    if (!b) return;
    b->size = 0;
}

static void mem_buf_free(yolo2_mem_buf_t *b)
{
    if (!b) return;
    free(b->data);
    b->data = NULL;
    b->size = 0;
    b->cap = 0;
}

static int mem_buf_append(yolo2_mem_buf_t *b, const void *data, size_t len)
{
    if (!b || !data || len == 0) return 0;

    if (b->size + len > b->cap) {
        size_t new_cap = b->cap ? b->cap : 64 * 1024;
        while (new_cap < b->size + len) {
            new_cap *= 2;
        }
        uint8_t *p = (uint8_t *)realloc(b->data, new_cap);
        if (!p) return -1;
        b->data = p;
        b->cap = new_cap;
    }

    memcpy(b->data + b->size, data, len);
    b->size += len;
    return 0;
}

static void stbi_write_cb(void *context, void *data, int size)
{
    yolo2_mem_buf_t *b = (yolo2_mem_buf_t *)context;
    if (!b || !data || size <= 0) return;
    (void)mem_buf_append(b, data, (size_t)size);
}

static int set_nonblocking(int fd)
{
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return -1;
    return 0;
}

static void close_client(yolo2_mjpeg_server_t *srv)
{
    if (!srv) return;
    if (srv->client_fd >= 0) {
        close(srv->client_fd);
        srv->client_fd = -1;
    }
}

static int send_all(int fd, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t off = 0;
    while (off < len) {
        const size_t chunk = len - off;
        ssize_t n = send(fd, p + off, chunk, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        off += (size_t)n;
    }
    return 0;
}

static int bind_listen(const char *bind_addr, int port)
{
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *res = NULL;
    const int rc = getaddrinfo((bind_addr && bind_addr[0]) ? bind_addr : NULL, port_str, &hints, &res);
    if (rc != 0 || !res) {
        fprintf(stderr, "ERROR: getaddrinfo(%s:%d) failed: %s\n", bind_addr ? bind_addr : "0.0.0.0", port, gai_strerror(rc));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;

        int yes = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            if (listen(fd, 1) == 0) {
                break;
            }
        }
        close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    return fd;
}

int yolo2_mjpeg_server_start(yolo2_mjpeg_server_t *srv, const char *bind_addr, int port)
{
    if (!srv || port <= 0 || port > 65535) return -1;

    memset(srv, 0, sizeof(*srv));
    srv->listen_fd = -1;
    srv->client_fd = -1;
    srv->port = port;
    snprintf(srv->bind_addr, sizeof(srv->bind_addr), "%s", (bind_addr && bind_addr[0]) ? bind_addr : "0.0.0.0");

    srv->listen_fd = bind_listen(srv->bind_addr, port);
    if (srv->listen_fd < 0) {
        return -1;
    }

    if (set_nonblocking(srv->listen_fd) != 0) {
        fprintf(stderr, "WARNING: Failed to set non-blocking listen socket\n");
    }

    return 0;
}

void yolo2_mjpeg_server_stop(yolo2_mjpeg_server_t *srv)
{
    if (!srv) return;

    close_client(srv);

    if (srv->listen_fd >= 0) {
        close(srv->listen_fd);
        srv->listen_fd = -1;
    }
}

int yolo2_mjpeg_server_poll_accept(yolo2_mjpeg_server_t *srv)
{
    if (!srv || srv->listen_fd < 0) return 0;
    if (srv->client_fd >= 0) return 1;

    struct sockaddr_storage addr;
    socklen_t addrlen = sizeof(addr);
    const int cfd = accept(srv->listen_fd, (struct sockaddr *)&addr, &addrlen);
    if (cfd < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 0;
        }
        return 0;
    }

    // Best-effort: avoid stalling inference on slow receivers.
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    (void)setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    // Write response header.
    const char *hdr =
        "HTTP/1.0 200 OK\r\n"
        "Cache-Control: no-cache\r\n"
        "Pragma: no-cache\r\n"
        "Connection: close\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "\r\n";

    if (send_all(cfd, hdr, strlen(hdr)) != 0) {
        close(cfd);
        return 0;
    }

    srv->client_fd = cfd;
    return 1;
}

int yolo2_mjpeg_server_send_rgb24(
    yolo2_mjpeg_server_t *srv,
    const uint8_t *rgb,
    int width,
    int height,
    int jpeg_quality)
{
    if (!srv || srv->listen_fd < 0) return -1;
    if (!rgb || width <= 0 || height <= 0) return -1;

    if (jpeg_quality < 1) jpeg_quality = 1;
    if (jpeg_quality > 100) jpeg_quality = 100;

    // Try to accept a client if none is connected.
    if (!yolo2_mjpeg_server_poll_accept(srv)) {
        return 0;
    }

    yolo2_mem_buf_t jpg = {0};
    mem_buf_reset(&jpg);

    // Encode to JPEG in memory.
    const int ok = stbi_write_jpg_to_func(stbi_write_cb, &jpg, width, height, 3, rgb, jpeg_quality);
    if (!ok || jpg.size == 0) {
        mem_buf_free(&jpg);
        return 0;
    }

    char part_hdr[256];
    const int n = snprintf(
        part_hdr,
        sizeof(part_hdr),
        "--frame\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: %zu\r\n"
        "\r\n",
        jpg.size);

    if (n <= 0 || (size_t)n >= sizeof(part_hdr)) {
        mem_buf_free(&jpg);
        return 0;
    }

    int rc = 0;
    if (send_all(srv->client_fd, part_hdr, (size_t)n) != 0 ||
        send_all(srv->client_fd, jpg.data, jpg.size) != 0 ||
        send_all(srv->client_fd, "\r\n", 2) != 0) {
        close_client(srv);
        rc = 0; // non-fatal; client can reconnect
    }

    mem_buf_free(&jpg);
    return rc;
}

