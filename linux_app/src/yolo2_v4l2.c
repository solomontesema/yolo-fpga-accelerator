/**
 * YOLOv2 Linux App - Minimal V4L2 capture implementation
 */

#include "yolo2_v4l2.h"
#include "yolo2_log.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define STBI_NO_THREAD_LOCALS
#include "stb_image.h"

static int xioctl(int fd, unsigned long request, void *arg)
{
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

const char *yolo2_v4l2_pixfmt_name(uint32_t pixfmt)
{
    switch (pixfmt) {
        case V4L2_PIX_FMT_MJPEG:
            return "mjpeg";
        case V4L2_PIX_FMT_YUYV:
            return "yuyv";
        default:
            return "unknown";
    }
}

static int yolo2_try_set_format(
    yolo2_v4l2_camera_t *cam,
    int width,
    int height,
    uint32_t pixfmt)
{
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = (uint32_t)width;
    fmt.fmt.pix.height = (uint32_t)height;
    fmt.fmt.pix.pixelformat = pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    if (xioctl(cam->fd, VIDIOC_S_FMT, &fmt) == -1) {
        return -1;
    }

    if (fmt.fmt.pix.pixelformat != pixfmt) {
        return -1;
    }

    cam->width = (int)fmt.fmt.pix.width;
    cam->height = (int)fmt.fmt.pix.height;
    cam->pixfmt = fmt.fmt.pix.pixelformat;
    return 0;
}

int yolo2_v4l2_open(
    yolo2_v4l2_camera_t *cam,
    const char *device,
    int width,
    int height,
    int fps,
    yolo2_v4l2_format_t requested_format)
{
    if (!cam || !device || !device[0]) {
        return -1;
    }

    memset(cam, 0, sizeof(*cam));
    cam->fd = -1;

    cam->fd = open(device, O_RDWR);
    if (cam->fd < 0) {
        fprintf(stderr, "ERROR: Failed to open camera device %s: %s\n", device, strerror(errno));
        return -1;
    }

    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (xioctl(cam->fd, VIDIOC_QUERYCAP, &cap) == -1) {
        fprintf(stderr, "ERROR: VIDIOC_QUERYCAP failed: %s\n", strerror(errno));
        yolo2_v4l2_close(cam);
        return -1;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "ERROR: %s is not a V4L2 video capture device\n", device);
        yolo2_v4l2_close(cam);
        return -1;
    }
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "ERROR: %s does not support V4L2 streaming I/O\n", device);
        yolo2_v4l2_close(cam);
        return -1;
    }

    // Try requested format, with required MJPEG -> YUYV fallback behavior.
    uint32_t primary = (requested_format == YOLO2_V4L2_FMT_YUYV) ? V4L2_PIX_FMT_YUYV : V4L2_PIX_FMT_MJPEG;
    uint32_t fallback = (primary == V4L2_PIX_FMT_MJPEG) ? V4L2_PIX_FMT_YUYV : V4L2_PIX_FMT_MJPEG;

    if (yolo2_try_set_format(cam, width, height, primary) != 0) {
        YOLO2_LOG_INFO("Camera format %s not supported, trying %s...\n",
                       yolo2_v4l2_pixfmt_name(primary), yolo2_v4l2_pixfmt_name(fallback));
        if (yolo2_try_set_format(cam, width, height, fallback) != 0) {
            fprintf(stderr,
                    "ERROR: Failed to set camera format (%s or %s) at %dx%d\n",
                    yolo2_v4l2_pixfmt_name(primary),
                    yolo2_v4l2_pixfmt_name(fallback),
                    width,
                    height);
            yolo2_v4l2_close(cam);
            return -1;
        }
    }

    cam->fps = fps;
    struct v4l2_streamparm parm;
    memset(&parm, 0, sizeof(parm));
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = (uint32_t)((fps > 0) ? fps : 30);
    if (xioctl(cam->fd, VIDIOC_S_PARM, &parm) == -1) {
        YOLO2_LOG_INFO("WARNING: Failed to set FPS to %d: %s\n", fps, strerror(errno));
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(cam->fd, VIDIOC_REQBUFS, &req) == -1) {
        fprintf(stderr, "ERROR: VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        yolo2_v4l2_close(cam);
        return -1;
    }
    if (req.count < 2) {
        fprintf(stderr, "ERROR: Insufficient V4L2 buffers (count=%u)\n", req.count);
        yolo2_v4l2_close(cam);
        return -1;
    }

    cam->buffers = (yolo2_v4l2_mmap_buf_t *)calloc(req.count, sizeof(*cam->buffers));
    if (!cam->buffers) {
        fprintf(stderr, "ERROR: Out of memory allocating V4L2 buffers\n");
        yolo2_v4l2_close(cam);
        return -1;
    }
    cam->num_buffers = req.count;

    for (unsigned int i = 0; i < cam->num_buffers; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(cam->fd, VIDIOC_QUERYBUF, &buf) == -1) {
            fprintf(stderr, "ERROR: VIDIOC_QUERYBUF failed: %s\n", strerror(errno));
            yolo2_v4l2_close(cam);
            return -1;
        }

        cam->buffers[i].length = buf.length;
        cam->buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam->fd, buf.m.offset);
        if (cam->buffers[i].start == MAP_FAILED) {
            fprintf(stderr, "ERROR: mmap failed: %s\n", strerror(errno));
            yolo2_v4l2_close(cam);
            return -1;
        }
    }

    for (unsigned int i = 0; i < cam->num_buffers; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(cam->fd, VIDIOC_QBUF, &buf) == -1) {
            fprintf(stderr, "ERROR: VIDIOC_QBUF failed: %s\n", strerror(errno));
            yolo2_v4l2_close(cam);
            return -1;
        }
    }

    YOLO2_LOG_INFO("Camera opened: %s (%dx%d @ ~%dfps, fmt=%s)\n",
                   device, cam->width, cam->height, cam->fps, yolo2_v4l2_pixfmt_name(cam->pixfmt));
    return 0;
}

int yolo2_v4l2_start(yolo2_v4l2_camera_t *cam)
{
    if (!cam || cam->fd < 0) return -1;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cam->fd, VIDIOC_STREAMON, &type) == -1) {
        fprintf(stderr, "ERROR: VIDIOC_STREAMON failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

int yolo2_v4l2_stop(yolo2_v4l2_camera_t *cam)
{
    if (!cam || cam->fd < 0) return -1;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(cam->fd, VIDIOC_STREAMOFF, &type) == -1) {
        YOLO2_LOG_INFO("WARNING: VIDIOC_STREAMOFF failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

void yolo2_v4l2_close(yolo2_v4l2_camera_t *cam)
{
    if (!cam) return;
    if (cam->buffers) {
        for (unsigned int i = 0; i < cam->num_buffers; ++i) {
            if (cam->buffers[i].start && cam->buffers[i].start != MAP_FAILED) {
                munmap(cam->buffers[i].start, cam->buffers[i].length);
            }
        }
        free(cam->buffers);
    }
    cam->buffers = NULL;
    cam->num_buffers = 0;
    if (cam->fd >= 0) {
        close(cam->fd);
    }
    cam->fd = -1;
}

int yolo2_v4l2_dequeue(yolo2_v4l2_camera_t *cam, yolo2_v4l2_frame_t *frame)
{
    if (!cam || cam->fd < 0 || !frame) return -1;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(cam->fd, VIDIOC_DQBUF, &buf) == -1) {
        if (errno == EAGAIN) {
            return 0;
        }
        fprintf(stderr, "ERROR: VIDIOC_DQBUF failed: %s\n", strerror(errno));
        return -1;
    }

    if (buf.index >= cam->num_buffers) {
        fprintf(stderr, "ERROR: V4L2 returned out-of-range buffer index %u\n", buf.index);
        return -1;
    }

    frame->data = (const uint8_t *)cam->buffers[buf.index].start;
    frame->size = (size_t)buf.bytesused;
    frame->index = buf.index;
    return 1;
}

int yolo2_v4l2_enqueue(yolo2_v4l2_camera_t *cam, const yolo2_v4l2_frame_t *frame)
{
    if (!cam || cam->fd < 0 || !frame) return -1;

    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = frame->index;

    if (xioctl(cam->fd, VIDIOC_QBUF, &buf) == -1) {
        fprintf(stderr, "ERROR: VIDIOC_QBUF failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

int yolo2_decode_mjpeg_to_rgb24(const uint8_t *mjpeg, size_t mjpeg_size, uint8_t *rgb, int width, int height)
{
    if (!mjpeg || mjpeg_size == 0 || !rgb || width <= 0 || height <= 0) {
        return -1;
    }

    int w = 0, h = 0, c = 0;
    unsigned char *decoded = stbi_load_from_memory(mjpeg, (int)mjpeg_size, &w, &h, &c, 3);
    if (!decoded) {
        fprintf(stderr, "ERROR: MJPEG decode failed: %s\n", stbi_failure_reason());
        return -1;
    }

    if (w != width || h != height) {
        fprintf(stderr,
                "ERROR: MJPEG decoded size %dx%d does not match expected %dx%d\n",
                w,
                h,
                width,
                height);
        stbi_image_free(decoded);
        return -1;
    }

    memcpy(rgb, decoded, (size_t)width * (size_t)height * 3u);
    stbi_image_free(decoded);
    return 0;
}

static uint8_t clamp_u8(int v)
{
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

void yolo2_yuyv_to_rgb24(const uint8_t *yuyv, uint8_t *rgb, int width, int height)
{
    if (!yuyv || !rgb || width <= 0 || height <= 0) {
        return;
    }

    const int num_pixels = width * height;
    const int num_pairs = num_pixels / 2;

    const uint8_t *src = yuyv;
    uint8_t *dst = rgb;

    for (int i = 0; i < num_pairs; ++i) {
        const int y0 = (int)src[0];
        const int u = (int)src[1];
        const int y1 = (int)src[2];
        const int v = (int)src[3];
        src += 4;

        const int c0 = y0 - 16;
        const int c1 = y1 - 16;
        const int d = u - 128;
        const int e = v - 128;

        // Integer approximation (BT.601)
        const int r0 = (298 * c0 + 409 * e + 128) >> 8;
        const int g0 = (298 * c0 - 100 * d - 208 * e + 128) >> 8;
        const int b0 = (298 * c0 + 516 * d + 128) >> 8;

        const int r1 = (298 * c1 + 409 * e + 128) >> 8;
        const int g1 = (298 * c1 - 100 * d - 208 * e + 128) >> 8;
        const int b1 = (298 * c1 + 516 * d + 128) >> 8;

        dst[0] = clamp_u8(r0);
        dst[1] = clamp_u8(g0);
        dst[2] = clamp_u8(b0);
        dst[3] = clamp_u8(r1);
        dst[4] = clamp_u8(g1);
        dst[5] = clamp_u8(b1);
        dst += 6;
    }
}

