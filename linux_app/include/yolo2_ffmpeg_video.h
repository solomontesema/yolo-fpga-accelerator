/**
 * YOLOv2 Linux App - ffmpeg-based video file reader
 *
 * Spawns `ffmpeg` and reads fixed-size RGB24 frames from stdout.
 */

#ifndef YOLO2_FFMPEG_VIDEO_H
#define YOLO2_FFMPEG_VIDEO_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int fd;        // read end (raw rgb24 frames)
    pid_t pid;     // ffmpeg process pid
    int width;
    int height;
    int fps;
} yolo2_ffmpeg_video_t;

int yolo2_ffmpeg_video_open(yolo2_ffmpeg_video_t *v, const char *path, int width, int height, int fps);

// Returns: 1 on success, 0 on EOF, -1 on error.
int yolo2_ffmpeg_video_read_frame(yolo2_ffmpeg_video_t *v, uint8_t *rgb, size_t rgb_size);

int yolo2_ffmpeg_video_close(yolo2_ffmpeg_video_t *v);

#ifdef __cplusplus
}
#endif

#endif /* YOLO2_FFMPEG_VIDEO_H */

