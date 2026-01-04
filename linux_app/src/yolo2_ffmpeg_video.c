/**
 * YOLOv2 Linux App - ffmpeg video reader
 */

#include "yolo2_ffmpeg_video.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int is_executable(const char *path)
{
    return (path && path[0] && access(path, X_OK) == 0);
}

static int find_in_path(const char *exe, char *out, size_t out_size)
{
    if (!exe || !exe[0] || !out || out_size == 0) return -1;

    const char *path_env = getenv("PATH");
    if (!path_env || !path_env[0]) return -1;

    char *copy = strdup(path_env);
    if (!copy) return -1;

    int rc = -1;
    for (char *tok = strtok(copy, ":"); tok; tok = strtok(NULL, ":")) {
        if (!tok[0]) continue;
        char cand[1024];
        snprintf(cand, sizeof(cand), "%s/%s", tok, exe);
        if (is_executable(cand)) {
            snprintf(out, out_size, "%s", cand);
            rc = 0;
            break;
        }
    }

    free(copy);
    return rc;
}

static ssize_t read_full(int fd, void *buf, size_t count)
{
    uint8_t *p = (uint8_t *)buf;
    size_t done = 0;
    while (done < count) {
        ssize_t n = read(fd, p + done, count - done);
        if (n == 0) {
            return (ssize_t)done; // EOF
        }
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        done += (size_t)n;
    }
    return (ssize_t)done;
}

int yolo2_ffmpeg_video_open(yolo2_ffmpeg_video_t *v, const char *path, int width, int height, int fps)
{
    if (!v || !path || !path[0] || width <= 0 || height <= 0 || fps <= 0) {
        return -1;
    }

    memset(v, 0, sizeof(*v));
    v->fd = -1;
    v->pid = -1;
    v->width = width;
    v->height = height;
    v->fps = fps;

    char ffmpeg_path[1024];
    if (find_in_path("ffmpeg", ffmpeg_path, sizeof(ffmpeg_path)) != 0) {
        fprintf(stderr,
                "ERROR: ffmpeg not found in PATH.\n"
                "       Install on KV260 with: sudo apt-get update && sudo apt-get install -y ffmpeg\n");
        return -1;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        fprintf(stderr, "ERROR: pipe() failed: %s\n", strerror(errno));
        return -1;
    }

    char vf[256];
    snprintf(
        vf,
        sizeof(vf),
        "scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2,fps=%d",
        width,
        height,
        width,
        height,
        fps);

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "ERROR: fork() failed: %s\n", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }

    if (pid == 0) {
        // child: stdout -> pipe write end
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);

        // stdin -> /dev/null (avoid blocking on stdin)
        int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) {
            dup2(devnull, STDIN_FILENO);
            close(devnull);
        }

        char fps_str[32];
        snprintf(fps_str, sizeof(fps_str), "%d", fps);

        char *argv[] = {
            (char *)ffmpeg_path,
            (char *)"-hide_banner",
            (char *)"-loglevel",
            (char *)"error",
            (char *)"-nostdin",
            (char *)"-i",
            (char *)path,
            (char *)"-vf",
            (char *)vf,
            (char *)"-r",
            (char *)fps_str,
            (char *)"-f",
            (char *)"rawvideo",
            (char *)"-pix_fmt",
            (char *)"rgb24",
            (char *)"-",
            NULL,
        };

        execv(ffmpeg_path, argv);
        _exit(127);
    }

    // parent: keep read end
    close(pipefd[1]);
    v->fd = pipefd[0];
    v->pid = pid;
    return 0;
}

int yolo2_ffmpeg_video_read_frame(yolo2_ffmpeg_video_t *v, uint8_t *rgb, size_t rgb_size)
{
    if (!v || v->fd < 0 || !rgb || rgb_size == 0) {
        return -1;
    }

    const size_t expected = (size_t)v->width * (size_t)v->height * 3u;
    if (rgb_size < expected) {
        return -1;
    }

    const ssize_t n = read_full(v->fd, rgb, expected);
    if (n < 0) {
        fprintf(stderr, "ERROR: read() from ffmpeg failed: %s\n", strerror(errno));
        return -1;
    }
    if ((size_t)n < expected) {
        return 0; // EOF
    }
    return 1;
}

int yolo2_ffmpeg_video_close(yolo2_ffmpeg_video_t *v)
{
    if (!v) return -1;

    if (v->fd >= 0) {
        close(v->fd);
        v->fd = -1;
    }

    int status = 0;
    if (v->pid > 0) {
        // Best-effort wait.
        (void)waitpid(v->pid, &status, 0);
        v->pid = -1;
    }

    return 0;
}
