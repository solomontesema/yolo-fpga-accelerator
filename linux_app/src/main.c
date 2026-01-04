/**
 * YOLOv2 FPGA Accelerator - Linux Main Application
 * 
 * This application runs object detection using the custom HLS accelerator
 * on the AMD/Xilinx KV260 board.
 * 
 * Usage: sudo ./yolo2_linux [options]
 *   -i <image>    Input image path (default: /home/ubuntu/test_images/dog.jpg)
 *   -w <dir>      Weights directory (default: /home/ubuntu/weights)
 *   -c <config>   Network config file (default: /home/ubuntu/config/yolov2.cfg)
 *   -l <labels>   Labels file (default: /home/ubuntu/config/coco.names)
 *   -t <thresh>   Detection threshold (default: 0.24)
 *   -n <nms>      NMS threshold (default: 0.45)
 *   --camera <dev>   Camera device (e.g., /dev/video0)
 *   --video <path>   Video file path (decoded via ffmpeg)
 *   -h            Show this help
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <errno.h>
#include <linux/videodev2.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "yolo2_config.h"
#include "yolo2_accel_linux.h"
#include "dma_buffer_manager.h"
#include "yolo2_inference.h"
#include "yolo2_network.h"
#include "yolo2_image_loader.h"
#include "yolo2_draw.h"
#include "yolo2_v4l2.h"
#include "yolo2_ffmpeg_video.h"
#include "yolo2_mjpeg_streamer.h"
#include "yolo2_postprocess.h"
#include "yolo2_labels.h"
#include "file_loader.h"
#include "yolo2_log.h"

// Default paths
static char weights_dir[512] = "/home/ubuntu/weights";
static char config_path[512] = "/home/ubuntu/config/yolov2.cfg";
static char labels_path[512] = "/home/ubuntu/config/coco.names";
static char image_path[512] = "/home/ubuntu/test_images/dog.jpg";
static float det_thresh = 0.24f;
static float nms_thresh = 0.45f;

// Streaming inputs (mutually exclusive with image mode)
static char camera_device[256] = "";
static char video_path[512] = "";

// Streaming controls
static int max_frames = -1;   // per inference runs; -1 = default per mode
static int infer_every = 1;   // run inference every N frames

// Camera controls
static int cam_width = 640;
static int cam_height = 480;
static int cam_fps = 30;
static yolo2_v4l2_format_t cam_format = YOLO2_V4L2_FMT_MJPEG;

// Video controls
static int video_width = 640;
static int video_height = 480;
static int video_fps = 30;

// Headless visual output
static char save_annotated_dir[512] = "";
static char output_json_path[512] = "";

// Streaming output (MJPEG over HTTP)
static char stream_mjpeg_bind[64] = "0.0.0.0";
static int stream_mjpeg_port = 0;     // 0 = disabled
static int stream_mjpeg_quality = 80; // JPEG quality 1..100
static int stream_mjpeg_fps = 4;      // send rate for MJPEG (keeps VLC alive even when inference is slow)

typedef enum {
    INPUT_MODE_IMAGE = 0,
    INPUT_MODE_CAMERA = 1,
    INPUT_MODE_VIDEO = 2,
} input_mode_t;

static int mkdir_p(const char *path)
{
    if (!path || !path[0]) {
        return -1;
    }

    char tmp[PATH_MAX];
    const size_t len = strnlen(path, sizeof(tmp));
    if (len == 0 || len >= sizeof(tmp)) {
        return -1;
    }

    memcpy(tmp, path, len);
    tmp[len] = '\0';

    // Strip trailing slash
    if (len > 1 && tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }

    for (char *p = tmp + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
            return -1;
        }
        *p = '/';
    }

    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static yolo2_v4l2_format_t parse_cam_format(const char *s)
{
    if (!s) return YOLO2_V4L2_FMT_MJPEG;
    if (strcmp(s, "mjpeg") == 0) return YOLO2_V4L2_FMT_MJPEG;
    if (strcmp(s, "yuyv") == 0) return YOLO2_V4L2_FMT_YUYV;
    return YOLO2_V4L2_FMT_MJPEG;
}

static int parse_int(const char *s, int *out)
{
    if (!s || !out) return -1;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || (end && *end != '\0')) {
        return -1;
    }
    if (v > INT_MAX || v < INT_MIN) {
        return -1;
    }
    *out = (int)v;
    return 0;
}

static int parse_bind_port(const char *s, char *bind_out, size_t bind_out_size, int *port_out)
{
    if (!s || !s[0] || !bind_out || bind_out_size == 0 || !port_out) {
        return -1;
    }

    // Accept a plain port number: "8080"
    {
        char *end = NULL;
        long v = strtol(s, &end, 10);
        if (end != s && end && *end == '\0') {
            if (v <= 0 || v > 65535) return -1;
            snprintf(bind_out, bind_out_size, "0.0.0.0");
            *port_out = (int)v;
            return 0;
        }
    }

    // Accept "bind:port" (IPv4/hostname; IPv6 not supported here).
    const char *colon = strrchr(s, ':');
    if (!colon) {
        return -1;
    }

    int port = 0;
    if (parse_int(colon + 1, &port) != 0 || port <= 0 || port > 65535) {
        return -1;
    }

    const size_t host_len = (size_t)(colon - s);
    if (host_len == 0) {
        snprintf(bind_out, bind_out_size, "0.0.0.0");
    } else {
        if (host_len >= bind_out_size) return -1;
        memcpy(bind_out, s, host_len);
        bind_out[host_len] = '\0';
    }

    *port_out = port;
    return 0;
}

static void rgb24_to_chw_float(const uint8_t *rgb, float *chw, int width, int height)
{
    const size_t plane = (size_t)width * (size_t)height;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = (size_t)y * (size_t)width + (size_t)x;
            const size_t src = idx * 3u;
            chw[idx] = (float)rgb[src + 0] / 255.0f;
            chw[plane + idx] = (float)rgb[src + 1] / 255.0f;
            chw[2u * plane + idx] = (float)rgb[src + 2] / 255.0f;
        }
    }
}

static void json_write_escaped(FILE *fp, const char *s)
{
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)s; p && *p; ++p) {
        const unsigned char c = *p;
        switch (c) {
            case '"':
                fputs("\\\"", fp);
                break;
            case '\\':
                fputs("\\\\", fp);
                break;
            case '\b':
                fputs("\\b", fp);
                break;
            case '\f':
                fputs("\\f", fp);
                break;
            case '\n':
                fputs("\\n", fp);
                break;
            case '\r':
                fputs("\\r", fp);
                break;
            case '\t':
                fputs("\\t", fp);
                break;
            default:
                if (c < 0x20) {
                    fprintf(fp, "\\u%04x", (unsigned)c);
                } else {
                    fputc((int)c, fp);
                }
                break;
        }
    }
    fputc('"', fp);
}

static void print_usage(const char *prog_name) {
    printf("YOLOv2 FPGA Accelerator - Linux Application\n");
    printf("\n");
    printf("Usage: sudo %s [options]\n", prog_name);
    printf("\n");
    printf("Options:\n");
    printf("  -i <image>    Input image path (default: %s)\n", image_path);
    printf("  --camera <dev>           Camera device (e.g., /dev/video0)\n");
    printf("  --video <path>           Video file path (decoded via ffmpeg)\n");
    printf("  -w <dir>      Weights directory (default: %s)\n", weights_dir);
    printf("  -c <config>   Network config file (default: %s)\n", config_path);
    printf("  -l <labels>   Labels file (default: %s)\n", labels_path);
    printf("  -t <thresh>   Detection threshold (default: %.2f)\n", det_thresh);
    printf("  -n <nms>      NMS threshold (default: %.2f)\n", nms_thresh);
    printf("  -v <level>    Verbosity 0..3 (overrides YOLO2_VERBOSE)\n");
    printf("  --max-frames <N>          Stop after N inference runs (0 = infinite)\n");
    printf("  --infer-every <N>         Run inference every N frames (default: 1)\n");
    printf("  --cam-width <W>           Camera width (default: %d)\n", cam_width);
    printf("  --cam-height <H>          Camera height (default: %d)\n", cam_height);
    printf("  --cam-fps <fps>           Camera FPS (default: %d)\n", cam_fps);
    printf("  --cam-format mjpeg|yuyv   Camera format (default: mjpeg; falls back to yuyv)\n");
    printf("  --video-width <W>         Video output width (default: %d)\n", video_width);
    printf("  --video-height <H>        Video output height (default: %d)\n", video_height);
    printf("  --video-fps <fps>         Video output FPS (default: %d)\n", video_fps);
    printf("  --save-annotated-dir <d>  Save annotated PNG frames to directory\n");
    printf("  --output-json <path>      Write detections JSONL (one object per inference)\n");
    printf("  --stream-mjpeg <p|b:p>    Stream annotated frames as MJPEG over HTTP (e.g. 8080 or 0.0.0.0:8080)\n");
    printf("  --stream-mjpeg-quality <q> JPEG quality 1..100 (default: %d)\n", stream_mjpeg_quality);
    printf("  --stream-mjpeg-fps <fps>  MJPEG send rate (default: %d)\n", stream_mjpeg_fps);
    printf("  -h            Show this help\n");
    printf("\n");
    printf("Notes:\n");
    printf("  - Must run with sudo for /dev/mem access\n");
    printf("  - Requires udmabuf kernel module for DMA buffers\n");
    printf("  - Requires FPGA bitstream to be loaded\n");
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static int dump_float_array_text(const char *path, const float *data, size_t count)
{
    if (!path || !path[0] || !data) {
        return -1;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open dump file %s: %s\n", path, strerror(errno));
        return -1;
    }

    for (size_t i = 0; i < count; ++i) {
        // One value per line for easy diffing.
        fprintf(fp, "%.9g\n", data[i]);
    }

    fclose(fp);
    YOLO2_LOG_INFO("  Dumped %zu floats to %s\n", count, path);
    return 0;
}

int main(int argc, char *argv[]) {
    int opt;
    int result = 1;
    double start_time, end_time;

    input_mode_t input_mode = INPUT_MODE_IMAGE;
    int image_arg_provided = 0;

    enum {
        OPT_CAMERA = 1000,
        OPT_VIDEO,
        OPT_MAX_FRAMES,
        OPT_INFER_EVERY,
        OPT_CAM_WIDTH,
        OPT_CAM_HEIGHT,
        OPT_CAM_FPS,
        OPT_CAM_FORMAT,
        OPT_VIDEO_WIDTH,
        OPT_VIDEO_HEIGHT,
        OPT_VIDEO_FPS,
        OPT_SAVE_ANNOTATED_DIR,
        OPT_OUTPUT_JSON,
        OPT_STREAM_MJPEG,
        OPT_STREAM_MJPEG_QUALITY,
        OPT_STREAM_MJPEG_FPS,
    };

    static const struct option long_opts[] = {
        {"camera", required_argument, NULL, OPT_CAMERA},
        {"video", required_argument, NULL, OPT_VIDEO},
        {"max-frames", required_argument, NULL, OPT_MAX_FRAMES},
        {"infer-every", required_argument, NULL, OPT_INFER_EVERY},
        {"cam-width", required_argument, NULL, OPT_CAM_WIDTH},
        {"cam-height", required_argument, NULL, OPT_CAM_HEIGHT},
        {"cam-fps", required_argument, NULL, OPT_CAM_FPS},
        {"cam-format", required_argument, NULL, OPT_CAM_FORMAT},
        {"video-width", required_argument, NULL, OPT_VIDEO_WIDTH},
        {"video-height", required_argument, NULL, OPT_VIDEO_HEIGHT},
        {"video-fps", required_argument, NULL, OPT_VIDEO_FPS},
        {"save-annotated-dir", required_argument, NULL, OPT_SAVE_ANNOTATED_DIR},
        {"output-json", required_argument, NULL, OPT_OUTPUT_JSON},
        {"stream-mjpeg", required_argument, NULL, OPT_STREAM_MJPEG},
        {"stream-mjpeg-quality", required_argument, NULL, OPT_STREAM_MJPEG_QUALITY},
        {"stream-mjpeg-fps", required_argument, NULL, OPT_STREAM_MJPEG_FPS},
        {NULL, 0, NULL, 0},
    };
    
    // Parse command line arguments
    while ((opt = getopt_long(argc, argv, "i:w:c:l:t:n:v:h", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'i':
                strncpy(image_path, optarg, sizeof(image_path) - 1);
                image_arg_provided = 1;
                break;
            case 'w':
                strncpy(weights_dir, optarg, sizeof(weights_dir) - 1);
                break;
            case 'c':
                strncpy(config_path, optarg, sizeof(config_path) - 1);
                break;
            case 'l':
                strncpy(labels_path, optarg, sizeof(labels_path) - 1);
                break;
            case 't':
                det_thresh = atof(optarg);
                break;
            case 'n':
                nms_thresh = atof(optarg);
                break;
            case 'v':
                yolo2_set_verbosity(atoi(optarg));
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
            case OPT_CAMERA:
                strncpy(camera_device, optarg, sizeof(camera_device) - 1);
                break;
            case OPT_VIDEO:
                strncpy(video_path, optarg, sizeof(video_path) - 1);
                break;
            case OPT_MAX_FRAMES:
                if (parse_int(optarg, &max_frames) != 0 || max_frames < 0) {
                    fprintf(stderr, "ERROR: Invalid --max-frames value: %s\n", optarg);
                    return 1;
                }
                break;
            case OPT_INFER_EVERY:
                if (parse_int(optarg, &infer_every) != 0 || infer_every <= 0) {
                    fprintf(stderr, "ERROR: Invalid --infer-every value: %s\n", optarg);
                    return 1;
                }
                break;
            case OPT_CAM_WIDTH:
                if (parse_int(optarg, &cam_width) != 0 || cam_width <= 0) {
                    fprintf(stderr, "ERROR: Invalid --cam-width value: %s\n", optarg);
                    return 1;
                }
                break;
            case OPT_CAM_HEIGHT:
                if (parse_int(optarg, &cam_height) != 0 || cam_height <= 0) {
                    fprintf(stderr, "ERROR: Invalid --cam-height value: %s\n", optarg);
                    return 1;
                }
                break;
            case OPT_CAM_FPS:
                if (parse_int(optarg, &cam_fps) != 0 || cam_fps <= 0) {
                    fprintf(stderr, "ERROR: Invalid --cam-fps value: %s\n", optarg);
                    return 1;
                }
                break;
            case OPT_CAM_FORMAT:
                cam_format = parse_cam_format(optarg);
                break;
            case OPT_VIDEO_WIDTH:
                if (parse_int(optarg, &video_width) != 0 || video_width <= 0) {
                    fprintf(stderr, "ERROR: Invalid --video-width value: %s\n", optarg);
                    return 1;
                }
                break;
            case OPT_VIDEO_HEIGHT:
                if (parse_int(optarg, &video_height) != 0 || video_height <= 0) {
                    fprintf(stderr, "ERROR: Invalid --video-height value: %s\n", optarg);
                    return 1;
                }
                break;
            case OPT_VIDEO_FPS:
                if (parse_int(optarg, &video_fps) != 0 || video_fps <= 0) {
                    fprintf(stderr, "ERROR: Invalid --video-fps value: %s\n", optarg);
                    return 1;
                }
                break;
            case OPT_SAVE_ANNOTATED_DIR:
                strncpy(save_annotated_dir, optarg, sizeof(save_annotated_dir) - 1);
                break;
            case OPT_OUTPUT_JSON:
                strncpy(output_json_path, optarg, sizeof(output_json_path) - 1);
                break;
            case OPT_STREAM_MJPEG: {
                int port = 0;
                char bind[64];
                if (parse_bind_port(optarg, bind, sizeof(bind), &port) != 0) {
                    fprintf(stderr, "ERROR: Invalid --stream-mjpeg value (expected <port> or <bind>:<port>): %s\n", optarg);
                    return 1;
                }
                snprintf(stream_mjpeg_bind, sizeof(stream_mjpeg_bind), "%s", bind);
                stream_mjpeg_port = port;
                break;
            }
            case OPT_STREAM_MJPEG_QUALITY:
                if (parse_int(optarg, &stream_mjpeg_quality) != 0 || stream_mjpeg_quality < 1 || stream_mjpeg_quality > 100) {
                    fprintf(stderr, "ERROR: Invalid --stream-mjpeg-quality (1..100): %s\n", optarg);
                    return 1;
                }
                break;
            case OPT_STREAM_MJPEG_FPS:
                if (parse_int(optarg, &stream_mjpeg_fps) != 0 || stream_mjpeg_fps < 1 || stream_mjpeg_fps > 30) {
                    fprintf(stderr, "ERROR: Invalid --stream-mjpeg-fps (1..30): %s\n", optarg);
                    return 1;
                }
                break;
        }
    }

    if (camera_device[0] && video_path[0]) {
        fprintf(stderr, "ERROR: --camera and --video are mutually exclusive\n");
        return 1;
    }
    if (camera_device[0]) {
        input_mode = INPUT_MODE_CAMERA;
    } else if (video_path[0]) {
        input_mode = INPUT_MODE_VIDEO;
    } else {
        input_mode = INPUT_MODE_IMAGE;
    }

    if (input_mode != INPUT_MODE_IMAGE && image_arg_provided) {
        fprintf(stderr, "ERROR: -i cannot be used with --camera/--video\n");
        return 1;
    }

    // Apply per-mode defaults (only if user did not override).
    if (max_frames < 0) {
        if (input_mode == INPUT_MODE_CAMERA) {
            max_frames = 0; // infinite
        } else if (input_mode == INPUT_MODE_VIDEO) {
            max_frames = 100;
        } else {
            max_frames = 1;
        }
    }
    
    YOLO2_LOG_INFO("\n");
    YOLO2_LOG_INFO("========================================\n");
    YOLO2_LOG_INFO("YOLOv2 FPGA Accelerator - Linux\n");
    YOLO2_LOG_INFO("========================================\n");
    YOLO2_LOG_INFO("\n");
    YOLO2_LOG_INFO("Configuration:\n");
    if (input_mode == INPUT_MODE_CAMERA) {
        YOLO2_LOG_INFO("  Camera:     %s\n", camera_device);
        YOLO2_LOG_INFO("  Cam size:   %dx%d @ %dfps\n", cam_width, cam_height, cam_fps);
        YOLO2_LOG_INFO("  Cam format: %s\n", (cam_format == YOLO2_V4L2_FMT_YUYV) ? "yuyv" : "mjpeg");
        YOLO2_LOG_INFO("  Max frames: %d (inference runs, 0=infinite)\n", max_frames);
        YOLO2_LOG_INFO("  Infer every:%d\n", infer_every);
    } else if (input_mode == INPUT_MODE_VIDEO) {
        YOLO2_LOG_INFO("  Video:      %s\n", video_path);
        YOLO2_LOG_INFO("  Vid size:   %dx%d @ %dfps\n", video_width, video_height, video_fps);
        YOLO2_LOG_INFO("  Max frames: %d (inference runs, 0=infinite)\n", max_frames);
        YOLO2_LOG_INFO("  Infer every:%d\n", infer_every);
    } else {
        YOLO2_LOG_INFO("  Image:      %s\n", image_path);
    }
    YOLO2_LOG_INFO("  Weights:    %s\n", weights_dir);
    YOLO2_LOG_INFO("  Config:     %s\n", config_path);
    YOLO2_LOG_INFO("  Labels:     %s\n", labels_path);
    YOLO2_LOG_INFO("  Threshold:  %.2f\n", det_thresh);
    YOLO2_LOG_INFO("  NMS:        %.2f\n", nms_thresh);
    YOLO2_LOG_INFO("  Verbosity:  %d\n", yolo2_get_verbosity());
    if (save_annotated_dir[0]) {
        YOLO2_LOG_INFO("  Save dir:   %s\n", save_annotated_dir);
    }
    if (output_json_path[0]) {
        YOLO2_LOG_INFO("  JSONL:      %s\n", output_json_path);
    }
    if (stream_mjpeg_port > 0) {
        YOLO2_LOG_INFO("  MJPEG:      http://<kv260-ip>:%d/ (bind %s, send %dfps)\n",
                       stream_mjpeg_port,
                       stream_mjpeg_bind,
                       stream_mjpeg_fps);
    }
    YOLO2_LOG_INFO("\n");
    
    // Build weight file paths
    char weights_file[512], bias_file[512];
    char weight_q_file[512], bias_q_file[512], iofm_q_file[512];
    snprintf(weights_file, sizeof(weights_file), "%s/weights_reorg_int16.bin", weights_dir);
    snprintf(bias_file, sizeof(bias_file), "%s/bias_int16.bin", weights_dir);
    snprintf(weight_q_file, sizeof(weight_q_file), "%s/weight_int16_Q.bin", weights_dir);
    snprintf(bias_q_file, sizeof(bias_q_file), "%s/bias_int16_Q.bin", weights_dir);
    snprintf(iofm_q_file, sizeof(iofm_q_file), "%s/iofm_Q.bin", weights_dir);
    
    yolo2_inference_context_t ctx;
    void *weights_data = NULL, *bias_data = NULL;
    size_t weights_size = 0, bias_size = 0;
    float *input_image = NULL;
    char **labels = NULL;
    int num_labels = 0;
    FILE *json_fp = NULL;
    yolo2_mjpeg_streamer_t *mjpeg_stream = NULL;
    
    // Initialize inference context
    yolo2_inference_init(&ctx);
    
    // Step 1: Initialize accelerator driver
    YOLO2_LOG_INFO("[1/8] Initializing accelerator driver...\n");
    result = yolo2_accel_init();
    if (result != YOLO2_SUCCESS) {
        fprintf(stderr, "ERROR: Accelerator initialization failed: %d\n", result);
        goto cleanup;
    }
    YOLO2_LOG_INFO("      Accelerator driver initialized OK\n\n");
    
    // Step 2: Initialize DMA buffer manager
    YOLO2_LOG_INFO("[2/8] Initializing DMA buffer manager...\n");
    result = dma_buffer_init();
    if (result != 0) {
        fprintf(stderr, "ERROR: DMA buffer initialization failed\n");
        goto cleanup;
    }
    YOLO2_LOG_INFO("      DMA buffer manager initialized OK\n\n");
    
    // Step 3: Load weights from filesystem
    YOLO2_LOG_INFO("[3/8] Loading weights...\n");
    result = load_weights(weights_file, &weights_data, &weights_size);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to load weights from %s\n", weights_file);
        goto cleanup;
    }
    
    result = load_bias(bias_file, &bias_data, &bias_size);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to load bias from %s\n", bias_file);
        goto cleanup;
    }
    YOLO2_LOG_INFO("      Weights: %zu bytes, Bias: %zu bytes\n\n", weights_size, bias_size);
    
    // Step 4: Load Q values (INT16 mode)
    YOLO2_LOG_INFO("[4/8] Loading Q values...\n");
    result = load_q_values(weight_q_file, &ctx.weight_q, &ctx.weight_q_size);
    if (result != 0) {
        YOLO2_LOG_INFO("      WARNING: Weight Q values not found (using defaults)\n");
    }
    
    result = load_q_values(bias_q_file, &ctx.bias_q, &ctx.bias_q_size);
    if (result != 0) {
        YOLO2_LOG_INFO("      WARNING: Bias Q values not found (using defaults)\n");
    }
    
    result = load_q_values(iofm_q_file, &ctx.act_q, &ctx.act_q_size);
    if (result != 0) {
        YOLO2_LOG_INFO("      WARNING: Activation Q values not found (using defaults)\n");
    }
    
    if (ctx.act_q && ctx.act_q_size > 0) {
        ctx.current_Qa = ctx.act_q[0];
        YOLO2_LOG_INFO("      Q values loaded OK\n");
    }
    YOLO2_LOG_INFO("\n");
    
    // Step 5: Allocate DMA buffers
    YOLO2_LOG_INFO("[5/8] Allocating DMA buffers...\n");
    
    result = memory_allocate_weights(weights_size, &ctx.weights_buf);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to allocate weights buffer\n");
        goto cleanup;
    }
    
    result = memory_allocate_bias(bias_size, &ctx.bias_buf);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to allocate bias buffer\n");
        goto cleanup;
    }
    
    result = memory_allocate_inference_buffer(&ctx.inference_buf);
    if (result != 0) {
        fprintf(stderr, "ERROR: Failed to allocate inference buffer\n");
        goto cleanup;
    }
    
    // Copy weights and bias to DMA buffers
    // Use chunked copy for large uncached DMA buffers to avoid bus errors
    YOLO2_LOG_INFO("      Copying weights to DMA buffers...\n");
    {
        const size_t chunk_size = 4096;  // Copy 4KB at a time
        size_t offset = 0;
        char *src = (char*)weights_data;
        volatile char *dst = (volatile char*)ctx.weights_buf.ptr;
        
        while (offset < weights_size) {
            size_t copy_len = (weights_size - offset > chunk_size) ? chunk_size : (weights_size - offset);
            for (size_t i = 0; i < copy_len; i++) {
                dst[offset + i] = src[offset + i];
            }
            offset += copy_len;
            
            // Progress indicator for large copy
            if ((offset % (10 * 1024 * 1024)) == 0) {
                YOLO2_LOG_INFO("        %zu MB copied...\n", offset / (1024 * 1024));
            }
        }
        __sync_synchronize();
        YOLO2_LOG_INFO("      Weights copied (%zu bytes)\n", weights_size);
    }
    
    // Copy bias (smaller, can use direct copy)
    {
        volatile char *dst = (volatile char*)ctx.bias_buf.ptr;
        char *src = (char*)bias_data;
        for (size_t i = 0; i < bias_size; i++) {
            dst[i] = src[i];
        }
        __sync_synchronize();
        YOLO2_LOG_INFO("      Bias copied (%zu bytes)\n", bias_size);
    }
    
    // Sync for device
    memory_flush_cache(ctx.weights_buf.ptr, weights_size);
    memory_flush_cache(ctx.bias_buf.ptr, bias_size);
    
    // Free temporary buffers
    free(weights_data);
    weights_data = NULL;
    free(bias_data);
    bias_data = NULL;
    
    YOLO2_LOG_INFO("      DMA buffers allocated OK\n\n");
    
    // Step 6: Parse network configuration
    YOLO2_LOG_INFO("[6/8] Parsing network configuration...\n");
    ctx.net = yolo2_parse_network_cfg(config_path);
    if (!ctx.net) {
        fprintf(stderr, "ERROR: Failed to parse network configuration\n");
        goto cleanup;
    }
    YOLO2_LOG_INFO("\n");
    
    // Step 7: Load input image
    input_image = (float*)malloc(INPUT_ELEMS * sizeof(float));
    if (!input_image) {
        fprintf(stderr, "ERROR: Failed to allocate input image buffer\n");
        goto cleanup;
    }

    // Load labels once (used by all modes).
    if (yolo2_load_labels(labels_path, &labels, &num_labels) < 0) {
        YOLO2_LOG_INFO("WARNING: Failed to load labels from %s\n", labels_path);
        labels = NULL;
        num_labels = 0;
    }

    if (input_mode == INPUT_MODE_IMAGE) {
        YOLO2_LOG_INFO("[7/8] Loading input image...\n");
        result = yolo2_load_image(image_path, input_image);
        if (result != 0) {
            fprintf(stderr, "ERROR: Failed to load image from %s\n", image_path);
            goto cleanup;
        }
        YOLO2_LOG_INFO("\n");
    } else {
        YOLO2_LOG_INFO("[7/8] Initializing streaming input...\n");
        if (save_annotated_dir[0]) {
            if (mkdir_p(save_annotated_dir) != 0) {
                fprintf(stderr, "ERROR: Failed to create output dir: %s\n", save_annotated_dir);
                result = 1;
                goto cleanup;
            }
        }
        if (output_json_path[0]) {
            json_fp = fopen(output_json_path, "w");
            if (!json_fp) {
                fprintf(stderr, "ERROR: Failed to open JSON output %s: %s\n", output_json_path, strerror(errno));
                result = 1;
                goto cleanup;
            }
        }
        YOLO2_LOG_INFO("\n");
    }
    
    // Debug: Test memory access pattern
    if (yolo2_get_verbosity() >= 3) {
        printf("\n[DEBUG] Testing memory write/read...\n");
        // Write test pattern to first few elements of inference buffer
        int16_t *test_buf = (int16_t *)ctx.inference_buf.ptr;
        uint64_t test_phys = ctx.inference_buf.phys_addr;
        
        printf("  Inference buffer: virt=%p, phys=0x%lx\n", 
               (void*)test_buf, (unsigned long)test_phys);
        
        // Write pattern
        for (int i = 0; i < 16; i++) {
            test_buf[i] = 0x1234 + i;
        }
        __sync_synchronize();
        
        // Read back
        printf("  Written: ");
        for (int i = 0; i < 8; i++) {
            printf("0x%04x ", (unsigned)test_buf[i]);
        }
        printf("\n");
        
        // Check if input data was written correctly
        int16_t *in_data = ctx.in_ptr[0];
        printf("  Input buffer ptr: %p (should be ~%p + 1024)\n", 
               (void*)in_data, (void*)test_buf);
    }
    
    // Step 8: Run inference
    YOLO2_LOG_INFO("\n[8/8] Running inference...\n");

    if (input_mode == INPUT_MODE_IMAGE) {
        start_time = get_time_ms();
        result = yolo2_run_inference(&ctx, input_image);
        end_time = get_time_ms();

        if (result != 0) {
            fprintf(stderr, "ERROR: Inference failed\n");
            goto cleanup;
        }

        YOLO2_LOG_INFO("\nInference time: %.2f ms\n", end_time - start_time);

        // Post-processing (unchanged image mode behavior)
        if (ctx.region_output && ctx.region_layer_idx >= 0) {
            layer_t *region_layer = &ctx.net->layers[ctx.region_layer_idx];

            YOLO2_LOG_INFO("\nRunning post-processing...\n");

            // Debug: Check region output values
            if (yolo2_get_verbosity() >= 3) {
                float min_val = ctx.region_output[0], max_val = ctx.region_output[0];
                float sum = 0;
                for (size_t i = 0; i < ctx.region_output_size; i++) {
                    if (ctx.region_output[i] < min_val) min_val = ctx.region_output[i];
                    if (ctx.region_output[i] > max_val) max_val = ctx.region_output[i];
                    sum += ctx.region_output[i];
                }
                printf("  Region output stats: min=%.6f, max=%.6f, mean=%.6f\n",
                       min_val, max_val, sum / ctx.region_output_size);

                // Print first few values for debugging
                printf("  First 10 values: ");
                for (int i = 0; i < 10 && i < (int)ctx.region_output_size; i++) {
                    printf("%.4f ", ctx.region_output[i]);
                }
                printf("\n");
            }

            // Dump region outputs by default for easy CPU vs HW comparison.
            // You can override the output paths via env vars or disable dumps entirely.
            const char *disable_dumps = getenv("YOLO2_NO_DUMP");
            const int do_dump = !(disable_dumps && disable_dumps[0] && disable_dumps[0] != '0');

            // - `YOLO2_DUMP_REGION_RAW`: raw dequantized conv30 output (pre-sigmoid/softmax)
            const char *dump_raw = getenv("YOLO2_DUMP_REGION_RAW");
            const char *dump_raw_path = (dump_raw && dump_raw[0]) ? dump_raw : "yolov2_region_raw_hw.txt";
            if (do_dump) {
                dump_float_array_text(dump_raw_path, ctx.region_output, ctx.region_output_size);
            }

            // Allocate output buffer
            float *region_output_processed = (float*)malloc(ctx.region_output_size * sizeof(float));
            if (!region_output_processed) {
                fprintf(stderr, "ERROR: Failed to allocate processed region output\n");
                goto cleanup;
            }

            // Forward region layer
            result = yolo2_forward_region_layer(region_layer, ctx.region_output, region_output_processed);
            if (result != 0) {
                fprintf(stderr, "ERROR: Forward region layer failed\n");
                free(region_output_processed);
                goto cleanup;
            }

            // - `YOLO2_DUMP_REGION`: region output after sigmoid/softmax (what post-processing consumes)
            const char *dump_processed = getenv("YOLO2_DUMP_REGION");
            const char *dump_processed_path = (dump_processed && dump_processed[0]) ? dump_processed : "yolov2_region_proc_hw.txt";
            if (do_dump) {
                dump_float_array_text(dump_processed_path, region_output_processed, ctx.region_output_size);
            }

            // Get detections
            const int max_dets = 1000;
            yolo2_detection_t *dets = (yolo2_detection_t*)malloc(max_dets * sizeof(yolo2_detection_t));
            if (!dets) {
                fprintf(stderr, "ERROR: Failed to allocate detections array\n");
                free(region_output_processed);
                goto cleanup;
            }

            // Use network input size for detection (letterbox will be corrected)
            int num_dets = yolo2_get_region_detections(region_layer, region_output_processed,
                                                       INPUT_WIDTH, INPUT_HEIGHT,
                                                       INPUT_WIDTH, INPUT_HEIGHT,
                                                       det_thresh, dets, max_dets);

            if (num_dets > 0) {
                // Apply NMS
                yolo2_do_nms_sort(dets, num_dets, region_layer->classes, nms_thresh);

                // Print detections
                yolo2_print_detections(dets, num_dets, det_thresh,
                                       (const char**)labels, num_labels);
            } else {
                printf("\nNo detections found above threshold %.2f\n", det_thresh);
            }

            // Cleanup
            yolo2_free_detections(dets, num_dets);
            free(dets);
            free(region_output_processed);
        } else {
            fprintf(stderr, "WARNING: Region layer output not available for post-processing\n");
        }

        YOLO2_LOG_INFO("\nInference completed successfully!\n");
        result = 0;
    } else {
        // Streaming loop (camera/video), writes headless annotated outputs.
        int stream_ok = 1;
        int mjpeg_started = 0;
        int frame_w = 0;
        int frame_h = 0;
        uint8_t *rgb_frame = NULL;
        float *frame_chw = NULL;

        int frame_idx = 0;
        int infer_idx = 0;

        if (stream_mjpeg_port > 0) {
            if (yolo2_mjpeg_streamer_start(
                    &mjpeg_stream,
                    stream_mjpeg_bind,
                    stream_mjpeg_port,
                    stream_mjpeg_fps,
                    stream_mjpeg_quality) != 0) {
                fprintf(stderr, "ERROR: Failed to start MJPEG streamer on %s:%d\n", stream_mjpeg_bind, stream_mjpeg_port);
                result = 1;
                goto cleanup;
            }
            mjpeg_started = 1;
        }

        if (input_mode == INPUT_MODE_CAMERA) {
            yolo2_v4l2_camera_t cam;
            if (yolo2_v4l2_open(&cam, camera_device, cam_width, cam_height, cam_fps, cam_format) != 0) {
                result = 1;
                goto cleanup;
            }
            if (yolo2_v4l2_start(&cam) != 0) {
                yolo2_v4l2_close(&cam);
                result = 1;
                goto cleanup;
            }

            frame_w = cam.width;
            frame_h = cam.height;
            const size_t rgb_size = (size_t)frame_w * (size_t)frame_h * 3u;
            rgb_frame = (uint8_t *)malloc(rgb_size);
            frame_chw = (float *)malloc((size_t)frame_w * (size_t)frame_h * 3u * sizeof(float));
            if (!rgb_frame || !frame_chw) {
                fprintf(stderr, "ERROR: Failed to allocate frame buffers\n");
                free(rgb_frame);
                free(frame_chw);
                yolo2_v4l2_stop(&cam);
                yolo2_v4l2_close(&cam);
                goto cleanup;
            }

            const int max_dets = 1000;
            yolo2_detection_t *dets = (yolo2_detection_t*)malloc((size_t)max_dets * sizeof(yolo2_detection_t));
            float *region_output_processed = NULL;
            size_t region_processed_cap = 0;
            if (!dets) {
                fprintf(stderr, "ERROR: Failed to allocate detections array\n");
                free(rgb_frame);
                free(frame_chw);
                yolo2_v4l2_stop(&cam);
                yolo2_v4l2_close(&cam);
                goto cleanup;
            }

            while (max_frames == 0 || infer_idx < max_frames) {

                yolo2_v4l2_frame_t frame;
                const int dq = yolo2_v4l2_dequeue(&cam, &frame);
                if (dq == 0) {
                    continue;
                }
                if (dq < 0) {
                    stream_ok = 0;
                    break;
                }

                const int do_infer = (infer_every <= 1) || ((frame_idx % infer_every) == 0);
                int decode_rc = 0;
                if (do_infer) {
                    if (cam.pixfmt == V4L2_PIX_FMT_MJPEG) {
                        decode_rc = yolo2_decode_mjpeg_to_rgb24(frame.data, frame.size, rgb_frame, frame_w, frame_h);
                    } else if (cam.pixfmt == V4L2_PIX_FMT_YUYV) {
                        yolo2_yuyv_to_rgb24(frame.data, rgb_frame, frame_w, frame_h);
                        decode_rc = 0;
                    } else {
                        fprintf(stderr, "ERROR: Unsupported camera pixfmt 0x%08x\n", cam.pixfmt);
                        decode_rc = -1;
                    }
                }

                // Always re-queue ASAP.
                (void)yolo2_v4l2_enqueue(&cam, &frame);

                frame_idx++;
                if (!do_infer || decode_rc != 0) {
                    continue;
                }

                infer_idx++;

                // Preprocess: RGB24 -> float CHW -> letterbox 416x416.
                rgb24_to_chw_float(rgb_frame, frame_chw, frame_w, frame_h);
                if (yolo2_letterbox_image(frame_chw, frame_w, frame_h, 3, input_image, INPUT_WIDTH, INPUT_HEIGHT) != 0) {
                    fprintf(stderr, "ERROR: Letterbox preprocess failed\n");
                    continue;
                }

                start_time = get_time_ms();
                result = yolo2_run_inference(&ctx, input_image);
                end_time = get_time_ms();
                if (result != 0) {
                    fprintf(stderr, "ERROR: Inference failed\n");
                    stream_ok = 0;
                    break;
                }

                YOLO2_LOG_INFO("Frame %d (infer %d) inference time: %.2f ms\n", frame_idx, infer_idx, end_time - start_time);

                if (!ctx.region_output || ctx.region_layer_idx < 0) {
                    fprintf(stderr, "WARNING: Region layer output not available\n");
                    continue;
                }

                layer_t *region_layer = &ctx.net->layers[ctx.region_layer_idx];
                if (!region_output_processed || region_processed_cap < ctx.region_output_size) {
                    float *new_buf = (float*)realloc(region_output_processed, ctx.region_output_size * sizeof(float));
                    if (!new_buf) {
                        fprintf(stderr, "ERROR: Failed to allocate processed region output\n");
                        stream_ok = 0;
                        break;
                    }
                    region_output_processed = new_buf;
                    region_processed_cap = ctx.region_output_size;
                }

                if (yolo2_forward_region_layer(region_layer, ctx.region_output, region_output_processed) != 0) {
                    fprintf(stderr, "ERROR: Forward region layer failed\n");
                    stream_ok = 0;
                    break;
                }

                // Map detections to the original frame size.
                int num_dets = yolo2_get_region_detections(region_layer, region_output_processed,
                                                           frame_w, frame_h,
                                                           INPUT_WIDTH, INPUT_HEIGHT,
                                                           det_thresh, dets, max_dets);
                if (num_dets > 0) {
                    yolo2_do_nms_sort(dets, num_dets, region_layer->classes, nms_thresh);
                }

                if (json_fp) {
                    fprintf(json_fp, "{");
                    fprintf(json_fp, "\"mode\":\"camera\",");
                    fprintf(json_fp, "\"source\":");
                    json_write_escaped(json_fp, camera_device);
                    fprintf(json_fp, ",\"frame_index\":%d,\"inference_index\":%d,", frame_idx, infer_idx);
                    fprintf(json_fp, "\"width\":%d,\"height\":%d,", frame_w, frame_h);
                    fprintf(json_fp, "\"detections\":[");

                    int first = 1;
                    for (int i = 0; i < num_dets; ++i) {
                        int best_class = -1;
                        float best_prob = 0.0f;
                        for (int cls = 0; cls < dets[i].classes; ++cls) {
                            if (dets[i].prob && dets[i].prob[cls] > best_prob) {
                                best_prob = dets[i].prob[cls];
                                best_class = cls;
                            }
                        }

                        if (best_prob <= det_thresh || best_class < 0) {
                            continue;
                        }

                        const char *label = (labels && best_class < num_labels) ? labels[best_class] : "unknown";
                        const yolo2_box_t b = dets[i].bbox;

                        const int x0 = (int)((b.x - b.w * 0.5f) * (float)frame_w);
                        const int y0 = (int)((b.y - b.h * 0.5f) * (float)frame_h);
                        const int x1 = (int)((b.x + b.w * 0.5f) * (float)frame_w);
                        const int y1 = (int)((b.y + b.h * 0.5f) * (float)frame_h);

                        if (!first) fprintf(json_fp, ",");
                        first = 0;

                        fprintf(json_fp, "{");
                        fprintf(json_fp, "\"class_id\":%d,", best_class);
                        fprintf(json_fp, "\"label\":");
                        json_write_escaped(json_fp, label);
                        fprintf(json_fp, ",\"prob\":%.6f,", best_prob);
                        fprintf(json_fp, "\"bbox_norm\":{\"x\":%.6f,\"y\":%.6f,\"w\":%.6f,\"h\":%.6f},",
                                b.x, b.y, b.w, b.h);
                        fprintf(json_fp, "\"bbox_px\":{\"x0\":%d,\"y0\":%d,\"x1\":%d,\"y1\":%d}",
                                x0, y0, x1, y1);
                        fprintf(json_fp, "}");
                    }

                    fprintf(json_fp, "]}\n");
                    fflush(json_fp);
                }

                const int want_annotated = (save_annotated_dir[0] != '\0') || mjpeg_started;
                if (want_annotated) {
                    yolo2_draw_detections_rgb24(rgb_frame, frame_w, frame_h, dets, num_dets, det_thresh, (const char **)labels, num_labels);
                }

                if (save_annotated_dir[0]) {
                    char out_path[PATH_MAX];
                    snprintf(out_path, sizeof(out_path), "%s/frame_%06d.png", save_annotated_dir, infer_idx);
                    (void)yolo2_write_png_rgb24(out_path, rgb_frame, frame_w, frame_h);
                }
                if (mjpeg_started) {
                    (void)yolo2_mjpeg_streamer_update_rgb24(mjpeg_stream, rgb_frame, frame_w, frame_h);
                }

                yolo2_free_detections(dets, num_dets);
            }

            free(region_output_processed);
            free(dets);

            yolo2_v4l2_stop(&cam);
            yolo2_v4l2_close(&cam);
        } else {
            yolo2_ffmpeg_video_t vid;
            if (yolo2_ffmpeg_video_open(&vid, video_path, video_width, video_height, video_fps) != 0) {
                result = 1;
                goto cleanup;
            }

            frame_w = vid.width;
            frame_h = vid.height;
            const size_t rgb_size = (size_t)frame_w * (size_t)frame_h * 3u;
            rgb_frame = (uint8_t *)malloc(rgb_size);
            frame_chw = (float *)malloc((size_t)frame_w * (size_t)frame_h * 3u * sizeof(float));
            if (!rgb_frame || !frame_chw) {
                fprintf(stderr, "ERROR: Failed to allocate frame buffers\n");
                (void)yolo2_ffmpeg_video_close(&vid);
                free(rgb_frame);
                free(frame_chw);
                goto cleanup;
            }

            const int max_dets = 1000;
            yolo2_detection_t *dets = (yolo2_detection_t*)malloc((size_t)max_dets * sizeof(yolo2_detection_t));
            float *region_output_processed = NULL;
            size_t region_processed_cap = 0;
            if (!dets) {
                fprintf(stderr, "ERROR: Failed to allocate detections array\n");
                (void)yolo2_ffmpeg_video_close(&vid);
                free(rgb_frame);
                free(frame_chw);
                goto cleanup;
            }

            while (max_frames == 0 || infer_idx < max_frames) {
                const int r = yolo2_ffmpeg_video_read_frame(&vid, rgb_frame, rgb_size);
                if (r == 0) {
                    break; // EOF
                }
                if (r < 0) {
                    stream_ok = 0;
                    break;
                }

                const int do_infer = (infer_every <= 1) || ((frame_idx % infer_every) == 0);
                frame_idx++;
                if (!do_infer) {
                    continue;
                }

                infer_idx++;

                rgb24_to_chw_float(rgb_frame, frame_chw, frame_w, frame_h);
                if (yolo2_letterbox_image(frame_chw, frame_w, frame_h, 3, input_image, INPUT_WIDTH, INPUT_HEIGHT) != 0) {
                    fprintf(stderr, "ERROR: Letterbox preprocess failed\n");
                    continue;
                }

                start_time = get_time_ms();
                result = yolo2_run_inference(&ctx, input_image);
                end_time = get_time_ms();
                if (result != 0) {
                    fprintf(stderr, "ERROR: Inference failed\n");
                    stream_ok = 0;
                    break;
                }

                YOLO2_LOG_INFO("Frame %d (infer %d) inference time: %.2f ms\n", frame_idx, infer_idx, end_time - start_time);

                if (!ctx.region_output || ctx.region_layer_idx < 0) {
                    fprintf(stderr, "WARNING: Region layer output not available\n");
                    continue;
                }

                layer_t *region_layer = &ctx.net->layers[ctx.region_layer_idx];
                if (!region_output_processed || region_processed_cap < ctx.region_output_size) {
                    float *new_buf = (float*)realloc(region_output_processed, ctx.region_output_size * sizeof(float));
                    if (!new_buf) {
                        fprintf(stderr, "ERROR: Failed to allocate processed region output\n");
                        stream_ok = 0;
                        break;
                    }
                    region_output_processed = new_buf;
                    region_processed_cap = ctx.region_output_size;
                }

                if (yolo2_forward_region_layer(region_layer, ctx.region_output, region_output_processed) != 0) {
                    fprintf(stderr, "ERROR: Forward region layer failed\n");
                    stream_ok = 0;
                    break;
                }

                int num_dets = yolo2_get_region_detections(region_layer, region_output_processed,
                                                           frame_w, frame_h,
                                                           INPUT_WIDTH, INPUT_HEIGHT,
                                                           det_thresh, dets, max_dets);
                if (num_dets > 0) {
                    yolo2_do_nms_sort(dets, num_dets, region_layer->classes, nms_thresh);
                }

                if (json_fp) {
                    fprintf(json_fp, "{");
                    fprintf(json_fp, "\"mode\":\"video\",");
                    fprintf(json_fp, "\"source\":");
                    json_write_escaped(json_fp, video_path);
                    fprintf(json_fp, ",\"frame_index\":%d,\"inference_index\":%d,", frame_idx, infer_idx);
                    fprintf(json_fp, "\"width\":%d,\"height\":%d,", frame_w, frame_h);
                    fprintf(json_fp, "\"detections\":[");

                    int first = 1;
                    for (int i = 0; i < num_dets; ++i) {
                        int best_class = -1;
                        float best_prob = 0.0f;
                        for (int cls = 0; cls < dets[i].classes; ++cls) {
                            if (dets[i].prob && dets[i].prob[cls] > best_prob) {
                                best_prob = dets[i].prob[cls];
                                best_class = cls;
                            }
                        }

                        if (best_prob <= det_thresh || best_class < 0) {
                            continue;
                        }

                        const char *label = (labels && best_class < num_labels) ? labels[best_class] : "unknown";
                        const yolo2_box_t b = dets[i].bbox;

                        const int x0 = (int)((b.x - b.w * 0.5f) * (float)frame_w);
                        const int y0 = (int)((b.y - b.h * 0.5f) * (float)frame_h);
                        const int x1 = (int)((b.x + b.w * 0.5f) * (float)frame_w);
                        const int y1 = (int)((b.y + b.h * 0.5f) * (float)frame_h);

                        if (!first) fprintf(json_fp, ",");
                        first = 0;

                        fprintf(json_fp, "{");
                        fprintf(json_fp, "\"class_id\":%d,", best_class);
                        fprintf(json_fp, "\"label\":");
                        json_write_escaped(json_fp, label);
                        fprintf(json_fp, ",\"prob\":%.6f,", best_prob);
                        fprintf(json_fp, "\"bbox_norm\":{\"x\":%.6f,\"y\":%.6f,\"w\":%.6f,\"h\":%.6f},",
                                b.x, b.y, b.w, b.h);
                        fprintf(json_fp, "\"bbox_px\":{\"x0\":%d,\"y0\":%d,\"x1\":%d,\"y1\":%d}",
                                x0, y0, x1, y1);
                        fprintf(json_fp, "}");
                    }

                    fprintf(json_fp, "]}\n");
                    fflush(json_fp);
                }

                const int want_annotated = (save_annotated_dir[0] != '\0') || mjpeg_started;
                if (want_annotated) {
                    yolo2_draw_detections_rgb24(rgb_frame, frame_w, frame_h, dets, num_dets, det_thresh, (const char **)labels, num_labels);
                }

                if (save_annotated_dir[0]) {
                    char out_path[PATH_MAX];
                    snprintf(out_path, sizeof(out_path), "%s/frame_%06d.png", save_annotated_dir, infer_idx);
                    (void)yolo2_write_png_rgb24(out_path, rgb_frame, frame_w, frame_h);
                }
                if (mjpeg_started) {
                    (void)yolo2_mjpeg_streamer_update_rgb24(mjpeg_stream, rgb_frame, frame_w, frame_h);
                }

                yolo2_free_detections(dets, num_dets);
            }

            free(region_output_processed);
            free(dets);

            (void)yolo2_ffmpeg_video_close(&vid);
        }

        free(rgb_frame);
        free(frame_chw);

        if (!stream_ok) {
            result = 1;
            goto cleanup;
        }
        if (infer_idx == 0) {
            fprintf(stderr, "ERROR: No inference frames processed\n");
            result = 1;
            goto cleanup;
        }

        YOLO2_LOG_INFO("\nStreaming inference completed successfully (%d inference frames)\n", infer_idx);
        result = 0;
    }
    
cleanup:
    // Cleanup
    if (input_image) free(input_image);
    if (weights_data) free(weights_data);
    if (bias_data) free(bias_data);
    if (labels) yolo2_free_labels(labels, num_labels);
    if (json_fp) fclose(json_fp);
    if (mjpeg_stream) yolo2_mjpeg_streamer_stop(mjpeg_stream);
    if (ctx.net) yolo2_free_network(ctx.net);
    
    yolo2_inference_cleanup(&ctx);
    dma_buffer_cleanup();
    yolo2_accel_cleanup();
    
    YOLO2_LOG_INFO("\n========================================\n");
    YOLO2_LOG_INFO("Application finished\n");
    YOLO2_LOG_INFO("========================================\n\n");
    
    return result;
}
