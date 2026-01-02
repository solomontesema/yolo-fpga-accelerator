/**
 * YOLOv2 Linux App - Lightweight logging helpers
 *
 * Runtime control via env var:
 *   YOLO2_VERBOSE=0..3
 *     0: errors only
 *     1: high-level info (default)
 *     2: per-layer info
 *     3: debug (addresses/status polling)
 */

#ifndef YOLO2_LOG_H
#define YOLO2_LOG_H

#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

int yolo2_get_verbosity(void);
void yolo2_set_verbosity(int level);

#ifdef __cplusplus
}
#endif

#define YOLO2_LOG(level, ...) \
    do { \
        if (yolo2_get_verbosity() >= (level)) { \
            fprintf(stdout, __VA_ARGS__); \
        } \
    } while (0)

#define YOLO2_LOG_INFO(...)  YOLO2_LOG(1, __VA_ARGS__)
#define YOLO2_LOG_LAYER(...) YOLO2_LOG(2, __VA_ARGS__)
#define YOLO2_LOG_DEBUG(...) YOLO2_LOG(3, __VA_ARGS__)

#endif /* YOLO2_LOG_H */
