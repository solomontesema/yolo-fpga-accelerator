/**
 * YOLOv2 Linux App - Logging helpers
 *
 * `YOLO2_VERBOSE` controls runtime verbosity:
 *   0: quiet (errors + detections only)
 *   1: high-level info (default)
 *   2: per-layer info
 *   3: debug (addresses/status polling)
 *
 * CLI can override via `yolo2_set_verbosity()` (used by `main.c` option `-v`).
 */

#include "yolo2_log.h"

#include <stdlib.h>

static int g_yolo2_verbosity = -1;     // -1 => not set by CLI
static int g_cached_env = -2;          // -2 => not loaded, 0..3 => cached value

static int clamp_level(int level)
{
    if (level < 0) return 0;
    if (level > 3) return 3;
    return level;
}

void yolo2_set_verbosity(int level)
{
    g_yolo2_verbosity = clamp_level(level);
}

int yolo2_get_verbosity(void)
{
    if (g_yolo2_verbosity >= 0) {
        return g_yolo2_verbosity;
    }

    if (g_cached_env != -2) {
        return g_cached_env;
    }

    const char *env = getenv("YOLO2_VERBOSE");
    if (!env || env[0] == '\0') {
        g_cached_env = 1;
        return g_cached_env;
    }

    char *end = NULL;
    long value = strtol(env, &end, 10);
    if (end == env || *end != '\0') {
        g_cached_env = 1;
        return g_cached_env;
    }

    g_cached_env = clamp_level((int)value);
    return g_cached_env;
}

