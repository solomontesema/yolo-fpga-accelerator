/**
 * YOLOv2 Linux App - Headless drawing + PNG output
 */

#include "yolo2_draw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stb_image_write.h"

static int clamp_int(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void set_pixel_rgb24(uint8_t *rgb, int width, int height, int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (!rgb) return;
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    uint8_t *p = rgb + ((size_t)y * (size_t)width + (size_t)x) * 3u;
    p[0] = r;
    p[1] = g;
    p[2] = b;
}

static void fill_rect_rgb24(uint8_t *rgb, int width, int height, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b)
{
    if (!rgb || width <= 0 || height <= 0) return;

    if (x0 > x1) {
        int t = x0;
        x0 = x1;
        x1 = t;
    }
    if (y0 > y1) {
        int t = y0;
        y0 = y1;
        y1 = t;
    }

    x0 = clamp_int(x0, 0, width - 1);
    x1 = clamp_int(x1, 0, width - 1);
    y0 = clamp_int(y0, 0, height - 1);
    y1 = clamp_int(y1, 0, height - 1);

    for (int y = y0; y <= y1; ++y) {
        uint8_t *row = rgb + ((size_t)y * (size_t)width + (size_t)x0) * 3u;
        for (int x = x0; x <= x1; ++x) {
            row[0] = r;
            row[1] = g;
            row[2] = b;
            row += 3;
        }
    }
}

void yolo2_draw_rect_rgb24(
    uint8_t *rgb,
    int width,
    int height,
    int x0,
    int y0,
    int x1,
    int y1,
    int thickness,
    uint8_t r,
    uint8_t g,
    uint8_t b)
{
    if (!rgb || width <= 0 || height <= 0) return;

    if (thickness <= 0) thickness = 1;

    if (x0 > x1) {
        int t = x0;
        x0 = x1;
        x1 = t;
    }
    if (y0 > y1) {
        int t = y0;
        y0 = y1;
        y1 = t;
    }

    x0 = clamp_int(x0, 0, width - 1);
    x1 = clamp_int(x1, 0, width - 1);
    y0 = clamp_int(y0, 0, height - 1);
    y1 = clamp_int(y1, 0, height - 1);

    for (int t = 0; t < thickness; ++t) {
        int xx0 = clamp_int(x0 + t, 0, width - 1);
        int xx1 = clamp_int(x1 - t, 0, width - 1);
        int yy0 = clamp_int(y0 + t, 0, height - 1);
        int yy1 = clamp_int(y1 - t, 0, height - 1);

        // Horizontal edges
        for (int x = xx0; x <= xx1; ++x) {
            set_pixel_rgb24(rgb, width, height, x, yy0, r, g, b);
            set_pixel_rgb24(rgb, width, height, x, yy1, r, g, b);
        }

        // Vertical edges
        for (int y = yy0; y <= yy1; ++y) {
            set_pixel_rgb24(rgb, width, height, xx0, y, r, g, b);
            set_pixel_rgb24(rgb, width, height, xx1, y, r, g, b);
        }
    }
}

// Minimal 5x7 font (uppercase-style) for: space, '.', digits, a-z.
// Each row is 5 bits, MSB on the left.
static const uint8_t *glyph5x7(char c)
{
    static const uint8_t space[7] = {0, 0, 0, 0, 0, 0, 0};
    static const uint8_t dot[7] = {0, 0, 0, 0, 0, 0x04, 0x04};

    // digits
    static const uint8_t g0[7] = {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    static const uint8_t g1[7] = {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
    static const uint8_t g2[7] = {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
    static const uint8_t g3[7] = {0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E};
    static const uint8_t g4[7] = {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
    static const uint8_t g5[7] = {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E};
    static const uint8_t g6[7] = {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E};
    static const uint8_t g7[7] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    static const uint8_t g8[7] = {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
    static const uint8_t g9[7] = {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C};

    // letters (A-Z patterns, mapped from a-z)
    static const uint8_t ga[7] = {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    static const uint8_t gb[7] = {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    static const uint8_t gc[7] = {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
    static const uint8_t gd[7] = {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E};
    static const uint8_t ge[7] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    static const uint8_t gf[7] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    static const uint8_t gg[7] = {0x0E, 0x11, 0x10, 0x10, 0x13, 0x11, 0x0F};
    static const uint8_t gh[7] = {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    static const uint8_t gi[7] = {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E};
    static const uint8_t gj[7] = {0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E};
    static const uint8_t gk[7] = {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    static const uint8_t gl[7] = {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
    static const uint8_t gm[7] = {0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11};
    static const uint8_t gn[7] = {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    static const uint8_t go[7] = {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    static const uint8_t gp[7] = {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
    static const uint8_t gq[7] = {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
    static const uint8_t gr[7] = {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    static const uint8_t gs[7] = {0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E};
    static const uint8_t gt[7] = {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    static const uint8_t gu[7] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    static const uint8_t gv[7] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04};
    static const uint8_t gw[7] = {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A};
    static const uint8_t gx[7] = {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11};
    static const uint8_t gy[7] = {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04};
    static const uint8_t gz[7] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};

    if (c == ' ') return space;
    if (c == '.') return dot;

    if (c >= 'A' && c <= 'Z') {
        c = (char)(c - 'A' + 'a');
    }

    switch (c) {
        case '0': return g0;
        case '1': return g1;
        case '2': return g2;
        case '3': return g3;
        case '4': return g4;
        case '5': return g5;
        case '6': return g6;
        case '7': return g7;
        case '8': return g8;
        case '9': return g9;
        case 'a': return ga;
        case 'b': return gb;
        case 'c': return gc;
        case 'd': return gd;
        case 'e': return ge;
        case 'f': return gf;
        case 'g': return gg;
        case 'h': return gh;
        case 'i': return gi;
        case 'j': return gj;
        case 'k': return gk;
        case 'l': return gl;
        case 'm': return gm;
        case 'n': return gn;
        case 'o': return go;
        case 'p': return gp;
        case 'q': return gq;
        case 'r': return gr;
        case 's': return gs;
        case 't': return gt;
        case 'u': return gu;
        case 'v': return gv;
        case 'w': return gw;
        case 'x': return gx;
        case 'y': return gy;
        case 'z': return gz;
        default:
            return space;
    }
}

static void draw_char5x7_rgb24(uint8_t *rgb, int width, int height, int x, int y, char c, int scale, uint8_t r, uint8_t g, uint8_t b)
{
    const uint8_t *rows = glyph5x7(c);
    const int glyph_w = 5;
    const int glyph_h = 7;

    if (scale <= 0) scale = 1;

    for (int row = 0; row < glyph_h; ++row) {
        const uint8_t bits = rows[row];
        for (int col = 0; col < glyph_w; ++col) {
            const int on = (bits >> (glyph_w - 1 - col)) & 1;
            if (!on) continue;
            const int px0 = x + col * scale;
            const int py0 = y + row * scale;
            for (int yy = 0; yy < scale; ++yy) {
                for (int xx = 0; xx < scale; ++xx) {
                    set_pixel_rgb24(rgb, width, height, px0 + xx, py0 + yy, r, g, b);
                }
            }
        }
    }
}

static int text_width5x7(const char *text)
{
    if (!text) return 0;
    const int glyph_w = 5;
    const int spacing = 1;
    const int len = (int)strlen(text);
    if (len <= 0) return 0;
    return len * (glyph_w + spacing) - spacing;
}

static void draw_text5x7_rgb24(uint8_t *rgb, int width, int height, int x, int y, const char *text, int scale, uint8_t r, uint8_t g, uint8_t b)
{
    if (!text || !text[0]) return;
    const int glyph_w = 5;
    const int spacing = 1;
    int cx = x;
    for (const char *p = text; *p; ++p) {
        draw_char5x7_rgb24(rgb, width, height, cx, y, *p, scale, r, g, b);
        cx += (glyph_w + spacing) * scale;
    }
}

static void pick_color(int class_id, uint8_t *r, uint8_t *g, uint8_t *b)
{
    static const uint8_t palette[][3] = {
        {255,  30,  30},  // red
        { 30, 255,  30},  // green
        { 30,  30, 255},  // blue
        {255, 255,  30},  // yellow
        {255,  30, 255},  // magenta
        { 30, 255, 255},  // cyan
        {255, 128,  30},  // orange
        {128,  30, 255},  // purple
    };

    const int n = (int)(sizeof(palette) / sizeof(palette[0]));
    const int idx = (class_id >= 0) ? (class_id % n) : 0;
    *r = palette[idx][0];
    *g = palette[idx][1];
    *b = palette[idx][2];
}

int yolo2_draw_detections_rgb24(
    uint8_t *rgb,
    int width,
    int height,
    const yolo2_detection_t *dets,
    int num_dets,
    float thresh,
    const char **labels,
    int num_labels)
{
    if (!rgb || width <= 0 || height <= 0 || !dets || num_dets <= 0) {
        return 0;
    }

    int drawn = 0;
    for (int i = 0; i < num_dets; ++i) {
        int best_class = -1;
        float best_prob = 0.0f;

        for (int cls = 0; cls < dets[i].classes; ++cls) {
            const float p = dets[i].prob ? dets[i].prob[cls] : 0.0f;
            if (p > best_prob) {
                best_prob = p;
                best_class = cls;
            }
        }

        if (best_prob <= thresh || best_class < 0) {
            continue;
        }

        const char *label = NULL;
        char label_fallback[32];
        if (labels && best_class >= 0 && best_class < num_labels) {
            label = labels[best_class];
        } else {
            snprintf(label_fallback, sizeof(label_fallback), "class%d", best_class);
            label = label_fallback;
        }

        const yolo2_box_t b = dets[i].bbox;
        const float cx = b.x;
        const float cy = b.y;
        const float bw = b.w;
        const float bh = b.h;

        int x0 = (int)((cx - bw * 0.5f) * (float)width);
        int y0 = (int)((cy - bh * 0.5f) * (float)height);
        int x1 = (int)((cx + bw * 0.5f) * (float)width);
        int y1 = (int)((cy + bh * 0.5f) * (float)height);

        x0 = clamp_int(x0, 0, width - 1);
        x1 = clamp_int(x1, 0, width - 1);
        y0 = clamp_int(y0, 0, height - 1);
        y1 = clamp_int(y1, 0, height - 1);

        uint8_t r, g, bb;
        pick_color(best_class, &r, &g, &bb);
        yolo2_draw_rect_rgb24(rgb, width, height, x0, y0, x1, y1, 2, r, g, bb);

        // Label: "<name> <prob>"
        char text[128];
        snprintf(text, sizeof(text), "%s %.2f", label, best_prob);

        const int scale = 2;
        const int glyph_h = 7;
        const int pad = 2;
        const int tw = text_width5x7(text) * scale;
        const int th = glyph_h * scale;

        int tx = x0;
        int ty = y0 - th - pad * 2;
        if (ty < 0) {
            ty = y0 + 1;
        }

        // Background box
        const int bg_x0 = tx;
        const int bg_y0 = ty;
        const int bg_x1 = tx + tw + pad * 2;
        const int bg_y1 = ty + th + pad * 2;
        fill_rect_rgb24(rgb, width, height, bg_x0, bg_y0, bg_x1, bg_y1, r, g, bb);

        // Text color: choose black/white based on background brightness.
        const int brightness = (int)r + (int)g + (int)bb;
        const uint8_t tr = (brightness > (int)(255 * 3 / 2)) ? 0 : 255;
        const uint8_t tg = tr;
        const uint8_t tb = tr;
        draw_text5x7_rgb24(rgb, width, height, tx + pad, ty + pad, text, scale, tr, tg, tb);
        drawn++;
    }

    return drawn;
}

int yolo2_write_png_rgb24(const char *path, const uint8_t *rgb, int width, int height)
{
    if (!path || !path[0] || !rgb || width <= 0 || height <= 0) {
        return -1;
    }

    const int stride = width * 3;
    const int ok = stbi_write_png(path, width, height, 3, rgb, stride);
    if (!ok) {
        fprintf(stderr, "ERROR: Failed to write PNG: %s\n", path);
        return -1;
    }
    return 0;
}
