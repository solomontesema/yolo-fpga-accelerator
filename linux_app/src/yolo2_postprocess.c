/**
 * YOLOv2 Post-Processing - Linux Implementation
 * 
 * NMS and detection extraction from region layer output.
 */

#include "yolo2_postprocess.h"
#include "yolo2_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Sigmoid activation
static inline float sigmoid(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

// Apply sigmoid to array
static void activate_array_sigmoid(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = sigmoid(x[i]);
    }
}

static void softmax_stride(const float *input, int n, int stride, float *output)
{
    float largest = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        float v = input[i * stride];
        if (v > largest) largest = v;
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = expf(input[i * stride] - largest);
        sum += e;
        output[i * stride] = e;
    }

    if (sum <= 0.0f) {
        const float inv = 1.0f / (float)n;
        for (int i = 0; i < n; ++i) {
            output[i * stride] = inv;
        }
        return;
    }

    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        output[i * stride] *= inv_sum;
    }
}

// Entry index calculation
static int entry_index(layer_t *l, int batch, int location, int entry) {
    int n = location / (l->w * l->h);
    int loc = location % (l->w * l->h);
    return batch * l->outputs + n * l->w * l->h * (4 + l->classes + 1) + entry * l->w * l->h + loc;
}

// Get region box from output
static yolo2_box_t get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride) {
    yolo2_box_t b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = expf(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = expf(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

// Correct boxes for letterboxing
static void correct_region_boxes(yolo2_detection_t *dets, int n, int w, int h, int netw, int neth, int relative) {
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    
    for (int i = 0; i < n; ++i) {
        yolo2_box_t *b = &dets[i].bbox;
        b->x = (b->x - (netw - new_w) / 2.0f / netw) / ((float)new_w / netw);
        b->y = (b->y - (neth - new_h) / 2.0f / neth) / ((float)new_h / neth);
        b->w *= (float)netw / new_w;
        b->h *= (float)neth / new_h;
        if (!relative) {
            b->x *= w;
            b->w *= w;
            b->y *= h;
            b->h *= h;
        }
    }
}

/**
 * Forward region layer
 */
int yolo2_forward_region_layer(layer_t *l, float *input, float *output) {
    if (!l || !input || !output) {
        return -1;
    }
    
    // Copy input to output
    memcpy(output, input, l->outputs * sizeof(float));
    
    // Apply sigmoid to coordinates and objectness
    for (int n = 0; n < l->num; ++n) {
        int index = entry_index(l, 0, n * l->w * l->h, 0);
        activate_array_sigmoid(output + index, 2 * l->w * l->h);
        
        index = entry_index(l, 0, n * l->w * l->h, l->coords);
        activate_array_sigmoid(output + index, l->w * l->h);
    }

    // Apply softmax to class predictions if requested (YOLOv2 region layer uses this).
    if (l->softmax) {
        const int spatial = l->w * l->h;
        for (int n = 0; n < l->num; ++n) {
            for (int loc = 0; loc < spatial; ++loc) {
                int index = entry_index(l, 0, n * spatial + loc, l->coords + 1);
                softmax_stride(output + index, l->classes, spatial, output + index);
            }
        }
    }
    
    return 0;
}

/**
 * Get detections from region layer output
 */
int yolo2_get_region_detections(layer_t *l, float *output,
                                int img_w, int img_h, int net_w, int net_h,
                                float thresh, yolo2_detection_t *dets, int max_dets) {
    if (!l || !output || !dets) {
        return -1;
    }
    
    // Default biases for YOLOv2 (5 anchors)
    static const float biases[10] = {
        0.57273f, 0.677385f, 1.87446f, 2.06253f, 3.33843f,
        5.47434f, 7.88282f, 3.52778f, 9.77052f, 9.16828f
    };
    
    int count = 0;
    for (int i = 0; i < l->w * l->h; ++i) {
        int row = i / l->w;
        int col = i % l->w;
        
        for (int n = 0; n < l->num; ++n) {
            if (count >= max_dets) {
                fprintf(stderr, "WARNING: Maximum detections reached\n");
                return count;
            }
            
            int obj_index = entry_index(l, 0, n * l->w * l->h + i, l->coords);
            float objectness = output[obj_index];
            
            if (objectness <= thresh) {
                continue;
            }
            
            int box_index = entry_index(l, 0, n * l->w * l->h + i, 0);
            dets[count].bbox = get_region_box(output, (float*)biases, n, box_index, col, row, l->w, l->h, l->w * l->h);
            dets[count].objectness = objectness;
            dets[count].classes = l->classes;
            dets[count].sort_class = -1;
            
            dets[count].prob = (float*)malloc(l->classes * sizeof(float));
            if (!dets[count].prob) {
                fprintf(stderr, "ERROR: Failed to allocate probability array\n");
                return count;
            }
            
            for (int j = 0; j < l->classes; ++j) {
                int class_index = entry_index(l, 0, n * l->w * l->h + i, l->coords + 1 + j);
                float prob = objectness * output[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0.0f;
            }
            
            ++count;
        }
    }
    
    correct_region_boxes(dets, count, img_w, img_h, net_w, net_h, 1);
    
    return count;
}

// Box IoU
static float box_iou(yolo2_box_t a, yolo2_box_t b) {
    float overlap_x1 = (a.x > b.x) ? a.x : b.x;
    float overlap_y1 = (a.y > b.y) ? a.y : b.y;
    float overlap_x2 = (a.x + a.w < b.x + b.w) ? (a.x + a.w) : (b.x + b.w);
    float overlap_y2 = (a.y + a.h < b.y + b.h) ? (a.y + a.h) : (b.y + b.h);
    
    if (overlap_x2 < overlap_x1 || overlap_y2 < overlap_y1) {
        return 0.0f;
    }
    
    float intersection = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1);
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - intersection;
    
    if (union_area <= 0.0f) return 0.0f;
    return intersection / union_area;
}

// NMS comparator
static int nms_comparator(const void *pa, const void *pb) {
    yolo2_detection_t a = *(yolo2_detection_t*)pa;
    yolo2_detection_t b = *(yolo2_detection_t*)pb;
    float diff = 0;
    
    if (b.sort_class >= 0 && b.sort_class < a.classes) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    
    if (diff < 0) return 1;
    if (diff > 0) return -1;
    return 0;
}

/**
 * Non-maximum suppression
 */
void yolo2_do_nms_sort(yolo2_detection_t *dets, int total, int classes, float nms_thresh) {
    if (!dets || total <= 0) return;
    
    // Remove zero objectness
    int k = total - 1;
    for (int i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0.0f) {
            yolo2_detection_t swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;
    
    // Apply NMS for each class
    for (int cls = 0; cls < classes; ++cls) {
        for (int i = 0; i < total; ++i) {
            dets[i].sort_class = cls;
        }
        
        qsort(dets, total, sizeof(yolo2_detection_t), nms_comparator);
        
        for (int i = 0; i < total; ++i) {
            if (dets[i].prob[cls] == 0.0f) continue;
            
            yolo2_box_t a = dets[i].bbox;
            for (int j = i + 1; j < total; ++j) {
                yolo2_box_t b = dets[j].bbox;
                if (box_iou(a, b) > nms_thresh) {
                    dets[j].prob[cls] = 0.0f;
                }
            }
        }
    }
}

/**
 * Free detections
 */
void yolo2_free_detections(yolo2_detection_t *dets, int n) {
    if (!dets) return;
    
    for (int i = 0; i < n; ++i) {
        if (dets[i].prob) {
            free(dets[i].prob);
            dets[i].prob = NULL;
        }
    }
}

/**
 * Print detections
 */
void yolo2_print_detections(yolo2_detection_t *dets, int n, float thresh, const char **labels, int num_labels) {
    if (!dets || n <= 0) {
        printf("No detections found\n");
        return;
    }
    
    printf("\n========================================\n");
    printf("Detections (thresh=%.2f):\n", thresh);
    printf("========================================\n");
    
    int printed = 0;
    for (int i = 0; i < n; ++i) {
        int best_class = -1;
        float best_prob = 0.0f;
        
        for (int cls = 0; cls < dets[i].classes; ++cls) {
            if (dets[i].prob[cls] > best_prob) {
                best_prob = dets[i].prob[cls];
                best_class = cls;
            }
        }
        
        if (best_prob > thresh && best_class >= 0) {
            const char *name = (best_class < num_labels && labels) ? labels[best_class] : "unknown";
            yolo2_box_t b = dets[i].bbox;
            
            printf("  %-16s prob=%.2f%% box=[x=%.3f y=%.3f w=%.3f h=%.3f]\n",
                   name, best_prob * 100.0f, b.x, b.y, b.w, b.h);
            printed++;
        }
    }
    
    if (printed == 0) {
        printf("  No detections above threshold\n");
    } else {
        printf("\nTotal: %d detections\n", printed);
    }
    printf("========================================\n\n");
}
