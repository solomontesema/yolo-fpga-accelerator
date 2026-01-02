/**
 * YOLOv2 Post-Processing - NMS and Detection Extraction
 */

#ifndef YOLO2_POSTPROCESS_H
#define YOLO2_POSTPROCESS_H

#include <stdint.h>
#include <stddef.h>
#include "yolo2_network.h"

// Bounding box structure
typedef struct {
    float x;  // Center x (normalized 0-1)
    float y;  // Center y (normalized 0-1)
    float w;  // Width (normalized 0-1)
    float h;  // Height (normalized 0-1)
} yolo2_box_t;

// Detection structure
typedef struct {
    yolo2_box_t bbox;
    float objectness;      // Object confidence
    float *prob;           // Class probabilities array
    int classes;           // Number of classes
    int sort_class;        // For NMS sorting
} yolo2_detection_t;

/**
 * Forward region layer (apply sigmoid activations)
 */
int yolo2_forward_region_layer(layer_t *l, float *input, float *output);

/**
 * Get detections from region layer output
 */
int yolo2_get_region_detections(layer_t *l, float *output, 
                                int img_w, int img_h, int net_w, int net_h,
                                float thresh, yolo2_detection_t *dets, int max_dets);

/**
 * Non-maximum suppression
 */
void yolo2_do_nms_sort(yolo2_detection_t *dets, int total, int classes, float nms_thresh);

/**
 * Free detections array
 */
void yolo2_free_detections(yolo2_detection_t *dets, int n);

/**
 * Print detections to console
 */
void yolo2_print_detections(yolo2_detection_t *dets, int n, float thresh, const char **labels, int num_labels);

#endif /* YOLO2_POSTPROCESS_H */
