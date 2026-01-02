/**
 * YOLOv2 Label Loader
 */

#ifndef YOLO2_LABELS_H
#define YOLO2_LABELS_H

#include <stddef.h>

/**
 * Load class labels from file
 * 
 * labels_path: Path to labels file (one label per line)
 * labels: Pointer to receive array of label strings
 * num_labels: Pointer to receive number of labels
 * 
 * Returns: Number of labels loaded, or -1 on error
 */
int yolo2_load_labels(const char *labels_path, char ***labels, int *num_labels);

/**
 * Free labels array
 */
void yolo2_free_labels(char **labels, int num_labels);

#endif /* YOLO2_LABELS_H */
