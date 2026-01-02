/**
 * YOLOv2 Label Loader - Linux Implementation
 */

#include "yolo2_labels.h"
#include "yolo2_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Load class labels from file
 */
int yolo2_load_labels(const char *labels_path, char ***labels, int *num_labels) {
    FILE *file;
    char line[256];
    int count = 0;
    int capacity = 100;
    char **label_array = NULL;
    
    if (!labels_path || !labels || !num_labels) {
        return -1;
    }
    
    file = fopen(labels_path, "r");
    if (!file) {
        fprintf(stderr, "ERROR: Cannot open labels file: %s\n", labels_path);
        return -1;
    }
    
    // Allocate initial array
    label_array = (char**)malloc(capacity * sizeof(char*));
    if (!label_array) {
        fprintf(stderr, "ERROR: Failed to allocate label array\n");
        fclose(file);
        return -1;
    }
    
    // Read labels line by line
    while (fgets(line, sizeof(line), file) != NULL) {
        // Strip whitespace
        int len = strlen(line);
        while (len > 0 && (line[len-1] == ' ' || line[len-1] == '\t' || 
                          line[len-1] == '\r' || line[len-1] == '\n')) {
            line[--len] = '\0';
        }
        
        // Skip empty lines
        if (len == 0) continue;
        
        // Resize array if needed
        if (count >= capacity) {
            capacity *= 2;
            char **new_array = (char**)realloc(label_array, capacity * sizeof(char*));
            if (!new_array) {
                fprintf(stderr, "ERROR: Failed to reallocate label array\n");
                for (int i = 0; i < count; ++i) {
                    free(label_array[i]);
                }
                free(label_array);
                fclose(file);
                return -1;
            }
            label_array = new_array;
        }
        
        // Allocate and copy label string
        label_array[count] = (char*)malloc((len + 1) * sizeof(char));
        if (!label_array[count]) {
            fprintf(stderr, "ERROR: Failed to allocate label string\n");
            for (int i = 0; i < count; ++i) {
                free(label_array[i]);
            }
            free(label_array);
            fclose(file);
            return -1;
        }
        strcpy(label_array[count], line);
        count++;
    }
    
    fclose(file);
    
    *labels = label_array;
    *num_labels = count;
    
    YOLO2_LOG_INFO("Loaded %d class labels from %s\n", count, labels_path);
    
    return count;
}

/**
 * Free labels array
 */
void yolo2_free_labels(char **labels, int num_labels) {
    if (!labels) return;
    
    for (int i = 0; i < num_labels; ++i) {
        if (labels[i]) {
            free(labels[i]);
        }
    }
    free(labels);
}
