#pragma once

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>
#include <fcntl.h>

#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

// -----------------------------------------------------------------------------
// Model enums
// -----------------------------------------------------------------------------
typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR,
    HARDTAN, LHTAN
} ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

// -----------------------------------------------------------------------------
// Layer / network types
// -----------------------------------------------------------------------------
struct network;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    void (*forward)(struct layer, struct network);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontsave;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;
    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;

    float * z_cpu;
    float * r_cpu;
    float * h_cpu;
    float * hh_cpu;
    float * prev_cell_cpu;
    float * cell_cpu;
    float * f_cpu;
    float * i_cpu;
    float * g_cpu;
    float * o_cpu;
    float * c_cpu;
    float * dc_cpu;

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    size_t workspace_size;
};

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

struct network{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;
    int gpu_index;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;
};

// -----------------------------------------------------------------------------
// Data containers
// -----------------------------------------------------------------------------
typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct{
    float x, y;
    float w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA,
    COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA,
    STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA,
    SEGMENTATION_DATA, INSTANCE_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    struct network *net;
} size_params;

// -----------------------------------------------------------------------------
// Public API (declarations)
// -----------------------------------------------------------------------------
list *make_list();
void free_list(list *l);
void free_list_contents(list *l);
void **list_to_array(list *l);
void list_insert(list *l, void *val);
void del_arg(int argc, char **argv, int index);
int find_arg(int argc, char* argv[], const char *arg);
int find_int_arg(int argc, char **argv, const char *arg, int def);
float find_float_arg(int argc, char **argv, const char *arg, float def);
char *find_char_arg(int argc, char **argv, const char *arg, char *def);
unsigned char *read_file(char *filename);
list *split_str(char *s, char delim);
void strip(char *s);
void strip_char(char *s, char bad);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);

void free_layer(layer l);
void free_image(image m);
void free_detections(detection *dets, int n);
void error(const char *s);

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
char *get_activation_string(ACTIVATION a);
ACTIVATION get_activation(char *s);
float activate(float x, ACTIVATION a);
void activate_array(float *x, int n, ACTIVATION a);
float gradient(float x, ACTIVATION a);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void add_bias(float *output, float *biases, int batch, int n, int size);
void scale_bias(float *output, float *scales, int batch, int n, int size);
void im2col_cpu(float* data_im, int channels, int height, int width, int ksize, int stride, int pad, float* data_col);
void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
float *network_predict(network *net, float *input);
float *get_network_output(network *net);
layer get_network_output_layer(network *net);
int get_network_output_size(network *net);

void forward_region_layer(const layer l, float *net_input);

void set_batch_network(network *net, int b);
network *load_network(char *cfgfile);
layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
layer parse_convolutional(list *options, size_params params);
layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes);
layer parse_route(list *options, size_params params, network *net);
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords);
layer parse_region(list *options, size_params params);
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
layer parse_reorg(list *options, size_params params);
layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
layer parse_maxpool(list *options, size_params params);

image **load_alphabet();
image make_empty_image(int w, int h, int c);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void fill_image(image m, float s);
void embed_image(image source, image dest, int dx, int dy);
void fill_cpu(int N, float ALPHA, float *X, int INCX);
float get_pixel(image m, int x, int y, int c);
void set_pixel(image m, int x, int y, int c, float val);
image load_image_stb(char *filename, int channels);
image letterbox_image(image im, int w, int h);
void save_image_png(image im, const char *name);

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void do_nms_sort(detection *dets, int total, int classes, float thresh);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

void file_error(char *s);
