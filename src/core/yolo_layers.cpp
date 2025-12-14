#include <core/yolo.h>
#include <core/yolo_cfg.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

size_t get_workspace_size(layer l)
{
    return static_cast<size_t>(l.out_h) * l.out_w * l.size * l.size * l.c / l.groups * sizeof(float);
}

} // namespace

int convolutional_out_height(layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
    layer l;
    memset(&l, 0, sizeof(layer));
    l.type = CONVOLUTIONAL;
    (void)adam;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    l.out_w = convolutional_out_width(l);
    l.out_h = convolutional_out_height(l);
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    std::fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n",
                 n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c,
                 (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
    return l;
}

layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
    std::fprintf(stderr,"route ");
    layer l;
    memset(&l,0,sizeof(layer));

    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    int outputs = 0;
    for(int i = 0; i < n; ++i){
        std::fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    std::fprintf(stderr, "\n");
    l.outputs = outputs;
    l.inputs = outputs;
    l.forward = NULL;
    return l;
}

layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    const char *activation_s = option_find_str(options, "activation", (char *)"logistic");
    ACTIVATION activation = get_activation(const_cast<char *>(activation_s));

    int h = params.h;
    int w = params.w;
    int c = params.c;
    int batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    layer l = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net->adam);
    l.flipped = option_find_int_quiet(options, "flipped", 0);
    l.dot = option_find_float_quiet(options, "dot", 0);

    return l;
}

layer parse_route(list *options, size_params params, network *net)
{
    char *l = option_find(options, "layers");
    if(!l) error("Route Layer must specify input layers");
    int len = strlen(l);
    int n = 1;
    for(int i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int *layers = (int *)calloc(n, sizeof(int));
    int *sizes = (int *)calloc(n, sizeof(int));
    for(int i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    layer route_layer = make_route_layer(batch, n, layers, sizes);

    layer first = net->layers[layers[0]];
    route_layer.out_w = first.out_w;
    route_layer.out_h = first.out_h;
    route_layer.out_c = first.out_c;
    for(int i = 1; i < n; ++i){
        int index = layers[i];
        layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            route_layer.out_c += next.out_c;
        }else{
            route_layer.out_h = route_layer.out_w = route_layer.out_c = 0;
        }
    }

    return route_layer;
}

layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + coords + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.coords = coords;
    l.biases = (float *)calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30*(l.coords + 1);
    l.output = (float *)calloc(batch*l.outputs, sizeof(float));
    for(int i = 0; i < n*2; ++i){
        l.biases[i] = .5f;
    }
    std::fprintf(stderr, "detection\n");
    srand(0);
    return l;
}

layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
    assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.background = option_find_int_quiet(options, "background", 0);
    l.max_boxes = option_find_int_quiet(options, "max",30);
    l.jitter = option_find_float(options, "jitter", .2f);
    l.rescore = option_find_int_quiet(options, "rescore",0);

    l.thresh = option_find_float(options, "thresh", .5f);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_int_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.mask_scale = option_find_float(options, "mask_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match",0);

    char *a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        for(int i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(int i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}

layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra)
{
    layer l;
    memset(&l,0,sizeof(layer));

    l.type = REORG;
    l.batch = batch;
    l.stride = stride;
    l.extra = extra;
    l.h = h;
    l.w = w;
    l.c = c;
    l.flatten = flatten;
    if(reverse){
        l.out_w = w*stride;
        l.out_h = h*stride;
        l.out_c = c/(stride*stride);
    }else{
        l.out_w = w/stride;
        l.out_h = h/stride;
        l.out_c = c*(stride*stride);
    }
    l.reverse = reverse;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    if(l.extra){
        l.out_w = l.out_h = l.out_c = 0;
        l.outputs = l.inputs + l.extra;
    }

    if(extra){
        std::fprintf(stderr, "reorg              %4d   ->  %4d\n",  l.inputs, l.outputs);
    } else {
        std::fprintf(stderr, "reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  stride, w, h, c, l.out_w, l.out_h, l.out_c);
    }
    return l;
}

layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int reverse = option_find_int_quiet(options, "reverse",0);
    int flatten = option_find_int_quiet(options, "flatten",0);
    int extra = option_find_int_quiet(options, "extra",0);

    int h = params.h;
    int w = params.w;
    int c = params.c;
    int batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.");

    layer layer = make_reorg_layer(batch,w,h,c,stride,reverse, flatten, extra);
    return layer;
}

layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    layer l;
    memset(&l,0,sizeof(layer));
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;

    std::fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",
                 size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int h = params.h;
    int w = params.w;
    int c = params.c;
    int batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    layer maxpool_layer = make_maxpool_layer(batch,h,w,c,size,stride,padding);
    return maxpool_layer;
}

void free_layer(layer l)
{
    if(l.cweights)           free(l.cweights);
    if(l.indexes)            free(l.indexes);
    if(l.input_layers)       free(l.input_layers);
    if(l.input_sizes)        free(l.input_sizes);
    if(l.map)                free(l.map);
    if(l.rand)               free(l.rand);
    if(l.cost)               free(l.cost);
    if(l.state)              free(l.state);
    if(l.prev_state)         free(l.prev_state);
    if(l.forgot_state)       free(l.forgot_state);
    if(l.forgot_delta)       free(l.forgot_delta);
    if(l.state_delta)        free(l.state_delta);
    if(l.concat)             free(l.concat);
    if(l.concat_delta)       free(l.concat_delta);
    if(l.binary_weights)     free(l.binary_weights);
    if(l.biases)             free(l.biases);
    if(l.bias_updates)       free(l.bias_updates);
    if(l.scales)             free(l.scales);
    if(l.scale_updates)      free(l.scale_updates);
    if(l.weights)            free(l.weights);
    if(l.weight_updates)     free(l.weight_updates);
    if(l.delta)              free(l.delta);
    if(l.output)             free(l.output);
    if(l.squared)            free(l.squared);
    if(l.norms)              free(l.norms);
    if(l.spatial_mean)       free(l.spatial_mean);
    if(l.mean)               free(l.mean);
    if(l.variance)           free(l.variance);
    if(l.mean_delta)         free(l.mean_delta);
    if(l.variance_delta)     free(l.variance_delta);
    if(l.rolling_mean)       free(l.rolling_mean);
    if(l.rolling_variance)   free(l.rolling_variance);
    if(l.x)                  free(l.x);
    if(l.x_norm)             free(l.x_norm);
    if(l.m)                  free(l.m);
    if(l.v)                  free(l.v);
    if(l.z_cpu)              free(l.z_cpu);
    if(l.r_cpu)              free(l.r_cpu);
    if(l.h_cpu)              free(l.h_cpu);
    if(l.binary_input)       free(l.binary_input);
}
