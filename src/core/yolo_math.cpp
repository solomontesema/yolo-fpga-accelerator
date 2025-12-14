#include <core/yolo.h>

#include <cmath>
#include <cstring>

static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static inline float loggy_activate(float x){return 2./(1. + exp(-x)) - 1;}
static inline float relu_activate(float x){return x*(x>0);}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline float plse_activate(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float){return 1;}
static inline float logistic_gradient(float x){return (1-x)*x;}
static inline float loggy_gradient(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient(float x){return (x>0);}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient(float x){return (x>0)+.1;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
static inline float tanh_gradient(float x){return 1-x*x;}
static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:  return (char *)"logistic";
        case LOGGY:     return (char *)"loggy";
        case RELU:      return (char *)"relu";
        case ELU:       return (char *)"elu";
        case RELIE:     return (char *)"relie";
        case RAMP:      return (char *)"ramp";
        case LINEAR:    return (char *)"linear";
        case TANH:      return (char *)"tanh";
        case PLSE:      return (char *)"plse";
        case LEAKY:     return (char *)"leaky";
        case STAIR:     return (char *)"stair";
        case HARDTAN:   return (char *)"hardtan";
        case LHTAN:     return (char *)"lhtan";
        default:        break;
    }
    return (char *)"relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:   return linear_activate(x);
        case LOGISTIC: return logistic_activate(x);
        case LOGGY:    return loggy_activate(x);
        case RELU:     return relu_activate(x);
        case ELU:      return elu_activate(x);
        case RELIE:    return relie_activate(x);
        case RAMP:     return ramp_activate(x);
        case LEAKY:    return leaky_activate(x);
        case TANH:     return tanh_activate(x);
        case PLSE:     return plse_activate(x);
        case STAIR:    return stair_activate(x);
        case HARDTAN:  return hardtan_activate(x);
        case LHTAN:    return lhtan_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    for(int i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:   return linear_gradient(x);
        case LOGISTIC: return logistic_gradient(x);
        case LOGGY:    return loggy_gradient(x);
        case RELU:     return relu_gradient(x);
        case ELU:      return elu_gradient(x);
        case RELIE:    return relie_gradient(x);
        case RAMP:     return ramp_gradient(x);
        case LEAKY:    return leaky_gradient(x);
        case TANH:     return tanh_gradient(x);
        case PLSE:     return plse_gradient(x);
        case STAIR:    return stair_gradient(x);
        case HARDTAN:  return hardtan_gradient(x);
        case LHTAN:    return lhtan_gradient(x);
    }
    return 0;
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    for(int i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    for(int i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    for(int b = 0; b < batch; ++b){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    (void)channels;
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col)
{
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (int h = 0; h < height_col; ++h) {
            for (int w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    float sum = 0;
    float largest = -FLT_MAX;
    for(int i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(int i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(int i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    for(int b = 0; b < batch; ++b){
        for(int g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    for(int b = 0; b < batch; ++b){
        for(int f = 0; f < filters; ++f){
            float m = mean[f];
            float v = variance[f];
            float denom = std::sqrt(v) + 0.000001f;
            for(int i = 0; i < spatial; ++i){
                int idx = b*filters*spatial + f*spatial + i;
                x[idx] = (x[idx] - m) / denom;
            }
        }
    }
}
