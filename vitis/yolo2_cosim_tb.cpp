#include "../hls/models/yolov2/yolo2_accel.hpp"
#include "../hls/models/yolov2/model_config.hpp"
#include "../hls/core/params.hpp"
#include "../hls/core/types.hpp"
#include "../include/core/yolo.h"
#include "../include/core/precision.hpp"

// Constants are already defined in params.hpp, no need to redefine

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>

// Helper to read binary files
template <typename T>
std::vector<T> read_binary(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (sz < 0 || sz % sizeof(T) != 0) {
        std::fclose(fp);
        throw std::runtime_error("Invalid size for file: " + path);
    }
    std::vector<T> buf(sz / sizeof(T));
    size_t rd = std::fread(buf.data(), sizeof(T), buf.size(), fp);
    std::fclose(fp);
    if (rd != buf.size()) {
        throw std::runtime_error("Short read: " + path);
    }
    return buf;
}

// Helper to write binary files
template <typename T>
void write_binary(const std::string &path, const T *data, size_t count) {
    FILE *fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    size_t written = std::fwrite(data, sizeof(T), count, fp);
    std::fclose(fp);
    if (written != count) {
        throw std::runtime_error("Short write: " + path);
    }
}

// Load label lines (simple helper, ignores empty lines)
std::vector<std::string> load_label_lines(const std::string &path) {
    std::vector<std::string> labels;
    FILE *fp = std::fopen(path.c_str(), "r");
    if (!fp) {
        return labels;
    }
    char line[512];
    while (std::fgets(line, sizeof(line), fp)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        if (!s.empty()) {
            labels.push_back(s);
        }
    }
    std::fclose(fp);
    return labels;
}

// Memory offset generation (from yolo2_model.cpp)
void generate_iofm_offset(IO_Dtype* in_ptr[32], IO_Dtype* out_ptr[32], 
                          IO_Dtype *Memory_buf, network *net, const ModelConfig &cfg) {
    IO_Dtype *Memory_top = Memory_buf + 512;
    IO_Dtype *Memory_bottom = Memory_top + cfg.mem_len;
    
    for(int x = 0; x < 18; x++) {
        int out_w = net->layers[x].out_w;
        int out_w_align_256b = (out_w >> 3) << 3;
        if(out_w & 0x7)
            out_w_align_256b += 8;

        if(x % 2 == 0) {
            in_ptr[x] = Memory_top;
            out_ptr[x] = Memory_bottom - net->layers[x].out_c * net->layers[x].out_h * out_w_align_256b;
        } else {
            in_ptr[x] = out_ptr[x-1];
            out_ptr[x] = Memory_top;
        }
    }

    for(int x = 18; x < 25; x++) {
        int out_w = net->layers[x].out_w;
        int out_w_align_256b = (out_w >> 3) << 3;
        if(out_w & 0x7)
            out_w_align_256b += 8;

        if(x % 2 == 0) {
            in_ptr[x] = Memory_top;
            out_ptr[x] = Memory_bottom - cfg.route16_len - net->layers[x].out_c * net->layers[x].out_h * out_w_align_256b;
        } else {
            in_ptr[x] = out_ptr[x-1];
            out_ptr[x] = Memory_top;
        }
    }

    in_ptr[26] = Memory_bottom - cfg.route16_len;
    out_ptr[26] = Memory_top;

    in_ptr[27] = Memory_top;
    out_ptr[27] = Memory_bottom - cfg.route16_len - cfg.conv24_len - cfg.conv27_len;

    in_ptr[29] = out_ptr[27];
    out_ptr[29] = Memory_top;

    in_ptr[30] = Memory_top;
    out_ptr[30] = Memory_bottom - (net->layers[30].outputs + cfg.detection_workspace);

    in_ptr[31] = out_ptr[30];
}

// Reorg CPU implementation (from yolo2_model.cpp)
void reorg_cpu(IO_Dtype *x, int w, int h, int c, int stride, IO_Dtype *out) {
    int out_c = c / (stride * stride);

    for(int k = 0; k < c; ++k) {
        for(int j = 0; j < h; ++j) {
            for(int i = 0; i < w; ++i) {
                int in_index  = i + w*(j + h*k);
                int c2 = k % out_c;
                int offset = k / out_c;
                int w2 = i*stride + offset % stride;
                int h2 = j*stride + offset / stride;
                int out_index = w2 + w*stride*(h2 + h*stride*c2);
                out[in_index] = x[out_index];
            }
        }
    }
}

// Helper function to join paths
std::string join_path(const std::string& base, const std::string& part) {
    if (base.empty()) return part;
    if (base.back() == '/') return base + part;
    return base + "/" + part;
}

// Helper function to check if file exists
bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// Helper function to find project root by looking for config/yolov2.cfg
std::string find_project_root() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        return ".";
    }
    
    std::string current = cwd;
    
    // Try going up directories to find config/yolov2.cfg
    for (int i = 0; i < 10; ++i) {
        std::string config_file = join_path(current, "config/yolov2.cfg");
        if (file_exists(config_file)) {
            return current;
        }
        
        // Go up one directory
        size_t last_slash = current.find_last_of('/');
        if (last_slash == std::string::npos || last_slash == 0) {
            break; // Reached filesystem root
        }
        current = current.substr(0, last_slash);
    }
    
    // Fallback: try relative to current directory
    return ".";
}

// Helper to check if path is absolute
bool is_absolute_path(const std::string& path) {
    return !path.empty() && path[0] == '/';
}

int main(int argc, char *argv[]) {
    printf("YOLO2_FPGA Co-Simulation Testbench\n");
    printf("====================================\n\n");

    // Find project root
    std::string proj_root = find_project_root();
    if (proj_root != ".") {
        printf("Found project root: %s\n", proj_root.c_str());
        // Change to project root so load_alphabet() can find data/labels/
        if (chdir(proj_root.c_str()) != 0) {
            printf("WARNING: Failed to change to project root. Alphabet images may not load.\n");
        }
    }
    
    // Parse command line arguments (use absolute paths if project root found)
    std::string img_path = (proj_root != ".") ? 
        join_path(proj_root, "examples/test_images/dog.jpg") :
        "examples/test_images/dog.jpg";
    std::string cfg_path = (proj_root != ".") ?
        join_path(proj_root, "config/yolov2.cfg") :
        "config/yolov2.cfg";
    std::string weights_dir = (proj_root != ".") ?
        join_path(proj_root, "weights") :
        "weights";
    std::string output_dir = (proj_root != ".") ?
        join_path(proj_root, "cosim_output") :
        "cosim_output";
    
    if (argc > 1) {
        img_path = argv[1];
        if (!is_absolute_path(img_path) && proj_root != ".") {
            img_path = join_path(proj_root, img_path);
        }
    }
    if (argc > 2) {
        cfg_path = argv[2];
        if (!is_absolute_path(cfg_path) && proj_root != ".") {
            cfg_path = join_path(proj_root, cfg_path);
        }
    }
    if (argc > 3) {
        weights_dir = argv[3];
        if (!is_absolute_path(weights_dir) && proj_root != ".") {
            weights_dir = join_path(proj_root, weights_dir);
        }
    }
    if (argc > 4) {
        output_dir = argv[4];
        if (!is_absolute_path(output_dir) && proj_root != ".") {
            output_dir = join_path(proj_root, output_dir);
        }
    }

    printf("Configuration:\n");
    printf("  Image:      %s\n", img_path.c_str());
    printf("  Config:     %s\n", cfg_path.c_str());
    printf("  Weights:    %s\n", weights_dir.c_str());
    printf("  Output:     %s\n", output_dir.c_str());
    printf("\n");

    // Load network configuration
    printf("Loading network configuration...\n");
    network *net = load_network(const_cast<char *>(cfg_path.c_str()));
    if (!net) {
        fprintf(stderr, "ERROR: Failed to load network from %s\n", cfg_path.c_str());
        return 1;
    }
    set_batch_network(net, 1);
    printf("Network loaded: %d layers\n", net->n);

    // Load and preprocess image
    printf("Loading and preprocessing image...\n");
    image im = load_image_stb(const_cast<char *>(img_path.c_str()), 3);
    if (!im.data) {
        fprintf(stderr, "ERROR: Failed to load image from %s\n", img_path.c_str());
        return 1;
    }
    printf("Original image: w=%d, h=%d, c=%d\n", im.w, im.h, im.c);

    image sized = letterbox_image(im, 416, 416);
    printf("Letterboxed image: w=%d, h=%d, c=%d\n", sized.w, sized.h, sized.c);

    // Verify dimensions
    if (sized.w != 416 || sized.h != 416 || sized.c != 3) {
        fprintf(stderr, "ERROR: Letterboxed image dimensions incorrect: %dx%dx%d\n", 
                sized.w, sized.h, sized.c);
        return 1;
    }

    // Load weights and bias
    printf("Loading weights and bias...\n");
    const ModelConfig &cfg = yolo2_model_config();
    
    int conv_layers = 0;
    for (int i = 0; i < net->n; ++i) {
        if (net->layers[i].type == CONVOLUTIONAL) conv_layers++;
    }

    size_t expected_w = 0;
    size_t expected_b = 0;
    for (int i = 0; i < conv_layers && i < static_cast<int>(cfg.weight_offsets.size()); ++i) {
        expected_w += cfg.weight_offsets[i];
        expected_b += cfg.beta_offsets[i];
    }
    // CRITICAL: The HLS depth must match the ACTUAL file size, not just weight_offsets sum
    // The extractor writes ALL convolutional layers, which may exceed weight_offsets
    // We'll update expected_w/b to match actual file sizes after reading them
    const size_t axi_weight_depth = 50941792; // words, from pragma depth (minimum)
    const size_t axi_beta_depth   = 10761;    // words, from pragma depth (minimum)
    if (expected_w < axi_weight_depth) expected_w = axi_weight_depth;
    if (expected_b < axi_beta_depth)   expected_b = axi_beta_depth;

    // Select correct weight files based on INT16_MODE
#ifdef INT16_MODE
    std::string weights_path = join_path(weights_dir, "weights_reorg_int16.bin");
    std::string bias_path = join_path(weights_dir, "bias_int16.bin");
    printf("Loading weights and bias (INT16 mode)...\n");
#else
    std::string weights_path = join_path(weights_dir, "weights_reorg.bin");
    std::string bias_path = join_path(weights_dir, "bias.bin");
    printf("Loading weights and bias (FP32 mode)...\n");
#endif

    printf("  Weights file: %s\n", weights_path.c_str());
    printf("  Bias file: %s\n", bias_path.c_str());
    printf("  Expected weights: %zu elements (%zu bytes)\n", expected_w, expected_w * sizeof(IO_Dtype));
    printf("  Expected bias: %zu elements (%zu bytes)\n", expected_b, expected_b * sizeof(IO_Dtype));

    // Read with correct data type based on IO_Dtype
    auto w_buf = read_binary<IO_Dtype>(weights_path);
    auto b_buf = read_binary<IO_Dtype>(bias_path);

    printf("  Loaded weights: %zu elements\n", w_buf.size());
    printf("  Loaded bias: %zu elements\n", b_buf.size());

    // CRITICAL: Use actual file size, not just expected_w from weight_offsets
    // The extractor writes ALL convolutional layers, which may be more than weight_offsets accounts for
    // The HLS depth must match the actual file size to prevent wrapc from reading beyond allocated buffer
    size_t actual_w = w_buf.size();
    size_t actual_b = b_buf.size();
    
    // Update expected sizes to match actual file sizes (but ensure minimum depth)
    if (actual_w > expected_w) {
        printf("  WARNING: Weight file has %zu elements, but weight_offsets sum is %zu\n", 
               actual_w, expected_w);
        printf("  Using actual file size to match HLS depth\n");
        expected_w = actual_w;
    }
    if (actual_b > expected_b) {
        printf("  WARNING: Bias file has %zu elements, but beta_offsets sum is %zu\n", 
               actual_b, expected_b);
        printf("  Using actual file size to match HLS depth\n");
        expected_b = actual_b;
    }

    if (w_buf.size() < expected_w) {
        fprintf(stderr, "ERROR: weights file too small: got %zu, expected %zu\n", 
                w_buf.size(), expected_w);
#ifdef INT16_MODE
        fprintf(stderr, "  Make sure weights_reorg_int16.bin was generated from weights.bin using yolov2_weight_gen --precision int16\n");
#else
        fprintf(stderr, "  Make sure weights_reorg.bin was generated from weights.bin using yolov2_weight_gen\n");
#endif
        return 1;
    }
    if (b_buf.size() < expected_b) {
        fprintf(stderr, "ERROR: bias file too small: got %zu, expected %zu\n", 
                b_buf.size(), expected_b);
#ifdef INT16_MODE
        fprintf(stderr, "  Make sure bias_int16.bin has batch normalization folded (generated by weights_extractor --int16)\n");
#else
        fprintf(stderr, "  Make sure bias.bin has batch normalization folded (generated by weights_extractor)\n");
#endif
        return 1;
    }
    
    // Note: File size may be larger than weight_offsets sum if extractor writes all layers
    // The testbench now uses actual file size to ensure buffer allocation matches

    // Allocate weights and bias on heap with page alignment for AXI co-simulation compatibility
    // AXI interfaces typically require 4KB (4096 byte) alignment
    const size_t alignment = 4096;
    IO_Dtype *Weight_buf = nullptr;
    IO_Dtype *Beta_buf = nullptr;
    
    if (posix_memalign((void**)&Weight_buf, alignment, expected_w * sizeof(IO_Dtype)) != 0) {
        fprintf(stderr, "ERROR: Failed to allocate aligned weights buffer\n");
        return 1;
    }
    if (posix_memalign((void**)&Beta_buf, alignment, expected_b * sizeof(IO_Dtype)) != 0) {
        fprintf(stderr, "ERROR: Failed to allocate aligned bias buffer\n");
        free(Weight_buf);
        return 1;
    }
    
    // Zero-initialize and copy data
    memset(Weight_buf, 0, expected_w * sizeof(IO_Dtype));
    memset(Beta_buf, 0, expected_b * sizeof(IO_Dtype));
    memcpy(Weight_buf, w_buf.data(), expected_w * sizeof(IO_Dtype));
    memcpy(Beta_buf, b_buf.data(), expected_b * sizeof(IO_Dtype));
    
    // Touch memory pages to ensure they're mapped (helps with co-simulation)
    volatile IO_Dtype dummy_w = 0, dummy_b = 0;
    for (size_t i = 0; i < expected_w; i += 4096 / sizeof(IO_Dtype)) {
        dummy_w += Weight_buf[i];
        Weight_buf[i] = Weight_buf[i];  // Ensure page is mapped
    }
    for (size_t i = 0; i < expected_b; i += 4096 / sizeof(IO_Dtype)) {
        dummy_b += Beta_buf[i];
        Beta_buf[i] = Beta_buf[i];  // Ensure page is mapped
    }
    (void)dummy_w; (void)dummy_b;  // Suppress unused variable warnings

    printf("Weights loaded: %zu elements at %p\n", expected_w, (void*)Weight_buf);
    printf("Bias loaded: %zu elements at %p\n", expected_b, (void*)Beta_buf);

    // Load Q values for INT16 mode
    std::vector<int32_t> weight_q, bias_q, act_q;
#ifdef INT16_MODE
    printf("Loading INT16 quantization Q values...\n");
    try {
        std::string weight_q_path = join_path(weights_dir, "weight_int16_Q.bin");
        std::string bias_q_path = join_path(weights_dir, "bias_int16_Q.bin");
        std::string iofm_q_path = join_path(weights_dir, "iofm_Q.bin");
        
        weight_q = read_binary<int32_t>(weight_q_path);
        bias_q = read_binary<int32_t>(bias_q_path);
        act_q = read_binary<int32_t>(iofm_q_path);
        
        printf("  Weight Q values: %zu entries\n", weight_q.size());
        printf("  Bias Q values: %zu entries\n", bias_q.size());
        printf("  Activation Q values (iofm): %zu entries\n", act_q.size());
        
        if (weight_q.size() < static_cast<size_t>(conv_layers)) {
            fprintf(stderr, "ERROR: Weight Q table too small: got %zu, expected %d\n", 
                    weight_q.size(), conv_layers);
            return 1;
        }
        if (bias_q.size() < static_cast<size_t>(conv_layers)) {
            fprintf(stderr, "ERROR: Bias Q table too small: got %zu, expected %d\n", 
                    bias_q.size(), conv_layers);
            return 1;
        }
        if (act_q.empty()) {
            fprintf(stderr, "ERROR: Activation Q table (iofm_Q.bin) is required for INT16 mode\n");
            return 1;
        }
        
        // Handle special case for route layer (from old implementation)
        if (act_q.size() > 21) {
            if (act_q[20] < act_q[21]) {
                act_q[21] = act_q[20];
            } else {
                act_q[20] = act_q[21];
            }
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "ERROR: Failed to load Q values: %s\n", e.what());
        return 1;
    }
#endif

    // Allocate memory buffers with page alignment for AXI co-simulation
    printf("Allocating memory buffers...\n");
    size_t mem_size = cfg.mem_len + 512*2;
    size_t mem_bytes = mem_size * sizeof(IO_Dtype);
    printf("  Memory size: %zu elements (%zu bytes)\n", mem_size, mem_bytes);
    
    IO_Dtype *Memory_buf = nullptr;
    if (posix_memalign((void**)&Memory_buf, alignment, mem_bytes) != 0) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        free(Weight_buf);
        free(Beta_buf);
        return 1;
    }
    // Zero-initialize and touch all memory to ensure it's accessible
    memset(Memory_buf, 0, mem_bytes);
    // Touch memory pages to ensure they're mapped (helps with co-simulation)
    // CRITICAL: Touch ALL pages, not just every 4KB, to ensure entire buffer is mapped
    volatile IO_Dtype dummy = 0;
    for (size_t i = 0; i < mem_size; i += 1024) {  // Touch every 1024 words (4KB)
        dummy += Memory_buf[i];
        Memory_buf[i] = 0;  // Write back to ensure page is mapped
    }
    // Also ensure the entire Input range is accessible (hardware stub will read full depth)
    IO_Dtype *Memory_top = Memory_buf + 512;
    IO_Dtype *Memory_bottom = Memory_top + cfg.mem_len;
    for (size_t i = 0; i < cfg.mem_len; i += 1024) {
        dummy += Memory_top[i];
        Memory_top[i] = 0;
    }
    (void)dummy;  // Suppress unused variable warning
    printf("  Memory allocated at: %p (aligned to %zu bytes)\n", (void*)Memory_buf, alignment);

    IO_Dtype* in_ptr[32];
    IO_Dtype* out_ptr[32];
    generate_iofm_offset(in_ptr, out_ptr, Memory_buf, net, cfg);
    
    // Verify first layer pointers are valid and within bounds
    printf("  Verifying pointers and bounds...\n");
    IO_Dtype *Memory_end = Memory_buf + mem_size;
    
    printf("    Memory_buf = %p\n", (void*)Memory_buf);
    printf("    Memory_top = %p (offset %zu)\n", (void*)Memory_top, (size_t)(Memory_top - Memory_buf));
    printf("    Memory_bottom = %p (offset %zu)\n", (void*)Memory_bottom, (size_t)(Memory_bottom - Memory_buf));
    printf("    Memory_end = %p (offset %zu)\n", (void*)Memory_end, (size_t)(Memory_end - Memory_buf));
    printf("    in_ptr[0] = %p (offset %zu)\n", (void*)in_ptr[0], (size_t)(in_ptr[0] - Memory_buf));
    printf("    out_ptr[0] = %p (offset %zu)\n", (void*)out_ptr[0], (size_t)(out_ptr[0] - Memory_buf));
    
    if (!in_ptr[0] || !out_ptr[0]) {
        fprintf(stderr, "ERROR: Invalid pointers from generate_iofm_offset\n");
        return 1;
    }
    
    // Verify pointers are within allocated memory
    if (in_ptr[0] < Memory_buf || in_ptr[0] >= Memory_end) {
        fprintf(stderr, "ERROR: in_ptr[0] out of bounds: %p not in [%p, %p)\n",
                (void*)in_ptr[0], (void*)Memory_buf, (void*)Memory_end);
        return 1;
    }
    if (out_ptr[0] < Memory_buf || out_ptr[0] >= Memory_end) {
        fprintf(stderr, "ERROR: out_ptr[0] out of bounds: %p not in [%p, %p)\n",
                (void*)out_ptr[0], (void*)Memory_buf, (void*)Memory_end);
        return 1;
    }
    
    // Verify wrapc write bounds for layer 0
    // HLS Output depth = 5,537,792 words = 22,151,168 bytes
    const size_t output_depth_words = 5537792;
    const size_t output_depth_bytes = output_depth_words * sizeof(IO_Dtype);
    IO_Dtype *out_ptr0_end = out_ptr[0] + output_depth_words;
    printf("    out_ptr[0] write range: [%p, %p) (%zu words, %zu bytes)\n",
           (void*)out_ptr[0], (void*)out_ptr0_end, output_depth_words, output_depth_bytes);
    if (out_ptr0_end > Memory_end) {
        fprintf(stderr, "ERROR: wrapc write would exceed buffer: %p > %p (overflow: %zu bytes)\n",
                (void*)out_ptr0_end, (void*)Memory_end, (size_t)(out_ptr0_end - Memory_end) * sizeof(IO_Dtype));
        return 1;
    }
    
    // Verify wrapc read bounds for layer 0
    // HLS Input depth = 6,922,240 words = 27,688,960 bytes
    const size_t input_depth_words = 6922240;
    const size_t input_depth_bytes = input_depth_words * sizeof(IO_Dtype);
    IO_Dtype *in_ptr0_end = in_ptr[0] + input_depth_words;
    printf("    in_ptr[0] read range: [%p, %p) (%zu words, %zu bytes)\n",
           (void*)in_ptr[0], (void*)in_ptr0_end, input_depth_words, input_depth_bytes);
    if (in_ptr0_end > Memory_end) {
        fprintf(stderr, "ERROR: wrapc read would exceed buffer: %p > %p (overflow: %zu bytes)\n",
                (void*)in_ptr0_end, (void*)Memory_end, (size_t)(in_ptr0_end - Memory_end) * sizeof(IO_Dtype));
        return 1;
    }

    // Copy input image to first layer input buffer
    const int input_elems = 416 * 416 * 3;
    printf("  Copying input image (%d elements)...\n", input_elems);
    
#ifdef INT16_MODE
    // For INT16 mode, quantize the input image using first layer input Q value
    if (act_q.empty()) {
        fprintf(stderr, "ERROR: Activation Q table required for INT16 input quantization\n");
        return 1;
    }
    const int q_in = act_q[0];
    const double scale = std::pow(2.0, q_in);
    printf("  Quantizing input with Q=%d (scale=2^%d=%.6f)\n", q_in, q_in, scale);
    
    for (int idx = 0; idx < input_elems; ++idx) {
        double v = sized.data[idx] * scale;
        // Clamp to int16_t range
        if (v > 32767.0) v = 32767.0;
        if (v < -32768.0) v = -32768.0;
        int64_t q = static_cast<int64_t>(std::llround(v));
        if (q > 32767) q = 32767;
        if (q < -32768) q = -32768;
        in_ptr[0][idx] = static_cast<IO_Dtype>(q);
    }
#else
    // For FP32 mode, copy directly
    memcpy(in_ptr[0], sized.data, input_elems * sizeof(IO_Dtype));
#endif
    
    // CRITICAL: Ensure the entire Input buffer range is accessible and zero-initialized
    // The hardware stub will copy the full depth (6,922,240 words) from in_ptr[0]
    // Even though we only have 519,168 words of actual data, the rest must be accessible
    // Note: input_depth_words is already defined above in the bounds checking section
    if (in_ptr[0] + input_depth_words > Memory_end) {
        fprintf(stderr, "ERROR: Input buffer range exceeds allocated memory\n");
        return 1;
    }
    // Touch the entire Input range to ensure all pages are mapped
    volatile IO_Dtype dummy2 = 0;
    for (size_t i = input_elems; i < input_depth_words; i += 1024) {
        dummy2 += in_ptr[0][i];
        in_ptr[0][i] = 0;  // Ensure zeros and page mapping
    }
    (void)dummy2;
    printf("Input image copied to buffer (entire Input range %zu words is accessible)\n", input_depth_words);

    // Region buffers for reorg/route operations
    const int region_len = 13 * 16 * 425;
    std::vector<IO_Dtype> region_buf(region_len, 0);
    std::vector<IO_Dtype> region_buf2(region_len, 0);
    
    // Track Q values for route layer alignment (INT16 mode)
#ifdef INT16_MODE
    int route24_q = 0;  // Q value from layer 24 (route source)
    int current_Qa = (!act_q.empty()) ? act_q[0] : 0;  // Current activation Q
    int pending_route_q = -1;  // Q value to use for next layer after route
#endif

    // Run inference
    printf("\nStarting inference...\n");
    printf("Running through %d layers...\n", net->n);

    int offset_index = 0;
    int woffset = 0;
    int boffset = 0;
    int TR, TC, TM, TN;
    int output_w, output_h;
    int mLoops;
    IO_Dtype* tmp_ptr_f0 = nullptr;

    auto start_time = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        
        switch(l.type) {
            case CONVOLUTIONAL: {
                output_w = (l.w - l.size + 2*l.pad) / l.stride + 1;
                output_h = (l.h - l.size + 2*l.pad) / l.stride + 1;

                TR = std::min(((OnChipIB_Height - l.size) / l.stride + 1), Tr);
                TR = std::min(output_h, TR);
                TC = std::min(((OnChipIB_Width - l.size) / l.stride + 1), Tc);
                TC = std::min(output_w, TC);
                TM = std::min(l.n, Tm);
                TN = std::min(l.c, Tn);
                mLoops = (int)ceil(((float)l.n) / TM);

                printf("  Layer %2d: CONV  IFM=%3d OFM=%3d K=%d S=%d P=%d -> %dx%d (TM=%d TN=%d TR=%d TC=%d)\n",
                       i, l.c, l.n, l.size, l.stride, l.pad, output_w, output_h, TM, TN, TR, TC);
                
                // Verify pointers are valid before calling
                if (!in_ptr[i]) {
                    fprintf(stderr, "ERROR: Layer %d: in_ptr[%d] is NULL\n", i, i);
                    return 1;
                }
                if (!out_ptr[i]) {
                    fprintf(stderr, "ERROR: Layer %d: out_ptr[%d] is NULL\n", i, i);
                    return 1;
                }
                if (!Weight_buf || !Beta_buf) {
                    fprintf(stderr, "ERROR: Layer %d: Weight_buf or Beta_buf is NULL\n", i);
                    return 1;
                }
                
                printf("    Calling YOLO2_FPGA: in_ptr=%p, out_ptr=%p, weight=%p, beta=%p\n",
                       (void*)in_ptr[i], (void*)out_ptr[i], 
                       (void*)(Weight_buf + woffset), (void*)(Beta_buf + boffset));
                
                // CRITICAL: Ensure all memory pages are accessible before hardware stub call
                // The hardware stub will copy the full depth, so we must ensure all pages are mapped
                if (i == 0) {
                    // For layer 0, ensure Input buffer is fully accessible
                    volatile IO_Dtype verify_input = 0;
                    for (size_t j = 0; j < input_depth_words; j += 1024) {
                        verify_input += in_ptr[i][j];
                        in_ptr[i][j] = in_ptr[i][j];  // Force page mapping
                    }
                    // Ensure Output buffer is fully accessible
                    volatile IO_Dtype verify_output = 0;
                    for (size_t j = 0; j < output_depth_words; j += 1024) {
                        verify_output += out_ptr[i][j];
                        out_ptr[i][j] = 0;  // Ensure zeros and page mapping
                    }
                    (void)verify_input; (void)verify_output;
                    printf("    Memory pages verified and mapped\n");
                }
                fflush(stdout);

                // Set Q values based on mode
                int Qw = 0, Qa_in = 0, Qa_out = 0, Qb = 0;
#ifdef INT16_MODE
                // For INT16 mode, use Q values from loaded tables
                if (offset_index < static_cast<int>(weight_q.size())) {
                    Qw = weight_q[offset_index];
                }
                if (offset_index < static_cast<int>(bias_q.size())) {
                    Qb = bias_q[offset_index];
                }
                // Activation Q: input is from previous layer's output, output is this layer's output
                if (offset_index < static_cast<int>(act_q.size())) {
                    Qa_in = act_q[offset_index];
                }
                if (offset_index + 1 < static_cast<int>(act_q.size())) {
                    Qa_out = act_q[offset_index + 1];
                } else if (offset_index < static_cast<int>(act_q.size())) {
                    Qa_out = act_q[offset_index];  // Use same Q if no next entry
                }
                
                // Special handling for route layer (layer 26) - use Q from layer 13
                if (i == 26 && act_q.size() > 13) {
                    Qa_in = act_q[13];
                }
                
                // Use pending route Q if set (for layers after route concatenation)
                if (pending_route_q >= 0) {
                    Qa_in = pending_route_q;
                    pending_route_q = -1;
                }
                
                // Update current Q for next layer
                current_Qa = Qa_out;
                
                printf("    Q values: Qw=%d, Qb=%d, Qa_in=%d, Qa_out=%d\n", Qw, Qb, Qa_in, Qa_out);
#endif

                YOLO2_FPGA(in_ptr[i], out_ptr[i], Weight_buf + woffset, Beta_buf + boffset,
                          l.c, l.n, l.size, l.stride, l.w, l.h, output_w, output_h, l.pad,
                          (l.activation == LEAKY) ? 1 : 0, l.batch_normalize ? 1 : 0,
                          TM, TN, TR, TC, (mLoops + 1) * TM, mLoops * TM, (mLoops + 1) * TM, 0,
                          Qw, Qa_in, Qa_out, Qb);
                
                printf("    Layer %d completed\n", i);
                fflush(stdout);

                woffset += cfg.weight_offsets[offset_index];
                boffset += cfg.beta_offsets[offset_index];
                offset_index++;
                break;
            }
            case MAXPOOL: {
                output_w = l.out_h;
                output_h = l.out_w;

                TR = std::min(((OnChipIB_Height - l.size) / l.stride + 1), Tr);
                TC = std::min(((OnChipIB_Width - l.size) / l.stride + 1), Tc);
                TR = std::min(output_h, TR);
                TC = std::min(output_w, TC);
                TM = std::min(Tm, Tn);
                TM = std::min(l.c, TM);
                mLoops = (int)ceil(((float)l.c) / TM);

                printf("  Layer %2d: POOL size=%d stride=%d -> %dx%d\n",
                       i, l.size, l.stride, output_w, output_h);

                YOLO2_FPGA(in_ptr[i], out_ptr[i], NULL, NULL, l.c, l.c,
                          l.size, l.stride, l.w, l.h, output_w, output_h, l.pad, 0, 0,
                          TM, 0, TR, TC, (mLoops + 2) * TM, mLoops * TM, (mLoops + 1) * TM, 1,
                          0, 0, 0, 0);
                break;
            }
            case REORG: {
                output_w = 26;
                output_h = 32 * 13;

                TR = std::min(((OnChipIB_Height - l.stride) / l.stride + 1), Tr);
                TR = std::min(output_h, TR);
                TC = std::min(((OnChipIB_Width - l.stride) / l.stride + 1), Tc);
                TC = std::min(output_w, TC);
                TM = std::min(Tm, Tn);
                TM = std::min(4, TM);
                mLoops = (int)ceil(((float)4) / TM);

                printf("  Layer %2d: REORG stride=%d\n", i, l.stride);

                tmp_ptr_f0 = in_ptr[i];
                for(int k = 0; k < 26*64; k++) {
                    memcpy((IO_Dtype *)(region_buf.data() + k*26), 
                           (IO_Dtype *)(tmp_ptr_f0 + k*32), 26*sizeof(IO_Dtype));
                }
                reorg_cpu(region_buf.data(), output_w, output_h, 4, 2, region_buf2.data());
                tmp_ptr_f0 = region_buf.data();
                memset(region_buf.data(), 0, 13*16*256*sizeof(IO_Dtype));
                for(int k = 0; k < 13*256; k++) {
                    memcpy((IO_Dtype *)(tmp_ptr_f0 + k*16), 
                           (IO_Dtype *)(region_buf2.data() + k*13), 13*sizeof(IO_Dtype));
                }
                
#ifdef INT16_MODE
                // Align quantization scales for route layer concatenation (layer 24)
                // This ensures the reorg branch matches the skip connection branch scale
                if (route24_q > 0 && current_Qa > 0) {
                    const int target_q = std::min(route24_q, current_Qa);
                    const int shift = current_Qa - target_q;
                    if (shift != 0) {
                        const int total = 13 * 16 * 256;
                        printf("    Aligning Q scales: current_Qa=%d, route24_q=%d, target=%d, shift=%d\n",
                               current_Qa, route24_q, target_q, shift);
                        for (int idx = 0; idx < total; ++idx) {
                            int32_t v = static_cast<int32_t>(tmp_ptr_f0[idx]);
                            if (shift > 0) {
                                v >>= shift;
                            } else {
                                v <<= -shift;
                            }
                            // Clamp to int16_t range
                            if (v > 32767) v = 32767;
                            if (v < -32768) v = -32768;
                            tmp_ptr_f0[idx] = static_cast<IO_Dtype>(v);
                        }
                        current_Qa = target_q;
                    }
                    pending_route_q = current_Qa;
                }
#endif
                
                memcpy(out_ptr[i], tmp_ptr_f0, 13*16*256*sizeof(IO_Dtype));
                break;
            }
            case ROUTE: {
                printf("  Layer %2d: ROUTE (no-op in HLS path)\n", i);
#ifdef INT16_MODE
                // Store Q value from layer 24 (route source) for later reorg alignment
                // Layer 24 routes from layer 13, so we need to track the Q value from that layer
                if (i == 24 && offset_index > 0 && offset_index - 1 < static_cast<int>(act_q.size())) {
                    // Layer 24 routes from layer 13, which corresponds to offset_index around 13
                    // Actually, we should use the current Qa_out from the previous conv layer
                    route24_q = current_Qa;
                    printf("    Stored route24_q=%d for reorg alignment\n", route24_q);
                }
#endif
                break;
            }
            case REGION: {
                printf("  Layer %2d: REGION (post-processing)\n", i);
                tmp_ptr_f0 = in_ptr[i];
                for(int k = 0; k < 13*425; k++) {
                    for(int j = 0; j < 16; j++) {
                        if(j < 13) {
                            region_buf[k*13 + j] = tmp_ptr_f0[k*16 + j];
                        }
                    }
                }
                // Convert to float for region layer processing
                std::vector<float> region_f(region_buf.size());
#ifdef INT16_MODE
                // For INT16 mode, dequantize using current_Qa (matches modern CPU version in yolo2_model.cpp)
                // current_Qa is updated after each conv layer and contains the Q value from the last conv layer output
                if (!act_q.empty()) {
                    const int q_out = current_Qa;
                    const float scale = std::ldexp(1.0f, -q_out);  // Dequantization: 2^(-Q), matches CPU version
                    printf("    Dequantizing region output with current_Qa=%d (scale=2^(-%d)=%.6f)\n", 
                           q_out, q_out, scale);
                    for (size_t t = 0; t < region_buf.size(); ++t) {
                        region_f[t] = static_cast<float>(region_buf[t]) * scale;
                    }
                } else {
                    fprintf(stderr, "ERROR: Activation Q table empty for INT16 dequantization\n");
                    // Direct cast as fallback (will be wrong)
                    for (size_t t = 0; t < region_buf.size(); ++t) {
                        region_f[t] = static_cast<float>(region_buf[t]);
                    }
                }
#else
                // For FP32 mode, direct cast
                for (size_t t = 0; t < region_buf.size(); ++t) {
                    region_f[t] = static_cast<float>(region_buf[t]);
                }
#endif
                forward_region_layer(l, region_f.data());
                break;
            }
            default:
                printf("  Layer %2d: UNKNOWN type %d (skipping)\n", i, l.type);
                break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    
    printf("\nInference completed in %.3f seconds\n", duration.count());

    // Run post-processing on the region output to print detections and save an annotated image.
    const float thresh = 0.24f;
    const float hier_thresh = 0.5f;
    const float nms = 0.45f;

    layer last = net->layers[net->n - 1];
    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
    if (dets) {
        if (nms > 0.0f) {
            do_nms_sort(dets, nboxes, last.classes, nms);
        }

        std::vector<std::string> labels = load_label_lines(join_path(proj_root, "config/coco.names"));
        std::vector<const char *> label_ptrs;
        for (const auto &s : labels) label_ptrs.push_back(s.c_str());

        printf("\nDetections (thresh=%.2f):\n", thresh);
        for (int i = 0; i < nboxes; ++i) {
            int best_class = -1;
            float best_prob = 0.f;
            for (int cls = 0; cls < last.classes; ++cls) {
                const float p = dets[i].prob[cls];
                if (p > best_prob) {
                    best_prob = p;
                    best_class = cls;
                }
            }
            if (best_prob > thresh && best_class >= 0) {
                const char *name = (best_class < (int)label_ptrs.size()) ? label_ptrs[best_class] : "cls";
                const box b = dets[i].bbox;
                printf("  %-16s prob=%.2f box=[x=%.1f y=%.1f w=%.1f h=%.1f]\n",
                       name, best_prob, b.x, b.y, b.w, b.h);
            }
        }

        // Optionally draw and save annotated image if labels are available.
        if (!label_ptrs.empty()) {
            // draw_detections expects image** alphabet
            // load_alphabet() looks for data/labels/ directory relative to current working directory
            // Ensure we're in the project root or set the path correctly
            image **alph = load_alphabet();
            if (alph) {
                draw_detections(im, dets, nboxes, thresh, const_cast<char **>(label_ptrs.data()), alph, last.classes);
                std::string img_out = join_path(output_dir, "cosim_output.png");
                save_image_png(im, img_out.c_str());
                printf("Annotated image written to %s\n", img_out.c_str());
                // No explicit free_alphabet helper available; allow OS cleanup.
            } else {
                printf("WARNING: Failed to load alphabet images (data/labels/ not found). Skipping image annotation.\n");
            }
        }

        free_detections(dets, nboxes);
    } else {
        printf("WARNING: get_network_boxes returned null; skipping detection printout\n");
    }

    // Save output
    printf("\nSaving output...\n");
    
    // Find the last layer output
    layer last_layer = net->layers[net->n - 1];
    IO_Dtype *final_output = nullptr;
    int final_output_size = 0;

    // For REGION layer, the output is in region_buf
    if (last_layer.type == REGION) {
        final_output = region_buf.data();
        final_output_size = 13 * 13 * 425;
    } else {
        // Find the last output pointer
        for (int i = net->n - 1; i >= 0; --i) {
            if (net->layers[i].type == CONVOLUTIONAL || net->layers[i].type == MAXPOOL) {
                final_output = out_ptr[i];
                final_output_size = net->layers[i].out_w * net->layers[i].out_h * net->layers[i].out_c;
                break;
            }
        }
    }

    // Create output directory if it doesn't exist
    struct stat info;
    if (stat(output_dir.c_str(), &info) != 0) {
        // Directory doesn't exist, try to create it
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", output_dir.c_str());
        system(cmd);
    }
    
    if (final_output) {
        std::string output_path = join_path(output_dir, "cosim_output.bin");
        write_binary<IO_Dtype>(output_path, final_output, final_output_size);
        printf("Output saved to: %s (%d elements)\n", output_path.c_str(), final_output_size);
    } else {
        printf("WARNING: Could not determine final output location\n");
    }

    // Save intermediate outputs for debugging (first few layers)
    printf("Saving intermediate layer outputs...\n");
    for (int i = 0; i < std::min(5, net->n); ++i) {
        if (net->layers[i].type == CONVOLUTIONAL || net->layers[i].type == MAXPOOL) {
            int out_size = net->layers[i].out_w * net->layers[i].out_h * net->layers[i].out_c;
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/layer_%02d_output.bin", output_dir.c_str(), i);
            write_binary<IO_Dtype>(fname, out_ptr[i], out_size);
        }
    }

    // Cleanup
    free(Memory_buf);
    free(Weight_buf);
    free(Beta_buf);
    free_image(im);
    free_image(sized);
    // Note: network cleanup would require free_network, but it's not always available
    // The OS will clean up on exit

    printf("\nCo-simulation testbench completed successfully!\n");
    return 0;
}
