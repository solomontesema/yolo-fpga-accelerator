#pragma once

#include <array>

struct ModelConfig {
    int mem_len;
    int route16_len;
    int conv27_len;
    int conv24_len;
    int detection_workspace;
    std::array<int, 32> weight_offsets;
    std::array<int, 32> beta_offsets;
};

// Descriptor for the YOLOv2 float32/HLS layout.
const ModelConfig &yolo2_model_config();
