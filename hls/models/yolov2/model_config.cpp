#include "model_config.hpp"

namespace {
constexpr std::array<int, 32> kYolo2WeightOffsets = {864, 18432, 73728, 8192, 73728,
                                                     294912, 32768, 294912, 1179648, 131072, 1179648, 131072,
                                                     1179648, 4718592, 524288, 4718592, 524288, 4718592, 9437184,
                                                     9437184, 32768, 11796480, 435200, 0, 0, 0, 0, 0, 0, 0, 0, 0};

constexpr std::array<int, 32> kYolo2BetaOffsets = {32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 1024,
                                                   512, 1024, 512, 1024, 1024, 1024, 64, 1024, 425, 0, 0, 0, 0, 0, 0, 0, 0, 0};
} // namespace

const ModelConfig &yolo2_model_config()
{
    static const ModelConfig cfg{
        /*mem_len=*/(416*416*32 + 208*208*32),
        /*route16_len=*/(26*32*512),
        /*conv27_len=*/(13*16*256),
        /*conv24_len=*/(13*16*1024),
        /*detection_workspace=*/(3*13*425),
        kYolo2WeightOffsets,
        kYolo2BetaOffsets
    };
    return cfg;
}
