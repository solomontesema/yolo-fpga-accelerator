/*
 * YOLOv2 Weight Reorganization Tool
 *
 * Reads YOLOv2 weights and writes them in the tiled order expected by the HLS
 * accelerator (TM x TN chunks, KxK major). Supports fp32 and int16 inputs.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

#include <core/yolo.h>
#include <core/yolo_cfg.hpp>
#include <models/yolov2/model_config.hpp>
#include <core/params.hpp>

namespace {

enum class Precision {
    FP32,
    INT16
};

const char *to_string(Precision p) {
    return p == Precision::FP32 ? "fp32" : "int16";
}

template <typename T>
void WeightReorg(const T *weight, T *weight_reorg, int IFM_NUM, int OFM_NUM, int Ksize) {
    const int KxK = Ksize * Ksize;
    const int IFM_NUMxKxK = IFM_NUM * KxK;

    std::vector<T> weight_buffer(Tm * Tn * K * K);
    std::vector<T> weight_buffer2(Tm * Tn * K * K);
    int offset = 0;

    for (int m = 0; m < OFM_NUM; m += Tm) {
        int TM_MIN = std::min(Tm, OFM_NUM - m);
        for (int n = 0; n < IFM_NUM; n += Tn) {
            int TN_MIN = std::min(Tn, IFM_NUM - n);
            int Woffset = m * IFM_NUMxKxK + n * KxK;

            for (int tm = 0; tm < TM_MIN; tm++) {
                std::memcpy(weight_buffer.data() + tm * TN_MIN * KxK,
                            weight + tm * IFM_NUMxKxK + Woffset,
                            TN_MIN * KxK * sizeof(T));
            }

            int TN_MINxTM_MIN = TN_MIN * TM_MIN;
            for (int tk = 0; tk < KxK; tk++)
                for (int tm = 0; tm < TM_MIN; tm++)
                    for (int tn = 0; tn < TN_MIN; tn++) {
                        weight_buffer2[tk * TN_MINxTM_MIN + tm * TN_MIN + tn] =
                            weight_buffer[tm * TN_MIN * KxK + tn * KxK + tk];
                    }

            std::memcpy(weight_reorg + offset, weight_buffer2.data(),
                        TM_MIN * TN_MIN * KxK * sizeof(T));
            offset += TM_MIN * TN_MIN * KxK;
        }
    }
}

struct GenConfig {
    std::string cfg_path = "config/yolov2.cfg";
    std::string weights_in;
    std::string weights_out;
    Precision precision = Precision::FP32;
};

Precision parse_precision(const std::string &arg) {
    if (arg == "fp32" || arg == "float" || arg == "f32") return Precision::FP32;
    if (arg == "int16" || arg == "i16" || arg == "fixed") return Precision::INT16;
    throw std::runtime_error("Unknown precision: " + arg);
}

GenConfig parse_args(int argc, char **argv) {
    GenConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if ((arg == "--cfg" || arg == "-c") && i + 1 < argc) {
            cfg.cfg_path = argv[++i];
        } else if ((arg == "--weights" || arg == "-w") && i + 1 < argc) {
            cfg.weights_in = argv[++i];
        } else if ((arg == "--out" || arg == "-o") && i + 1 < argc) {
            cfg.weights_out = argv[++i];
        } else if ((arg == "--precision" || arg == "-p") && i + 1 < argc) {
            cfg.precision = parse_precision(argv[++i]);
        } else if (arg == "--int16") {
            cfg.precision = Precision::INT16;
        } else if (arg == "--fp32") {
            cfg.precision = Precision::FP32;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--cfg <cfg>] [--weights <weights.bin>] [--out <weights_reorg.bin>] [--precision fp32|int16]\n", argv[0]);
            std::exit(0);
        }
    }
    return cfg;
}

template <typename T>
std::vector<T> read_weights(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Couldn't open file: " + path);
    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz % sizeof(T) != 0) {
        std::fclose(fp);
        throw std::runtime_error("Invalid weight file size: " + path);
    }
    std::vector<T> buf(sz / sizeof(T));
    size_t rd = std::fread(buf.data(), sizeof(T), buf.size(), fp);
    std::fclose(fp);
    if (rd != buf.size()) throw std::runtime_error("Failed to read weights: " + path);
    return buf;
}

template <typename T>
void write_weights(const std::string &path, const std::vector<T> &buf) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    FILE *fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Couldn't open file for write: " + path);
    size_t wr = std::fwrite(buf.data(), sizeof(T), buf.size(), fp);
    std::fclose(fp);
    if (wr != buf.size()) throw std::runtime_error("Failed to write weights: " + path);
}

} // namespace

int main(int argc, char **argv) {
    try {
        GenConfig cfg = parse_args(argc, argv);
        const std::string default_in_fp32 = "weights/weights.bin";
        const std::string default_out_fp32 = "weights/weights_reorg.bin";
        const std::string default_in_int16 = "weights/weight_int16.bin";
        const std::string default_out_int16 = "weights/weights_reorg_int16.bin";

        if (cfg.weights_in.empty()) {
            cfg.weights_in = (cfg.precision == Precision::FP32) ? default_in_fp32 : default_in_int16;
        }
        if (cfg.weights_out.empty()) {
            cfg.weights_out = (cfg.precision == Precision::FP32) ? default_out_fp32 : default_out_int16;
        }

        std::filesystem::path in_path(cfg.weights_in);
        std::filesystem::path out_path(cfg.weights_out);

        // Prevent accidental in-place overwrite if a user points --weights to the output path.
        if (in_path.lexically_normal() == out_path.lexically_normal()) {
            auto default_in = std::filesystem::path(cfg.precision == Precision::FP32 ? default_in_fp32 : default_in_int16);
            if (std::filesystem::exists(default_in)) {
                std::fprintf(stderr,
                             "Warning: input and output paths are the same (%s); falling back to %s\n",
                             cfg.weights_in.c_str(), default_in.string().c_str());
                cfg.weights_in = default_in.string();
                in_path = default_in;
            } else {
                throw std::runtime_error("Input weights path matches output; point --weights to a valid input file");
            }
        }

        // If a custom path is missing but the default exists, automatically fall back.
        if (!std::filesystem::exists(in_path)) {
            auto default_in = std::filesystem::path(cfg.precision == Precision::FP32 ? default_in_fp32 : default_in_int16);
            if (in_path.lexically_normal() != default_in.lexically_normal() &&
                std::filesystem::exists(default_in)) {
                std::fprintf(stderr,
                             "Warning: %s not found; using %s instead\n",
                             cfg.weights_in.c_str(), default_in.string().c_str());
                cfg.weights_in = default_in.string();
                in_path = default_in;
            }
        }

        std::printf("Precision      : %s\n", to_string(cfg.precision));
        std::printf("Input weights : %s\n", cfg.weights_in.c_str());
        std::printf("Output weights: %s\n", cfg.weights_out.c_str());

        network *net = load_network(const_cast<char *>(cfg.cfg_path.c_str()));
        if (!net) throw std::runtime_error("Failed to load cfg: " + cfg.cfg_path);

        const ModelConfig &mc = yolo2_model_config();

        if (cfg.precision == Precision::FP32) {
            auto weights = read_weights<float>(cfg.weights_in);
            std::vector<float> weights_reorg(weights.size(), 0.0f);
            size_t woffset = 0;
            int offset_index = 0;

            for (int i = 0; i < net->n; ++i) {
                layer l = net->layers[i];
                if (l.type == CONVOLUTIONAL) {
                    if (offset_index >= static_cast<int>(mc.weight_offsets.size()))
                        throw std::runtime_error("Weight offset table too small");
                    if (woffset + mc.weight_offsets[offset_index] > weights.size())
                        throw std::runtime_error("Weight file too small for layer " + std::to_string(i));

                    WeightReorg(weights.data() + woffset,
                                weights_reorg.data() + woffset,
                                l.c, l.n, l.size);

                    woffset += mc.weight_offsets[offset_index];
                    offset_index++;
                }
            }

            if (woffset != weights.size()) {
                // Allow some trailing zeros if offsets table shorter?
                if (woffset < weights.size()) {
                    std::fill(weights_reorg.begin() + woffset, weights_reorg.end(), 0.0f);
                } else {
                    throw std::runtime_error("Processed weights exceed input size");
                }
            }

            write_weights(cfg.weights_out, weights_reorg);
        } else {
            auto weights = read_weights<int16_t>(cfg.weights_in);
            size_t expected = 0;
            for (int off : mc.weight_offsets) expected += off;

            // Some legacy int16 blobs are already quantized and reorganized.
            if (weights.size() < expected) {
                std::fprintf(stderr,
                             "Warning: int16 weight file smaller than expected (%zu < %zu); assuming it is already reorganized. Copying through.\n",
                             weights.size(), expected);
                write_weights(cfg.weights_out, weights);
            } else {
                std::vector<int16_t> weights_reorg(weights.size(), 0);
                size_t woffset = 0;
                int offset_index = 0;

                for (int i = 0; i < net->n; ++i) {
                    layer l = net->layers[i];
                    if (l.type == CONVOLUTIONAL) {
                        if (offset_index >= static_cast<int>(mc.weight_offsets.size()))
                            throw std::runtime_error("Weight offset table too small");
                        if (woffset + mc.weight_offsets[offset_index] > weights.size())
                            throw std::runtime_error("Weight file too small for layer " + std::to_string(i));

                        WeightReorg(weights.data() + woffset,
                                    weights_reorg.data() + woffset,
                                    l.c, l.n, l.size);

                        woffset += mc.weight_offsets[offset_index];
                        offset_index++;
                    }
                }

                if (woffset != weights.size()) {
                    if (woffset < weights.size()) {
                        std::fill(weights_reorg.begin() + woffset, weights_reorg.end(), 0);
                    } else {
                        throw std::runtime_error("Processed weights exceed input size");
                    }
                }

                write_weights(cfg.weights_out, weights_reorg);
            }
        }

        std::printf("Reorganized weights written to %s\n", cfg.weights_out.c_str());

    } catch (const std::exception &ex) {
        std::fprintf(stderr, "Fatal error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
