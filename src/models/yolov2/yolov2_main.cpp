/*
 * YOLOv2 Object Detection - CLI entry point
 *
 * Single-translation-unit driver for the float32 CPU/HLS pipeline.
 * Responsibilities:
 *  - Parse CLI arguments (cfg, names, input image, output prefix, thresholds)
 *  - Load network/alphabet
 *  - Preprocess image to network size (letterbox)
 *  - Run inference via selected backend (default: HLS path)
 *  - Postprocess detections, draw labels, and save outputs
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <filesystem>

#include <core/yolo.h>
#include <core/precision.hpp>
#include <api.hpp>

namespace {

struct AppConfig {
    std::string cfg_path = "config/yolov2.cfg";
    std::string names_path = "config/coco.names";
    std::string input_path = "examples/test_images/dog.jpg";
    std::string output_prefix;
    float thresh = 0.25f;
    float nms = 0.45f;
    float hier_thresh = 0.5f;
    enum class Backend { Hls, Cpu } backend = Backend::Hls;
    Precision precision = Precision::FP32;
};

void print_usage(const char *prog) {
    std::printf(
        "Usage: %s [options]\n"
        "Options:\n"
        "  --cfg <path>          Network cfg file (default: config/yolov2.cfg)\n"
        "  --names <path>        Class names file (default: config/coco.names)\n"
        "  --input <path>        Input image (default: examples/test_images/dog.jpg)\n"
        "  --output <prefix>     Output file prefix without extension (default: <input>_prediction)\n"
        "  --thresh <float>      Confidence threshold (default: 0.5)\n"
        "  --nms <float>         NMS IoU threshold (default: 0.45)\n"
        "  --hier <float>        Hierarchical threshold (default: 0.5)\n"
        "  --backend <hls|cpu>   Backend selector (default: hls; cpu stub)\n"
        "  --precision <fp32|int16> Precision selector (default: fp32; int16 wiring in progress)\n"
        "  --help                Show this help message\n",
        prog);
}

bool starts_with(const std::string &s, const std::string &prefix) {
    return s.size() >= prefix.size() && std::memcmp(s.data(), prefix.data(), prefix.size()) == 0;
}

AppConfig parse_args(int argc, char **argv) {
    AppConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--cfg" && i + 1 < argc) {
            cfg.cfg_path = argv[++i];
        } else if (arg == "--names" && i + 1 < argc) {
            cfg.names_path = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            cfg.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_prefix = argv[++i];
        } else if (arg == "--thresh" && i + 1 < argc) {
            cfg.thresh = std::strtof(argv[++i], nullptr);
        } else if (arg == "--nms" && i + 1 < argc) {
            cfg.nms = std::strtof(argv[++i], nullptr);
        } else if (arg == "--hier" && i + 1 < argc) {
            cfg.hier_thresh = std::strtof(argv[++i], nullptr);
        } else if (arg == "--backend" && i + 1 < argc) {
            std::string backend_val = argv[++i];
            if (backend_val == "hls") {
                cfg.backend = AppConfig::Backend::Hls;
            } else if (backend_val == "cpu") {
                cfg.backend = AppConfig::Backend::Cpu;
            } else {
                std::fprintf(stderr, "Unsupported backend '%s'. Use 'hls' (available) or 'cpu' (stub).\n",
                             backend_val.c_str());
                std::exit(1);
            }
        } else if (arg == "--precision" && i + 1 < argc) {
            try {
                cfg.precision = parse_precision(argv[++i]);
            } catch (const std::exception &e) {
                std::fprintf(stderr, "%s\n", e.what());
                std::exit(1);
            }
        } else if (starts_with(arg, "--")) {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            std::exit(1);
        } else {
            // Positional argument: treat as input image if not already provided
            cfg.input_path = arg;
        }
    }
    return cfg;
}

std::string default_output_prefix(const std::string &input_path) {
    auto last_slash = input_path.find_last_of("/\\");
    const auto base_start = (last_slash == std::string::npos) ? 0 : last_slash + 1;
    std::string base = input_path.substr(base_start);
    auto last_dot = base.find_last_of('.');
    if (last_dot != std::string::npos) {
        base = base.substr(0, last_dot);
    }
    return base + "_prediction";
}

// Helpers to ensure resources are freed even on early exit.
void free_alphabet(image **alphabet) {
    if (!alphabet) return;
    const int nsize = 8;
    for (int j = 0; j < nsize; ++j) {
        if (!alphabet[j]) continue;
        for (int i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);
}

void free_network_deep(network *net) {
    if (!net) return;
    for (int i = 0; i < net->n; ++i) {
        free_layer(net->layers[i]);
    }
    free(net->layers);
    free(net->seen);
    free(net->t);
    free(net->cost);
    free(net);
}

struct AlphabetGuard {
    image **ptr = nullptr;
    ~AlphabetGuard() { free_alphabet(ptr); }
};

struct NetworkGuard {
    network *ptr = nullptr;
    ~NetworkGuard() { free_network_deep(ptr); }
};

struct ImageGuard {
    image img{};
    bool owns = false;
    explicit ImageGuard(image i = {}, bool own = false) : img(i), owns(own) {}
    ~ImageGuard() {
        if (owns) {
            free_image(img);
        }
    }
    ImageGuard(const ImageGuard &) = delete;
    ImageGuard &operator=(const ImageGuard &) = delete;
    ImageGuard(ImageGuard &&other) noexcept {
        img = other.img;
        owns = other.owns;
        other.owns = false;
    }
    ImageGuard &operator=(ImageGuard &&other) noexcept {
        if (this != &other) {
            if (owns) {
                free_image(img);
            }
            img = other.img;
            owns = other.owns;
            other.owns = false;
        }
        return *this;
    }
};

std::vector<std::string> load_label_lines(const std::string &path) {
    std::vector<std::string> labels;
    std::ifstream in(path.c_str());
    if (!in.is_open()) {
        std::fprintf(stderr, "Could not open names file: %s\n", path.c_str());
        std::exit(1);
    }
    std::string line;
    while (std::getline(in, line)) {
        // Strip trailing carriage returns/newlines.
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (!line.empty()) {
            labels.emplace_back(line);
        }
    }
    if (labels.empty()) {
        std::fprintf(stderr, "Names file %s is empty\n", path.c_str());
        std::exit(1);
    }
    return labels;
}

void run_detector(AppConfig cfg) {
    std::setbuf(stdout, nullptr);
    if (cfg.output_prefix.empty()) {
        cfg.output_prefix = default_output_prefix(cfg.input_path);
    }
    {
        namespace fs = std::filesystem;
        fs::path prefix(cfg.output_prefix);
        if (!prefix.has_parent_path()) {
            fs::create_directories("results");
            prefix = fs::path("results") / prefix;
        } else {
            fs::create_directories(prefix.parent_path());
        }
        cfg.output_prefix = prefix.string();
    }
    std::printf("YOLOv2 Object Detection - Starting\n");
    std::printf("  cfg:    %s\n", cfg.cfg_path.c_str());
    std::printf("  names:  %s\n", cfg.names_path.c_str());
    std::printf("  input:  %s\n", cfg.input_path.c_str());
    std::printf("  precision: %s\n", to_string(cfg.precision));
    std::printf("  output: %s[.png]\n", cfg.output_prefix.c_str());

    if (cfg.precision == Precision::INT16) {
        std::fprintf(stderr, "Int16 inference wiring is in progress; please run with --precision fp32 for now.\n");
        std::exit(1);
    }

    NetworkGuard net_guard;
    net_guard.ptr = load_network(const_cast<char *>(cfg.cfg_path.c_str()));
    if (!net_guard.ptr) {
        throw std::runtime_error("Failed to load network");
    }
    set_batch_network(net_guard.ptr, 1);

    const std::vector<std::string> label_strings = load_label_lines(cfg.names_path);
    std::vector<const char *> label_ptrs;
    label_ptrs.reserve(label_strings.size());
    for (const auto &s : label_strings) {
        label_ptrs.push_back(s.c_str());
    }

    AlphabetGuard alphabet_guard;
    alphabet_guard.ptr = load_alphabet();

    ImageGuard input_img(load_image_stb(const_cast<char *>(cfg.input_path.c_str()), 3), true);
    std::printf("Input img: %s (w=%d, h=%d, c=%d)\n",
                cfg.input_path.c_str(), input_img.img.w, input_img.img.h, input_img.img.c);

    const int net_w = net_guard.ptr->w;
    const int net_h = net_guard.ptr->h;
    ImageGuard sized(letterbox_image(input_img.img, net_w, net_h), true);

    const auto start = std::chrono::high_resolution_clock::now();
    switch (cfg.backend) {
        case AppConfig::Backend::Hls:
            yolov2_hls_ps(net_guard.ptr, sized.img.data);
            break;
        case AppConfig::Backend::Cpu:
            // CPU path placeholder: the current codebase does not load float weights
            // into network layers. This hook will be wired once weights loading is
            // refactored out of the HLS path.
            throw std::runtime_error("CPU backend is not wired yet (weights not loaded in this build).");
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::printf("%s: Predicted in %.3f seconds.\n", cfg.input_path.c_str(), elapsed);

    int nboxes = 0;
    layer last = net_guard.ptr->layers[net_guard.ptr->n - 1];
    detection *dets = get_network_boxes(net_guard.ptr, input_img.img.w, input_img.img.h,
                                        cfg.thresh, cfg.hier_thresh, 0, 1, &nboxes);
    if (!dets) {
        throw std::runtime_error("get_network_boxes returned null");
    }

    if (cfg.nms > 0.0f) {
        do_nms_sort(dets, nboxes, last.classes, cfg.nms);
    }

    const int available_labels = static_cast<int>(label_ptrs.size());
    if (available_labels < last.classes) {
        std::fprintf(stderr,
                     "Warning: names file provides %d labels, but network expects %d classes.\n",
                     available_labels, last.classes);
    }

    draw_detections(input_img.img, dets, nboxes, cfg.thresh,
                    const_cast<char **>(label_ptrs.data()), alphabet_guard.ptr, last.classes);

    free_detections(dets, nboxes);

    save_image_png(input_img.img, cfg.output_prefix.c_str());
    std::printf("Output written to %s.png\n", cfg.output_prefix.c_str());
    std::printf("YOLOv2 Object Detection - Complete\n");
}

} // namespace

int main(int argc, char **argv) {
    try {
        AppConfig cfg = parse_args(argc, argv);
        run_detector(cfg);
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "Fatal error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
