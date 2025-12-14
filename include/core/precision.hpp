#pragma once

#include <string>
#include <stdexcept>

enum class Precision {
    FP32,
    INT16
};

inline const char *to_string(Precision p) {
    switch (p) {
        case Precision::FP32: return "fp32";
        case Precision::INT16: return "int16";
    }
    return "unknown";
}

inline Precision parse_precision(const std::string &v, Precision fallback = Precision::FP32) {
    if (v.empty()) return fallback;
    if (v == "fp32" || v == "float" || v == "f32") return Precision::FP32;
    if (v == "int16" || v == "i16" || v == "fixed") return Precision::INT16;
    throw std::runtime_error("Unsupported precision: " + v);
}
