# YOLOv2 Float32 Object Detection Makefile
# 
# This Makefile builds the YOLOv2 object detection application.
# 
# Available targets:
#   make all      - Build all components (default)
#   make gen      - Generate weight reorganization files
#   make test     - Build the detection application
#   make clean    - Remove built files
#   make help     - Display this help message

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra
LDFLAGS := -lm
DEBUG_FLAGS := -g -O0 -DDEBUG

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
SCRIPT_DIR := scripts
CONFIG_DIR := config
WEIGHTS_DIR := weights

# Include paths
INCLUDES := -I$(INC_DIR) -I$(INC_DIR)/core -I$(INC_DIR)/models/yolov2 -Ihls -Ihls/core -Ihls/models/yolov2

# Source files
MAIN_SRC := $(SRC_DIR)/models/yolov2/yolov2_main.cpp
WEIGHT_GEN_SRC := $(SRC_DIR)/models/yolov2/yolov2_weight_gen.cpp
CORE_SRCS := $(SRC_DIR)/core/yolo_image.cpp $(SRC_DIR)/core/yolo_post.cpp $(SRC_DIR)/core/yolo_utils.cpp $(SRC_DIR)/core/yolo_cfg.cpp $(SRC_DIR)/core/yolo_math.cpp $(SRC_DIR)/core/yolo_region.cpp $(SRC_DIR)/core/yolo_layers.cpp $(SRC_DIR)/core/yolo_net.cpp
HLS_SRCS := hls/core/core_io.cpp hls/core/core_compute.cpp hls/core/core_scheduler.cpp hls/models/yolov2/yolo2_accel.cpp hls/models/yolov2/yolo2_model.cpp hls/models/yolov2/model_config.cpp
EXTRA_SRCS := $(SRC_DIR)/stb_image_implementation.cpp

# Executable names
TARGET := yolov2_detect
GEN_TARGET := yolov2_weight_gen

# Python script
HW_PARAMS_SCRIPT := $(SCRIPT_DIR)/hw_params_gen.py

# Color output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_BLUE := \033[34m

# Default target
.PHONY: all
all: test

# Help target
.PHONY: help
help:
	@echo "$(COLOR_BOLD)YOLOv2 Float32 Detection - Available Targets:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)make all$(COLOR_RESET)      - Build all components (default)"
	@echo "  $(COLOR_GREEN)make gen$(COLOR_RESET)      - Generate weight reorganization files"
	@echo "  $(COLOR_GREEN)make test$(COLOR_RESET)     - Build the detection application"
	@echo "  $(COLOR_GREEN)make debug$(COLOR_RESET)    - Build with debug symbols"
	@echo "  $(COLOR_GREEN)make clean$(COLOR_RESET)    - Remove built files"
	@echo "  $(COLOR_GREEN)make help$(COLOR_RESET)     - Display this help message"
	@echo ""
	@echo "$(COLOR_BOLD)Usage:$(COLOR_RESET)"
	@echo "  ./$(TARGET) [image_path]"
	@echo ""
	@echo "$(COLOR_BOLD)Note:$(COLOR_RESET) Ensure weights.bin and bias.bin are in $(WEIGHTS_DIR)/ directory"

# Generate hardware parameters and build weight generation executable
.PHONY: gen
gen: $(BUILD_DIR)
	@echo "$(COLOR_BLUE)Generating hardware parameters...$(COLOR_RESET)"
	@cd . && python3 $(HW_PARAMS_SCRIPT)
	@echo "$(COLOR_BLUE)Building weight generation executable...$(COLOR_RESET)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(GEN_TARGET) $(WEIGHT_GEN_SRC) $(CORE_SRCS) hls/models/yolov2/model_config.cpp $(EXTRA_SRCS) $(LDFLAGS)
	@echo "$(COLOR_GREEN)Weight generation build complete. Run ./$(GEN_TARGET) to generate weights_reorg.bin$(COLOR_RESET)"

# Build the main detection application
.PHONY: test
test: $(BUILD_DIR)
	@echo "$(COLOR_BLUE)Generating hardware parameters...$(COLOR_RESET)"
	@cd . && python3 $(HW_PARAMS_SCRIPT)
	@echo "$(COLOR_BLUE)Building detection executable...$(COLOR_RESET)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(MAIN_SRC) $(CORE_SRCS) $(HLS_SRCS) $(EXTRA_SRCS) -D REORG_TEST $(LDFLAGS)
	@echo "$(COLOR_GREEN)Detection build complete. Run ./$(TARGET) [image_path]$(COLOR_RESET)"

# Build with debug symbols
.PHONY: debug
debug: CXXFLAGS := -std=c++11 $(DEBUG_FLAGS) -Wall -Wextra
debug: test

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "$(COLOR_BLUE)Cleaning build artifacts...$(COLOR_RESET)"
	@rm -f $(TARGET) $(GEN_TARGET)
	@rm -f *.png
	@rm -f *.o
	@echo "$(COLOR_GREEN)Clean complete$(COLOR_RESET)"

# Deep clean (including generated files)
.PHONY: distclean
distclean: clean
	@echo "$(COLOR_BLUE)Deep cleaning all generated files...$(COLOR_RESET)"
	@rm -f $(WEIGHTS_DIR)/weights_reorg.bin
	@rm -f yolo_last_layer_output.txt
	@echo "$(COLOR_GREEN)Deep clean complete$(COLOR_RESET)"

.PHONY: check-weights
check-weights:
	@if [ ! -f "$(WEIGHTS_DIR)/weights.bin" ] || [ ! -f "$(WEIGHTS_DIR)/bias.bin" ]; then \
		echo "$(COLOR_BOLD)Warning: Weight files not found in $(WEIGHTS_DIR)/$(COLOR_RESET)"; \
		echo "Please refer to $(WEIGHTS_DIR)/README.md for instructions"; \
	else \
		echo "$(COLOR_GREEN)Weight files found$(COLOR_RESET)"; \
	fi
