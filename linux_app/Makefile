# YOLOv2 FPGA Accelerator - Linux Application Makefile
#
# Build for ARM64 (aarch64) on KV260
# Native compilation on the board

# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g
CFLAGS += -I./include -I./include/third_party

# Architecture-specific flags
ARCH_FLAGS = -march=armv8-a

# Libraries
LDFLAGS = -lm -lrt

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
STB_DIR = ./include/third_party

# Source files
SRCS = $(SRC_DIR)/main.c \
       $(SRC_DIR)/yolo2_accel_linux.c \
       $(SRC_DIR)/dma_buffer_manager.c \
       $(SRC_DIR)/yolo2_inference.c \
       $(SRC_DIR)/yolo2_network.c \
       $(SRC_DIR)/yolo2_postprocess.c \
       $(SRC_DIR)/yolo2_image_loader.c \
       $(SRC_DIR)/yolo2_draw.c \
       $(SRC_DIR)/yolo2_v4l2.c \
       $(SRC_DIR)/yolo2_ffmpeg_video.c \
       $(SRC_DIR)/yolo2_log.c \
       $(SRC_DIR)/yolo2_labels.c \
       $(SRC_DIR)/file_loader.c \
       $(SRC_DIR)/stb_image_impl.c \
       $(SRC_DIR)/stb_image_write_impl.c

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# Target executable
TARGET = yolo2_linux

# Test programs
TEST_ACCEL = test_accel
TEST_DMA = test_dma
TEST_PL_DDR = test_pl_ddr
CHECK_HP = check_hp_clocks

# Default target
all: $(TARGET)

# Main executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -c -o $@ $<

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Test program for accelerator register access
$(TEST_ACCEL): $(BUILD_DIR)/test_accel.o $(BUILD_DIR)/yolo2_accel_linux.o
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_accel.o: tests/test_accel.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -c -o $@ $<

# Test program for DMA buffer allocation
$(TEST_DMA): $(BUILD_DIR)/test_dma.o $(BUILD_DIR)/dma_buffer_manager.o
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_dma.o: tests/test_dma.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -c -o $@ $<

# Test program for PL-DDR connectivity
$(TEST_PL_DDR): $(BUILD_DIR)/test_pl_ddr.o
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/test_pl_ddr.o: tests/test_pl_ddr.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -c -o $@ $<

# Check HP port clocks
$(CHECK_HP): $(BUILD_DIR)/check_hp_clocks.o
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/check_hp_clocks.o: tests/check_hp_clocks.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -c -o $@ $<

# Create test directory
tests:
	mkdir -p tests

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(TEST_ACCEL) $(TEST_DMA) $(TEST_PL_DDR) $(CHECK_HP)

# Install (copy to /usr/local/bin)
install: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/
	@echo "Installed $(TARGET) to /usr/local/bin/"

# Uninstall
uninstall:
	sudo rm -f /usr/local/bin/$(TARGET)

# Run (with sudo)
run: $(TARGET)
	sudo ./$(TARGET)

# Debug build
debug: CFLAGS += -DDEBUG -g3 -O0
debug: clean all

# Release build
release: CFLAGS += -DNDEBUG -O3
release: clean all

# Dependencies
$(BUILD_DIR)/main.o: $(INC_DIR)/yolo2_config.h \
                     $(INC_DIR)/yolo2_accel_linux.h \
                     $(INC_DIR)/dma_buffer_manager.h \
                     $(INC_DIR)/yolo2_inference.h \
                     $(INC_DIR)/yolo2_network.h \
                     $(INC_DIR)/yolo2_image_loader.h \
                     $(INC_DIR)/yolo2_postprocess.h \
                     $(INC_DIR)/yolo2_labels.h \
                     $(INC_DIR)/file_loader.h

$(BUILD_DIR)/yolo2_accel_linux.o: $(INC_DIR)/yolo2_accel_linux.h \
                                  $(INC_DIR)/yolo2_config.h

$(BUILD_DIR)/dma_buffer_manager.o: $(INC_DIR)/dma_buffer_manager.h \
                                   $(INC_DIR)/yolo2_config.h

$(BUILD_DIR)/yolo2_inference.o: $(INC_DIR)/yolo2_inference.h \
                                $(INC_DIR)/yolo2_accel_linux.h \
                                $(INC_DIR)/yolo2_config.h \
                                $(INC_DIR)/yolo2_network.h \
                                $(INC_DIR)/dma_buffer_manager.h

$(BUILD_DIR)/yolo2_network.o: $(INC_DIR)/yolo2_network.h \
                              $(INC_DIR)/yolo2_config.h

$(BUILD_DIR)/yolo2_postprocess.o: $(INC_DIR)/yolo2_postprocess.h \
                                  $(INC_DIR)/yolo2_config.h

$(BUILD_DIR)/yolo2_image_loader.o: $(INC_DIR)/yolo2_image_loader.h \
                                   $(INC_DIR)/yolo2_config.h \
                                   $(STB_DIR)/stb_image.h

$(BUILD_DIR)/yolo2_labels.o: $(INC_DIR)/yolo2_labels.h

$(BUILD_DIR)/file_loader.o: $(INC_DIR)/file_loader.h

$(BUILD_DIR)/stb_image_impl.o: $(STB_DIR)/stb_image.h

.PHONY: all clean install uninstall run debug release tests
