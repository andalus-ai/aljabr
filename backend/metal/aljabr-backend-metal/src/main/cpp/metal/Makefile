# ==============================================================================
# Aljabr Metal Bridge — Makefile
# ==============================================================================
#
# Compiles the Metal Objective-C sources into a shared dylib for Java FFM loading.
#
# Requirements:
#   - macOS 13 (Ventura) or later
#   - Apple Silicon (arm64) or Intel Mac with Metal GPU
#   - Xcode Command Line Tools
#
# Usage:
#   make                 — build libaljabr_metal.dylib
#   make install         — copy to ~/.aljabr/libs/ and resources/native/darwin-aarch64/
#   make clean           — remove build artifacts
#   make info            — print detected SDK and device info
#
# Output:
#   target/native/darwin-aarch64/libaljabr_metal.dylib
# Install:
#   ~/.aljabr/libs/libaljabr_metal.dylib
# ==============================================================================

SDK        := $(shell xcrun --sdk macosx --show-sdk-path 2>/dev/null)
CC         := clang
TARGET_DIR := ../../../../target/native/darwin-aarch64
OBJ_DIR    := $(TARGET_DIR)/obj
OUTPUT     := libaljabr_metal.dylib
SRC        := aljabr_metal_support.m aljabr_metal_buffers.m aljabr_metal_scratch.m aljabr_metal_cpu_fallback.m aljabr_metal_elementwise.m aljabr_metal_gated_ffn.m aljabr_metal_attention.m aljabr_metal_runtime.m aljabr_metal_kernel_source.m aljabr_metal_pipelines.m aljabr_metal_mps_cache.m aljabr_metal_mps_matmul.m aljabr_metal_mps_validation.m aljabr_metal_mps_matvec_policy.m aljabr_metal_matvec_dispatch.m aljabr_metal_matvec_tuning.m aljabr_metal_matvec_api.m aljabr_metal_bridge.m aljabr_metal_fa4.m
OBJ        := $(addprefix $(OBJ_DIR)/,$(SRC:.m=.o))
INSTALL_DIR ?= $(HOME)/.aljabr/libs

# Detect architecture: default arm64 (Apple Silicon), allow override for x86_64
ARCH ?= arm64

CFLAGS := -arch $(ARCH) \
           -fPIC -fobjc-arc \
           -O3 -isysroot $(SDK) \
           -mmacosx-version-min=15.0

LDFLAGS := -arch $(ARCH) \
           -shared -isysroot $(SDK) \
           -mmacosx-version-min=15.0

FRAMEWORKS := -framework Metal \
               -framework MetalPerformanceShaders \
               -framework MetalPerformanceShadersGraph \
               -framework Foundation \
               -framework Accelerate

.PHONY: all install clean info

all: $(TARGET_DIR)/$(OUTPUT)

$(TARGET_DIR)/$(OUTPUT): $(OBJ) | $(TARGET_DIR)
	$(CC) $(LDFLAGS) $(FRAMEWORKS) -o $@ $(OBJ)
	@echo "✓ Built $(OUTPUT) → $(TARGET_DIR)/"
	@codesign --sign - $@ 2>/dev/null || true  # ad-hoc sign for local use

$(TARGET_DIR) $(OBJ_DIR):
	@mkdir -p $@

$(OBJ_DIR)/%.o: %.m aljabr_metal_support.h aljabr_metal_buffers.h aljabr_metal_scratch.h aljabr_metal_cpu_fallback.h aljabr_metal_elementwise.h aljabr_metal_gated_ffn.h aljabr_metal_attention.h aljabr_metal_runtime.h aljabr_metal_kernel_source.h aljabr_metal_pipelines.h aljabr_metal_mps_cache.h aljabr_metal_mps_matmul.h aljabr_metal_mps_validation.h aljabr_metal_mps_matvec_policy.h aljabr_metal_matvec_dispatch.h aljabr_metal_matvec_tuning.h aljabr_metal_matvec_api.h | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

install: all
	@mkdir -p $(INSTALL_DIR)
	cp $(TARGET_DIR)/$(OUTPUT) $(INSTALL_DIR)/
	@mkdir -p ../resources/native/darwin-aarch64
	cp $(TARGET_DIR)/$(OUTPUT) ../resources/native/darwin-aarch64/
	@echo "✓ Installed to $(INSTALL_DIR)/ and resources/native/darwin-aarch64/"

clean:
	rm -f $(TARGET_DIR)/$(OUTPUT) $(SRC:.m=.o)
	rm -rf $(OBJ_DIR)
	@echo "✓ Cleaned"

info:
	@echo "SDK:    $(SDK)"
	@echo "ARCH:   $(ARCH)"
	@echo "OUTPUT: $(TARGET_DIR)/$(OUTPUT)"
	@system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|VRAM" | head -4 || true
