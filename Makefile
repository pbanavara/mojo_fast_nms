# Makefile for CUDA NMS implementation

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_60

# Target executable
TARGET = nms_test

# Source files
CUDA_SRC = warp_bitmask_nms.cu

# Default target
all: $(TARGET)

# Compile CUDA source
$(TARGET): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Run the test
run: $(TARGET)
	./$(TARGET)

# Check if CUDA is available
check-cuda:
	@which nvcc > /dev/null 2>&1 || (echo "CUDA compiler (nvcc) not found. Please install CUDA toolkit." && exit 1)
	@echo "CUDA compiler found: $(shell which nvcc)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build the CUDA NMS test executable"
	@echo "  clean      - Remove build artifacts"
	@echo "  run        - Build and run the test"
	@echo "  check-cuda - Check if CUDA compiler is available"
	@echo "  help       - Show this help message"

.PHONY: all clean run check-cuda help 