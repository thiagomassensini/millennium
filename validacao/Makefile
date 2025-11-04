# Makefile for XOR Millennium Framework - Massive Validation System

CXX = g++
CXXFLAGS = -O3 -march=native -mtune=native -flto -fopenmp -funroll-loops -ffast-math -DNDEBUG -pthread
LDFLAGS = -lmysqlclient

TARGET = massive_validator
SOURCE = massive_validation.cpp

.PHONY: all clean test validate full_validation quick_test

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Compiling massive validation system..."
	$(CXX) $(CXXFLAGS) $(SOURCE) $(LDFLAGS) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

# Quick test (10k samples, fast)
quick_test: $(TARGET)
	@echo "Running quick validation (10,000 samples)..."
	./run_validation.sh 10000 $(shell nproc)

# Standard validation (1M samples)
validate: $(TARGET)
	@echo "Running standard validation (1,000,000 samples)..."
	./run_validation.sh 1000000 $(shell nproc)

# Full validation (10M samples, ~2-3 hours)
full_validation: $(TARGET)
	@echo "Running FULL validation (10,000,000 samples)..."
	@echo "WARNING: This will take 2-3 hours with 56 cores"
	@read -p "Continue? [y/N] " -n 1 -r; echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		./run_validation.sh 10000000 $(shell nproc); \
	fi

# Ultra validation (100M samples, overnight run)
ultra_validation: $(TARGET)
	@echo "Running ULTRA validation (100,000,000 samples)..."
	@echo "WARNING: This will take 12+ hours even with 56 cores"
	@read -p "Continue? [y/N] " -n 1 -r; echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		./run_validation.sh 100000000 $(shell nproc); \
	fi

clean:
	rm -f $(TARGET)
	rm -rf validation_results_*/
	@echo "Cleaned build artifacts and validation results"

help:
	@echo "XOR Millennium Framework - Massive Validation System"
	@echo ""
	@echo "Usage:"
	@echo "  make                  - Compile validator"
	@echo "  make quick_test       - Quick test (10k samples, ~30 seconds)"
	@echo "  make validate         - Standard validation (1M samples, ~5 minutes)"
	@echo "  make full_validation  - Full validation (10M samples, ~2-3 hours)"
	@echo "  make ultra_validation - Ultra validation (100M samples, ~12+ hours)"
	@echo "  make clean            - Remove build artifacts"
	@echo ""
	@echo "Requirements:"
	@echo "  - Set PRIME_DB_PASS environment variable"
	@echo "  - MySQL database with twin_primes table"
	@echo "  - g++ with OpenMP support"
	@echo ""
