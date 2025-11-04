#!/bin/bash
# Massive Validation Automation Script
# Usage: ./run_validation.sh [sample_size] [cores]

set -e

SAMPLE_SIZE=${1:-1000000}  # Default: 1M
CORES=${2:-56}             # Default: 56 cores
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="validation_results_${TIMESTAMP}"

echo "=========================================="
echo "XOR MILLENNIUM FRAMEWORK"
echo "Massive Validation System"
echo "=========================================="
echo "Sample size: $SAMPLE_SIZE"
echo "Cores: $CORES"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check database password
if [ -z "$PRIME_DB_PASS" ]; then
    echo "ERROR: Set PRIME_DB_PASS environment variable"
    exit 1
fi

# Compile validator
echo ""
echo "[1/5] Compiling validator..."
g++ -O3 -march=native -mtune=native -flto -fopenmp -funroll-loops \
    -ffast-math -DNDEBUG -pthread \
    massive_validation.cpp -lmysqlclient -o massive_validator

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "Compilation successful"

# Run validation
echo ""
echo "[2/5] Running validation tests..."
./massive_validator --cores $CORES --sample $SAMPLE_SIZE \
    --output "$OUTPUT_DIR/validation.json" 2>&1 | tee "$OUTPUT_DIR/validation.log"

# Move generated files
mv validation_results.csv "$OUTPUT_DIR/" 2>/dev/null || true
mv validation_report.tex "$OUTPUT_DIR/" 2>/dev/null || true

# Generate summary
echo ""
echo "[3/5] Generating summary..."
cat > "$OUTPUT_DIR/SUMMARY.txt" <<EOF
XOR MILLENNIUM FRAMEWORK - VALIDATION SUMMARY
==============================================
Timestamp: $(date)
Sample size: $SAMPLE_SIZE twin prime pairs
Cores: $CORES
Duration: $(grep "Total time:" "$OUTPUT_DIR/validation.log" | tail -1 || echo "N/A")

RESULTS:
--------
$(grep -A 20 "VALIDATION COMPLETE" "$OUTPUT_DIR/validation.log" || echo "See validation.log")

FILES GENERATED:
----------------
- validation.json     : Machine-readable results
- validation_results.csv : CSV format
- validation_report.tex : LaTeX appendix
- validation.log      : Full execution log
- SUMMARY.txt         : This file

STATUS: $([ -f "$OUTPUT_DIR/validation.json" ] && echo "SUCCESS" || echo "FAILED")
EOF

# Compile LaTeX report if available
echo ""
echo "[4/5] Compiling LaTeX report..."
if [ -f "$OUTPUT_DIR/validation_report.tex" ]; then
    cd "$OUTPUT_DIR"
    cat > validation_standalone.tex <<'EOFTEX'
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amsthm,amssymb}
\usepackage{geometry}
\usepackage{hyperref}
\geometry{margin=1in}

\title{XOR Millennium Framework\\Massive Validation Report}
\author{Thiago Fernandes Motta Massensini Silva}
\date{\today}

\begin{document}
\maketitle

\input{validation_report.tex}

\end{document}
EOFTEX
    
    pdflatex -interaction=nonstopmode validation_standalone.tex > /dev/null 2>&1
    pdflatex -interaction=nonstopmode validation_standalone.tex > /dev/null 2>&1
    
    if [ -f validation_standalone.pdf ]; then
        echo "PDF report generated: $OUTPUT_DIR/validation_standalone.pdf"
    fi
    cd - > /dev/null
fi

# Final report
echo ""
echo "[5/5] Validation complete!"
echo ""
cat "$OUTPUT_DIR/SUMMARY.txt"

echo ""
echo "=========================================="
echo "All files saved to: $OUTPUT_DIR/"
echo "=========================================="
