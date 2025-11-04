# XOR Millennium Framework - Massive Validation System

## Overview

This validation system performs comprehensive testing of the XOR Millennium Framework claims across 1 billion twin primes with full parallelization (56 cores, 60GB RAM).

## Features

- **Parallel Processing**: Leverages all 56 CPU cores with OpenMP
- **4 Test Suites**:
  1. Twin Prime Validity (both p and p+2 are prime)
  2. K Value Correctness (k_real computation)
  3. BSD Condition (p ≡ k²-1 mod k² for elliptic curves)
  4. Distribution Test (P(k) = 2^(-k) chi-squared)
  
- **Output Formats**:
  - JSON (machine-readable)
  - CSV (spreadsheet-compatible)
  - LaTeX (paper appendix)
  - Log (human-readable)

## Quick Start

```bash
# Set database password
export PRIME_DB_PASS="your_password"

# Quick test (30 seconds)
make quick_test

# Standard validation (5 minutes)
make validate

# Full validation (2-3 hours)
make full_validation
```

## Installation

### Requirements

- g++ with C++17 support
- OpenMP (usually included with g++)
- MySQL client library: `sudo apt install libmysqlclient-dev`
- MySQL database with twin_primes table

### Build

```bash
make
```

## Usage

### Option 1: Makefile (recommended)

```bash
make quick_test       # 10k samples (~30s)
make validate         # 1M samples (~5min)
make full_validation  # 10M samples (~2-3h)
make ultra_validation # 100M samples (~12h+)
```

### Option 2: Direct Script

```bash
./run_validation.sh [sample_size] [cores]

# Examples:
./run_validation.sh 10000 56       # 10k samples, 56 cores
./run_validation.sh 1000000 32     # 1M samples, 32 cores
```

### Option 3: Manual

```bash
# Compile
g++ -O3 -march=native -mtune=native -flto -fopenmp -funroll-loops \
    -ffast-math -DNDEBUG -pthread massive_validation.cpp \
    -lmysqlclient -o massive_validator

# Run
./massive_validator --cores 56 --sample 1000000 --output results.json
```

## Test Descriptions

### Test 1: Twin Prime Validity

Verifies that stored pairs (p, p+2) are both prime using Miller-Rabin deterministic test (7 bases, 100% accurate for 64-bit integers).

**Expected**: 100% valid (zero exceptions)

### Test 2: K Value Correctness

Recomputes k_real = log₂((p ⊕ (p+2)) + 2) - 1 for all pairs and compares with stored values.

**Expected**: 100% match

### Test 3: BSD Condition

Validates p ≡ k²-1 (mod k²) for k ∈ {2,4,8,16}, the condition required for deterministic rank formula.

**Expected**: 100% valid (zero exceptions)

### Test 4: Distribution Test

Chi-squared test comparing observed k distribution with theoretical P(k) = 2^(-k).

**Expected**: χ² < 23.685 (p > 0.05, excellent fit)

## Output Files

After validation, results are saved in `validation_results_YYYYMMDD_HHMMSS/`:

- `validation.json` - Complete results in JSON format
- `validation_results.csv` - Summary metrics in CSV
- `validation_report.tex` - LaTeX appendix for papers
- `validation_standalone.pdf` - Standalone PDF report
- `validation.log` - Full execution log
- `SUMMARY.txt` - Quick summary

## Performance

### Estimated Runtimes (56 cores)

| Sample Size | Time | Use Case |
|------------|------|----------|
| 10k | 30 seconds | Quick sanity check |
| 100k | 2 minutes | Development testing |
| 1M | 5 minutes | Standard validation |
| 10M | 2-3 hours | Full paper validation |
| 100M | 12+ hours | Ultra-comprehensive |

### Resource Usage

- **CPU**: 56 cores @ 100% during tests
- **RAM**: ~4-8 GB (scales with sample size)
- **Disk I/O**: Minimal (results < 10 MB)
- **Network**: MySQL queries (insignificant)

## Integration with Papers

The generated `validation_report.tex` can be directly included in LaTeX papers:

```latex
\appendix
\input{validation_results_YYYYMMDD_HHMMSS/validation_report.tex}
```

## Database Schema

Expected table structure:

```sql
CREATE TABLE twin_primes (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    p BIGINT UNSIGNED NOT NULL,
    p_plus_2 BIGINT UNSIGNED NOT NULL,
    k_real TINYINT,
    thread_id SMALLINT,
    range_start BIGINT UNSIGNED,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_p (p),
    INDEX idx_k (k_real)
);
```

## Troubleshooting

### "MySQL connect failed"

```bash
# Check MySQL is running
sudo systemctl status mysql

# Test connection
mysql -u prime_miner -p twin_primes_db
```

### "Compilation failed"

```bash
# Install dependencies
sudo apt install build-essential libmysqlclient-dev

# Check g++ version (need 7.0+)
g++ --version
```

### "Out of memory"

Reduce sample size or increase swap:

```bash
./run_validation.sh 100000 56  # Use 100k instead of 1M
```

## Citation

When using this validation system in publications, cite:

```bibtex
@misc{silva2025xor,
  author = {Thiago Fernandes Motta Massensini Silva},
  title = {XOR Millennium Framework: Massive Validation System},
  year = {2025},
  url = {https://github.com/thiagomassensini/rg}
}
```

## License

MIT License - See LICENSE file

## Author

Thiago Fernandes Motta Massensini Silva  
Email: thiago@massensini.com.br  
GitHub: @thiagomassensini

---

**Status**: Production Ready  
**Version**: 1.0  
**Last Updated**: November 3, 2025
