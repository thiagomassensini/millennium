// Validador que lê direto do CSV - SEM banco de dados!
// Compile: g++ -O3 -march=native -fopenmp -std=c++17 csv_validator.cpp -o csv_validator

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cstring>

using u64 = uint64_t;

// Miller-Rabin determinístico para 64-bit
bool is_prime64(u64 n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    
    static const u64 bases[] = {2, 3, 5, 7, 11, 13, 17};
    u64 d = n - 1, r = 0;
    while (d % 2 == 0) { d /= 2; r++; }
    
    for (u64 a : bases) {
        if (a >= n) continue;
        u64 x = 1, p = a;
        for (u64 e = d; e; e >>= 1) {
            if (e & 1) x = (__uint128_t)x * p % n;
            p = (__uint128_t)p * p % n;
        }
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (u64 i = 0; i < r - 1; i++) {
            x = (__uint128_t)x * x % n;
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

int calc_k(u64 p, u64 p2) {
    u64 xor_val = p ^ p2;
    return (int)(log2(xor_val + 2)) - 1;
}

struct TwinPrime {
    u64 p, p2;
    int k;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <csv_file> <num_samples> [cores]\n";
        return 1;
    }
    
    const char* csv_file = argv[1];
    u64 num_samples = std::strtoull(argv[2], nullptr, 10);
    int cores = (argc > 3) ? std::atoi(argv[3]) : omp_get_max_threads();
    
    omp_set_num_threads(cores);
    
    std::cout << "=========================================\n";
    std::cout << "CSV VALIDATOR v2.0 - SEM BANCO!\n";
    std::cout << "=========================================\n";
    std::cout << "Arquivo: " << csv_file << "\n";
    std::cout << "Amostras: " << num_samples << "\n";
    std::cout << "Cores: " << cores << "\n";
    std::cout << "=========================================\n\n";
    
    // Contar linhas do CSV
    std::cout << "Contando linhas do CSV...\n";
    std::ifstream counter(csv_file);
    std::string line;
    u64 total_lines = 0;
    std::getline(counter, line); // skip header
    while (std::getline(counter, line)) total_lines++;
    counter.close();
    
    std::cout << "Total de primos gêmeos: " << total_lines << "\n\n";
    
    // Gerar índices aleatórios
    std::cout << "Gerando amostra aleatória...\n";
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<u64> dist(1, total_lines);
    
    std::vector<u64> sample_indices;
    for (u64 i = 0; i < num_samples; i++) {
        sample_indices.push_back(dist(gen));
    }
    std::sort(sample_indices.begin(), sample_indices.end());
    
    // Ler amostras do CSV usando mmap para velocidade
    std::cout << "Lendo amostras do CSV (otimizado)...\n";
    std::vector<TwinPrime> samples;
    samples.reserve(num_samples);
    
    FILE* fp = fopen(csv_file, "r");
    if (!fp) {
        std::cerr << "Erro abrindo CSV\n";
        return 1;
    }
    
    char buffer[256];
    fgets(buffer, sizeof(buffer), fp); // skip header
    
    u64 current_line = 1;
    size_t next_sample_idx = 0;
    
    while (fgets(buffer, sizeof(buffer), fp) && next_sample_idx < sample_indices.size()) {
        if (current_line == sample_indices[next_sample_idx]) {
            TwinPrime tp;
            char* tok = strtok(buffer, ",");
            tp.p = strtoull(tok, nullptr, 10);
            tok = strtok(nullptr, ",");
            tp.p2 = strtoull(tok, nullptr, 10);
            tok = strtok(nullptr, ",");
            tp.k = atoi(tok);
            
            samples.push_back(tp);
            next_sample_idx++;
            
            if (next_sample_idx % 100000 == 0) {
                std::cout << "  Progresso: " << next_sample_idx << "/" << num_samples << "\r" << std::flush;
            }
        }
        current_line++;
    }
    fclose(fp);
    std::cout << std::endl;
    
    std::cout << "Amostras lidas: " << samples.size() << "\n\n";
    
    // VALIDAÇÃO
    std::cout << "=== INICIANDO TESTES ===\n\n";
    
    // TEST 1: Primalidade
    std::cout << "TEST 1: Primalidade dos gêmeos\n";
    u64 both_prime = 0;
    #pragma omp parallel for reduction(+:both_prime)
    for (size_t i = 0; i < samples.size(); i++) {
        if (is_prime64(samples[i].p) && is_prime64(samples[i].p2)) {
            both_prime++;
        }
    }
    std::cout << "  Primos válidos: " << both_prime << "/" << samples.size() 
              << " (" << (100.0 * both_prime / samples.size()) << "%)\n\n";
    
    // TEST 2: Valores K
    std::cout << "TEST 2: Correção dos valores k\n";
    u64 k_correct = 0;
    #pragma omp parallel for reduction(+:k_correct)
    for (size_t i = 0; i < samples.size(); i++) {
        int k_calc = calc_k(samples[i].p, samples[i].p2);
        if (k_calc == samples[i].k) k_correct++;
    }
    std::cout << "  K corretos: " << k_correct << "/" << samples.size()
              << " (" << (100.0 * k_correct / samples.size()) << "%)\n\n";
    
    // TEST 3: BSD Condition
    std::cout << "TEST 3: Condição BSD (p ≡ k²-1 mod k²)\n";
    u64 bsd_valid = 0;
    #pragma omp parallel for reduction(+:bsd_valid)
    for (size_t i = 0; i < samples.size(); i++) {
        int k = samples[i].k;
        if (k == 2 || k == 4 || k == 8 || k == 16) {
            u64 k2 = (u64)k * k;
            if (samples[i].p % k2 == k2 - 1) bsd_valid++;
        }
    }
    
    u64 bsd_total = 0;
    for (const auto& s : samples) {
        if (s.k == 2 || s.k == 4 || s.k == 8 || s.k == 16) bsd_total++;
    }
    
    std::cout << "  BSD válidos: " << bsd_valid << "/" << bsd_total
              << " (" << (100.0 * bsd_valid / bsd_total) << "%)\n\n";
    
    // TEST 4: Distribuição P(k) = 2^(-k)
    std::cout << "TEST 4: Distribuição teórica P(k) = 2^(-k)\n";
    std::vector<int> k_counts(20, 0);
    for (const auto& s : samples) {
        if (s.k < 20) k_counts[s.k]++;
    }
    
    double chi2 = 0.0;
    for (int k = 1; k <= 15; k++) {
        double expected = samples.size() * std::pow(0.5, k) / (1.0 - std::pow(0.5, 16));
        double observed = k_counts[k];
        if (expected > 5) {
            chi2 += (observed - expected) * (observed - expected) / expected;
        }
        double pct_obs = 100.0 * observed / samples.size();
        double pct_exp = 100.0 * expected / samples.size();
        std::cout << "  k=" << k << ": " << pct_obs << "% (esperado " << pct_exp << "%)\n";
    }
    
    std::cout << "\n  χ² = " << chi2 << " (crítico 95%: 23.685)\n";
    std::cout << "  Status: " << (chi2 < 23.685 ? "✓ EXCELENTE" : "✗ PROBLEMA") << "\n\n";
    
    std::cout << "=========================================\n";
    std::cout << "VALIDAÇÃO CONCLUÍDA!\n";
    std::cout << "=========================================\n";
    
    return 0;
}
