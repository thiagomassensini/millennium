// ULTRA VALIDATOR v4.0 - Parsing sequencial otimizado + validação paralela
// Compile: g++ -O3 -march=native -fopenmp -std=c++17 ultra_v4.cpp -o ultra_v4

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cstring>

using u64 = uint64_t;

inline bool is_prime64(u64 n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    
    static const u64 bases[] = {2, 3, 5, 7, 11, 13, 17};
    u64 d = n - 1, r = 0;
    while (!(d & 1)) { d >>= 1; r++; }
    
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

inline int calc_k(u64 p, u64 p2) {
    return __builtin_ctzll((p ^ p2) + 2) - 1;
}

struct TwinPrime {
    u64 p, p2;
    int k;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <csv_file> [cores]\n";
        return 1;
    }
    
    const char* csv_file = argv[1];
    int cores = (argc > 2) ? std::atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(cores);
    
    std::cout << "=========================================\n";
    std::cout << "ULTRA VALIDATOR v4.0 - PARSING FIXADO!\n";
    std::cout << "=========================================\n";
    std::cout << "Arquivo: " << csv_file << "\n";
    std::cout << "Cores: " << cores << "\n";
    std::cout << "=========================================\n\n";
    std::cout.flush();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Leitura SEQUENCIAL ultra-rápida (mais rápido que parsing paralelo bugado!)
    std::cout << "Lendo CSV sequencialmente (otimizado)...\n";
    std::cout.flush();
    
    std::vector<TwinPrime> primes;
    primes.reserve(1005000000); // 1.005 bilhão
    
    FILE* fp = fopen(csv_file, "r");
    if (!fp) {
        std::cerr << "Erro abrindo CSV\n";
        return 1;
    }
    
    char line[256];
    fgets(line, sizeof(line), fp); // skip header
    
    u64 count = 0;
    while (fgets(line, sizeof(line), fp)) {
        u64 p = 0, p2 = 0;
        int k = 0;
        
        // Parse super-rápido
        char* tok = strtok(line, ",");
        if (tok) p = strtoull(tok, nullptr, 10);
        tok = strtok(nullptr, ",");
        if (tok) p2 = strtoull(tok, nullptr, 10);
        tok = strtok(nullptr, ",");
        if (tok) k = atoi(tok);
        
        primes.push_back({p, p2, k});
        
        if (++count % 50000000 == 0) {
            std::cout << "  " << (count / 1000000) << "M linhas lidas\n";
            std::cout.flush();
        }
    }
    fclose(fp);
    
    auto parse_end = std::chrono::high_resolution_clock::now();
    double parse_time = std::chrono::duration<double>(parse_end - start).count();
    std::cout << "\n✓ Lidos " << primes.size() << " pares em " << parse_time << "s\n";
    std::cout << "  Taxa: " << (primes.size() / parse_time / 1000000) << "M linhas/s\n\n";
    std::cout.flush();
    
    // VALIDAÇÃO PARALELA
    std::cout << "=== VALIDAÇÃO EM " << cores << " CORES ===\n\n";
    std::cout.flush();
    
    // TEST 1: Primalidade
    std::cout << "TEST 1: Primalidade...\n";
    std::cout.flush();
    u64 both_prime = 0;
    auto t1_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for reduction(+:both_prime) schedule(dynamic, 10000)
    for (size_t i = 0; i < primes.size(); i++) {
        if (is_prime64(primes[i].p) && is_prime64(primes[i].p2)) {
            both_prime++;
        }
        if (i % 50000000 == 0 && omp_get_thread_num() == 0) {
            std::cout << "  " << (i * 100 / primes.size()) << "%\r";
            std::cout.flush();
        }
    }
    
    auto t1_end = std::chrono::high_resolution_clock::now();
    double t1_time = std::chrono::duration<double>(t1_end - t1_start).count();
    std::cout << "\n  ✓ Válidos: " << both_prime << "/" << primes.size() 
              << " (" << (100.0 * both_prime / primes.size()) << "%)\n";
    std::cout << "  Tempo: " << t1_time << "s (" << (t1_time/60) << " min)\n\n";
    std::cout.flush();
    
    // TEST 2: Valores K
    std::cout << "TEST 2: Valores K...\n";
    std::cout.flush();
    u64 k_correct = 0;
    auto t2_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for reduction(+:k_correct) schedule(dynamic, 10000)
    for (size_t i = 0; i < primes.size(); i++) {
        if (calc_k(primes[i].p, primes[i].p2) == primes[i].k) {
            k_correct++;
        }
    }
    
    auto t2_end = std::chrono::high_resolution_clock::now();
    double t2_time = std::chrono::duration<double>(t2_end - t2_start).count();
    std::cout << "  ✓ Corretos: " << k_correct << "/" << primes.size()
              << " (" << (100.0 * k_correct / primes.size()) << "%)\n";
    std::cout << "  Tempo: " << t2_time << "s\n\n";
    std::cout.flush();
    
    // TEST 3: BSD
    std::cout << "TEST 3: BSD Condition...\n";
    std::cout.flush();
    u64 bsd_valid = 0, bsd_total = 0;
    auto t3_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for reduction(+:bsd_valid,bsd_total) schedule(dynamic, 10000)
    for (size_t i = 0; i < primes.size(); i++) {
        int k = primes[i].k;
        if (k == 2 || k == 4 || k == 8 || k == 16) {
            bsd_total++;
            u64 k2 = (u64)k * k;
            if (primes[i].p % k2 == k2 - 1) bsd_valid++;
        }
    }
    
    auto t3_end = std::chrono::high_resolution_clock::now();
    double t3_time = std::chrono::duration<double>(t3_end - t3_start).count();
    std::cout << "  ✓ Válidos: " << bsd_valid << "/" << bsd_total
              << " (" << (100.0 * bsd_valid / bsd_total) << "%)\n";
    std::cout << "  Tempo: " << t3_time << "s\n\n";
    std::cout.flush();
    
    // TEST 4: Distribuição
    std::cout << "TEST 4: Distribuição...\n";
    std::cout.flush();
    std::vector<u64> k_counts(30, 0);
    
    #pragma omp parallel
    {
        std::vector<u64> local_counts(30, 0);
        #pragma omp for schedule(dynamic, 10000)
        for (size_t i = 0; i < primes.size(); i++) {
            if (primes[i].k < 30) local_counts[primes[i].k]++;
        }
        #pragma omp critical
        for (int k = 0; k < 30; k++) k_counts[k] += local_counts[k];
    }
    
    double chi2 = 0.0;
    for (int k = 1; k <= 15; k++) {
        double expected = primes.size() * std::pow(0.5, k) / (1.0 - std::pow(0.5, 16));
        double observed = k_counts[k];
        if (expected > 5) {
            chi2 += (observed - expected) * (observed - expected) / expected;
        }
        double pct_obs = 100.0 * observed / primes.size();
        double pct_exp = 100.0 * expected / primes.size();
        std::cout << "  k=" << k << ": " << pct_obs << "% (esperado " << pct_exp << "%)\n";
        std::cout.flush();
    }
    
    std::cout << "\n  χ² = " << chi2 << " (crítico 95%: 23.685)\n";
    std::cout << "  Status: " << (chi2 < 23.685 ? "✓ EXCELENTE" : "✗ PROBLEMA") << "\n\n";
    std::cout.flush();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start).count();
    
    std::cout << "=========================================\n";
    std::cout << "VALIDAÇÃO COMPLETA!\n";
    std::cout << "=========================================\n";
    std::cout << "Total testado: " << primes.size() << " pares\n";
    std::cout << "Tempo total: " << total_time << "s (" << (total_time/60) << " min)\n";
    std::cout << "Taxa: " << (primes.size() / total_time / 1000000) << " milhões/segundo\n";
    std::cout << "=========================================\n";
    std::cout.flush();
    
    return 0;
}
