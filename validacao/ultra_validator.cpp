// ULTRA VALIDATOR - Carrega CSV inteiro na RAM e valida BILHÃO completo!
// Compile: g++ -O3 -march=native -fopenmp -std=c++17 ultra_validator.cpp -o ultra_validator

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

using u64 = uint64_t;

// Miller-Rabin ultra-otimizado
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
    return __builtin_ctzll((p ^ p2) + 2);
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
    std::cout << "ULTRA VALIDATOR v3.0 - BILHÃO COMPLETO!\n";
    std::cout << "=========================================\n";
    std::cout << "Arquivo: " << csv_file << "\n";
    std::cout << "Cores: " << cores << "\n";
    std::cout << "=========================================\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Mapeamento de memória do arquivo inteiro (zero-copy!)
    std::cout << "Mapeando CSV na memória (mmap)...\n";
    int fd = open(csv_file, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Erro abrindo arquivo\n";
        return 1;
    }
    
    struct stat sb;
    fstat(fd, &sb);
    size_t file_size = sb.st_size;
    
    char* data = (char*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (data == MAP_FAILED) {
        std::cerr << "Erro no mmap\n";
        return 1;
    }
    
    madvise(data, file_size, MADV_SEQUENTIAL);
    
    auto map_end = std::chrono::high_resolution_clock::now();
    double map_time = std::chrono::duration<double>(map_end - start).count();
    std::cout << "✓ Mapeado em " << map_time << "s\n\n";
    
    // Parsing ultra-rápido em paralelo
    std::cout << "Parsing CSV em paralelo...\n";
    
    // Contar linhas primeiro
    u64 total_lines = 0;
    char* ptr = data;
    char* end = data + file_size;
    while (ptr < end) {
        if (*ptr == '\n') total_lines++;
        ptr++;
    }
    total_lines--; // header
    
    std::cout << "Total de primos: " << total_lines << "\n";
    
    // Alocar vetor
    std::vector<TwinPrime> primes(total_lines);
    
    // Parsing paralelo por chunks
    const size_t chunk_size = 100000;
    u64 parsed = 0;
    
    ptr = data;
    // Skip header
    while (*ptr != '\n') ptr++;
    ptr++;
    
    #pragma omp parallel
    {
        u64 local_count = 0;
        char line_buf[256];
        
        #pragma omp for schedule(dynamic, chunk_size)
        for (u64 i = 0; i < total_lines; i++) {
            // Encontrar início da linha i
            char* line_start = data;
            u64 newlines = 0;
            while (newlines <= i && line_start < end) {
                if (*line_start == '\n') newlines++;
                line_start++;
            }
            
            // Copiar linha para buffer local
            char* line_end = line_start;
            size_t len = 0;
            while (*line_end != '\n' && line_end < end && len < 255) {
                line_buf[len++] = *line_end++;
            }
            line_buf[len] = '\0';
            
            // Parse rápido
            u64 p = 0, p2 = 0;
            int k = 0;
            
            char* tok = strtok(line_buf, ",");
            if (tok) p = strtoull(tok, nullptr, 10);
            tok = strtok(nullptr, ",");
            if (tok) p2 = strtoull(tok, nullptr, 10);
            tok = strtok(nullptr, ",");
            if (tok) k = atoi(tok);
            
            primes[i] = {p, p2, k};
            
            if (++local_count % 10000000 == 0) {
                #pragma omp critical
                {
                    parsed += 10000000;
                    std::cout << "  Progresso: " << (parsed * 100 / total_lines) << "%\r" << std::flush;
                }
                local_count = 0;
            }
        }
    }
    
    auto parse_end = std::chrono::high_resolution_clock::now();
    double parse_time = std::chrono::duration<double>(parse_end - map_end).count();
    std::cout << "\n✓ Parsed em " << parse_time << "s\n\n";
    
    // VALIDAÇÃO MASSIVA
    std::cout << "=== VALIDAÇÃO EM " << cores << " CORES ===\n\n";
    
    // TEST 1: Primalidade
    std::cout << "TEST 1: Primalidade (1 bilhão de pares)...\n";
    u64 both_prime = 0;
    #pragma omp parallel for reduction(+:both_prime) schedule(dynamic, 10000)
    for (size_t i = 0; i < primes.size(); i++) {
        if (is_prime64(primes[i].p) && is_prime64(primes[i].p2)) {
            both_prime++;
        }
        if (i % 10000000 == 0 && omp_get_thread_num() == 0) {
            std::cout << "  " << (i * 100 / primes.size()) << "%\r" << std::flush;
        }
    }
    auto t1_end = std::chrono::high_resolution_clock::now();
    double t1_time = std::chrono::duration<double>(t1_end - parse_end).count();
    std::cout << "\n  ✓ Válidos: " << both_prime << "/" << primes.size() 
              << " (" << (100.0 * both_prime / primes.size()) << "%)\n";
    std::cout << "  Tempo: " << t1_time << "s\n\n";
    
    // TEST 2: Valores K
    std::cout << "TEST 2: Correção dos valores k...\n";
    u64 k_correct = 0;
    #pragma omp parallel for reduction(+:k_correct) schedule(dynamic, 10000)
    for (size_t i = 0; i < primes.size(); i++) {
        if (calc_k(primes[i].p, primes[i].p2) == primes[i].k) {
            k_correct++;
        }
        if (i % 10000000 == 0 && omp_get_thread_num() == 0) {
            std::cout << "  " << (i * 100 / primes.size()) << "%\r" << std::flush;
        }
    }
    auto t2_end = std::chrono::high_resolution_clock::now();
    double t2_time = std::chrono::duration<double>(t2_end - t1_end).count();
    std::cout << "\n  ✓ Corretos: " << k_correct << "/" << primes.size()
              << " (" << (100.0 * k_correct / primes.size()) << "%)\n";
    std::cout << "  Tempo: " << t2_time << "s\n\n";
    
    // TEST 3: BSD Condition
    std::cout << "TEST 3: Condição BSD (k∈{2,4,8,16})...\n";
    u64 bsd_valid = 0, bsd_total = 0;
    #pragma omp parallel for reduction(+:bsd_valid,bsd_total) schedule(dynamic, 10000)
    for (size_t i = 0; i < primes.size(); i++) {
        int k = primes[i].k;
        if (k == 2 || k == 4 || k == 8 || k == 16) {
            bsd_total++;
            u64 k2 = (u64)k * k;
            if (primes[i].p % k2 == k2 - 1) bsd_valid++;
        }
        if (i % 10000000 == 0 && omp_get_thread_num() == 0) {
            std::cout << "  " << (i * 100 / primes.size()) << "%\r" << std::flush;
        }
    }
    auto t3_end = std::chrono::high_resolution_clock::now();
    double t3_time = std::chrono::duration<double>(t3_end - t2_end).count();
    std::cout << "\n  ✓ Válidos: " << bsd_valid << "/" << bsd_total
              << " (" << (100.0 * bsd_valid / bsd_total) << "%)\n";
    std::cout << "  Tempo: " << t3_time << "s\n\n";
    
    // TEST 4: Distribuição
    std::cout << "TEST 4: Distribuição P(k)=2^(-k)...\n";
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
    }
    
    std::cout << "\n  χ² = " << chi2 << " (crítico 95%: 23.685)\n";
    std::cout << "  Status: " << (chi2 < 23.685 ? "✓ EXCELENTE" : "✗ PROBLEMA") << "\n\n";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start).count();
    
    std::cout << "=========================================\n";
    std::cout << "VALIDAÇÃO COMPLETA!\n";
    std::cout << "=========================================\n";
    std::cout << "Total de primos testados: " << primes.size() << "\n";
    std::cout << "Tempo total: " << total_time << "s (" << (total_time/60) << " min)\n";
    std::cout << "Taxa: " << (primes.size() / total_time / 1000000) << " milhões/segundo\n";
    std::cout << "=========================================\n";
    
    munmap(data, file_size);
    close(fd);
    
    return 0;
}
