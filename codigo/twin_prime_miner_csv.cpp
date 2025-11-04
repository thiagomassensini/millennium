// Versão otimizada: escreve em CSV para importação posterior no MySQL
// g++ -O3 -march=native -mtune=native -flto -fopenmp -funroll-loops -ffast-math -DNDEBUG \
//     twin_prime_miner_csv.cpp -o miner_csv

#include <bits/stdc++.h>
#include <omp.h>
#include <csignal>

struct Config {
    uint64_t start_range = 1000000000000000ULL;
    uint64_t end_range   = 1010000000000000ULL;
    int threads = 56;
    uint64_t chunk_per_task = 200000000ULL;
    std::string output_file = "twin_primes_results.csv";
    std::string checkpoint_file = "miner_checkpoint.txt";
    int checkpoint_interval = 60; // segundos
} CFG;

static std::atomic<bool> RUNNING(true);
static void sig_handler(int){ RUNNING.store(false); }

struct State {
    std::atomic<uint64_t> tests{0}, found{0};
    std::array<std::atomic<uint64_t>,25> k_counts;
    std::chrono::steady_clock::time_point t0, last_checkpoint;
    std::ofstream out_file;
    std::mutex file_mutex;
    std::atomic<uint64_t> current_position;
    State(){ 
        for(auto& c:k_counts) c.store(0); 
        t0 = last_checkpoint = std::chrono::steady_clock::now(); 
        current_position.store(0);
    }
} G;

// Primalidade Miller-Rabin 64-bit determin�stica
static inline bool is_prime64(uint64_t n){
    if (n < 2) return false;
    if ((n & 1ULL) == 0ULL) return n == 2ULL;
    if (n % 3ULL == 0ULL) return n == 3ULL;
    if (n < 9ULL) return true;

    static const uint64_t bases[] = {2ULL,325ULL,9375ULL,28178ULL,450775ULL,9780504ULL,1795265022ULL};
    uint64_t d = n - 1, r = 0;
    while ((d & 1ULL) == 0ULL){ d >>= 1; ++r; }

    auto mod_pow = [](uint64_t a, uint64_t e, uint64_t m){
        __uint128_t res = 1, base = a % m;
        while (e){
            if (e & 1ULL) res = (res * base) % m;
            base = (base * base) % m;
            e >>= 1ULL;
        }
        return (uint64_t)res;
    };

    for (uint64_t a : bases){
        if (a % n == 0ULL) continue;
        uint64_t x = mod_pow(a, d, n);
        if (x == 1ULL || x == n-1) continue;
        bool comp = true;
        for (uint64_t i=1;i<r;i++){
            x = (uint64_t)(((__uint128_t)x * x) % n);
            if (x == n-1){ comp = false; break; }
        }
        if (comp) return false;
    }
    return true;
}

// Wheel30
static const int8_t W30_OFFS[8] = {1,7,11,13,17,19,23,29};
static inline uint64_t next_w30(uint64_t n){
    uint64_t base = (n/30ULL)*30ULL, off = n%30ULL;
    for (int i=0;i<8;i++) if ((uint64_t)W30_OFFS[i] > off) return base + (uint64_t)W30_OFFS[i];
    return base + 30ULL + (uint64_t)W30_OFFS[0];
}

// K real
static inline int calc_k(uint64_t p){
    if ((p & 1ULL) == 0ULL) return -1;
    uint64_t x = p ^ (p+2ULL);
    if (x > UINT64_MAX - 2ULL) return -1;
    uint64_t v = x + 2ULL;
    if ((v & (v-1ULL)) != 0ULL) return -1;
    int k = __builtin_ctzll(v) - 1;
    return (k >= 0 && k < 25) ? k : -1;
}

// Checkpoint functions
static void save_checkpoint(uint64_t position){
    std::ofstream ckp(CFG.checkpoint_file);
    if (ckp.is_open()){
        ckp << position << "\n";
        ckp << G.tests.load() << "\n";
        ckp << G.found.load() << "\n";
        ckp.close();
    }
}

static uint64_t load_checkpoint(){
    std::ifstream ckp(CFG.checkpoint_file);
    if (ckp.is_open()){
        uint64_t pos, tests, found;
        ckp >> pos >> tests >> found;
        ckp.close();
        G.tests.store(tests);
        G.found.store(found);
        std::cout<<"✅ Checkpoint carregado: posição "<<pos<<" | tests="<<tests<<" | found="<<found<<"\n";
        return pos;
    }
    return CFG.start_range;
}

// Monitor thread
static void monitor_thread(){
    using namespace std::chrono_literals;
    uint64_t last_f=0,last_t=0;
    auto last = std::chrono::steady_clock::now();
    while (RUNNING.load()){
        std::this_thread::sleep_for(2s);
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::milliseconds>(now-last).count()/1000.0;
        uint64_t f=G.found.load(), t=G.tests.load();
        double inst_f = (f-last_f)/dt;
        double secs = std::chrono::duration_cast<std::chrono::seconds>(now-G.t0).count();
        double avg = secs>0 ? double(f)/secs : 0.0;

        std::cout<<"\r🚀 "<<std::setw(6)<<(int)secs<<"s"
                 <<" | 🧬 "<<std::setw(12)<<f
                 <<" | Avg "<<std::setw(7)<<std::fixed<<std::setprecision(1)<<avg<<"/s"
                 <<" | Inst "<<std::setw(7)<<std::setprecision(1)<<inst_f<<"/s"
                 <<std::flush;

        last=now; last_f=f; last_t=t;
    }
    std::cout<<std::endl;
}

// Scan com buffer local
static void scan_task(uint64_t a, uint64_t b, int tid){
    std::vector<std::tuple<uint64_t,uint64_t,int,uint64_t>> local_buffer;
    local_buffer.reserve(10000);

    uint64_t p = next_w30(a|1ULL);
    while (p < b && RUNNING.load()){
        G.tests.fetch_add(1, std::memory_order_relaxed);
        if (is_prime64(p) && is_prime64(p+2ULL)){
            G.found.fetch_add(1, std::memory_order_relaxed);
            int k = calc_k(p);
            if (k>=0 && k<25){
                G.k_counts[k].fetch_add(1, std::memory_order_relaxed);
                local_buffer.emplace_back(p, p+2ULL, k, a);
                
                // Flush quando buffer encher
                if (local_buffer.size() >= 10000){
                    std::lock_guard<std::mutex> lock(G.file_mutex);
                    for (auto& [p_val, p2_val, k_val, range_val] : local_buffer){
                        G.out_file << p_val <<","<< p2_val <<","<< k_val <<","<< tid <<","<< range_val <<"\n";
                    }
                    local_buffer.clear();
                }
            }
        }
        p = next_w30(p+1ULL);
    }

    // Flush final
    if (!local_buffer.empty()){
        std::lock_guard<std::mutex> lock(G.file_mutex);
        for (auto& [p_val, p2_val, k_val, range_val] : local_buffer){
            G.out_file << p_val <<","<< p2_val <<","<< k_val <<","<< tid <<","<< range_val <<"\n";
        }
    }
}

int main(int argc, char** argv){
    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);

    // CLI
    for (int i=1;i<argc;i++){
        std::string s(argv[i]);
        if (s=="--threads" && i+1<argc) CFG.threads = std::atoi(argv[++i]);
        else if (s=="--start" && i+1<argc) CFG.start_range = std::strtoull(argv[++i],nullptr,10);
        else if (s=="--end" && i+1<argc) CFG.end_range = std::strtoull(argv[++i],nullptr,10);
        else if (s=="--output" && i+1<argc) CFG.output_file = argv[++i];
    }

    // Verificar checkpoint
    uint64_t resume_from = load_checkpoint();
    bool is_resume = (resume_from > CFG.start_range);
    
    if (is_resume){
        // Modo append - continuar arquivo existente
        G.out_file.open(CFG.output_file, std::ios::out | std::ios::app);
        CFG.start_range = resume_from;
    } else {
        // Modo novo - criar arquivo com header
        G.out_file.open(CFG.output_file, std::ios::out | std::ios::trunc);
        if (G.out_file.is_open()){
            G.out_file << "p,p_plus_2,k_real,thread_id,range_start\n"; // Header CSV
        }
    }
    
    if (!G.out_file.is_open()){
        std::cerr<<"❌ Não foi possível abrir arquivo de saída\n"; return 1;
    }
    
    G.current_position.store(CFG.start_range);

    std::thread monitor(monitor_thread);

    std::cout<<"🚀 TWIN PRIME MINER CSV MODE\n";
    std::cout<<"💻 Threads: "<<CFG.threads<<" | Output: "<<CFG.output_file<<"\n";
    std::cout<<"📊 Range: "<<CFG.start_range<<" → "<<CFG.end_range<<"\n";
    std::cout<<std::string(80,'=')<<"\n";

    uint64_t current = CFG.start_range;
    while (RUNNING.load() && current < CFG.end_range){
        uint64_t block_end = std::min(current + CFG.threads*CFG.chunk_per_task, CFG.end_range);
        uint64_t tasks = (block_end - current + CFG.chunk_per_task - 1ULL) / CFG.chunk_per_task;

        #pragma omp parallel for schedule(dynamic,1) num_threads(CFG.threads)
        for (long long t=0; t<(long long)tasks; t++){
            int tid = omp_get_thread_num();
            uint64_t a = current + (uint64_t)t * CFG.chunk_per_task;
            uint64_t b = std::min(a + CFG.chunk_per_task, block_end);
            if (a < b) scan_task(a,b,tid);
        }

        current = block_end;
        G.current_position.store(current);
        
        // Salvar checkpoint periodicamente
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - G.last_checkpoint).count() >= CFG.checkpoint_interval){
            G.out_file.flush(); // Garantir que dados foram escritos
            save_checkpoint(current);
            G.last_checkpoint = now;
            std::cout<<"\n💾 Checkpoint @ "<<current<<"\n";
        }
    }

    RUNNING.store(false);
    if (monitor.joinable()) monitor.join();

    G.out_file.close();
    
    // Salvar checkpoint final
    save_checkpoint(current);

    double secs = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-G.t0).count();
    std::cout<<"\n🏁 FIM\n";
    std::cout<<"🧮 Testes: "<<G.tests.load()<<"\n";
    std::cout<<"🧬 Gêmeos: "<<G.found.load()<<"\n";
    std::cout<<"⏱  Tempo: "<<secs<<"s\n";
    std::cout<<"📁 Resultado salvo em: "<<CFG.output_file<<"\n";
    std::cout<<"\n📊 DISTRIBUIÇÃO K (1..10):\n";
    for (int k=1;k<=10;k++) std::cout<<"  k="<<k<<": "<<G.k_counts[k].load()<<"\n";

    std::cout<<"\n💡 Para importar no MySQL:\n";
    std::cout<<"LOAD DATA LOCAL INFILE '"<<CFG.output_file<<"' INTO TABLE twin_primes\n";
    std::cout<<"FIELDS TERMINATED BY ',' LINES TERMINATED BY '\\n'\n";
    std::cout<<"IGNORE 1 ROWS (p, p_plus_2, k_real, thread_id, range_start);\n";

    return 0;
}
