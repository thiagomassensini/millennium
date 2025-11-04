// g++ -O3 -march=native -mtune=native -flto -fopenmp -funroll-loops -ffast-math -DNDEBUG -pthread \
//     twin_prime_miner_v5_ultra_mpmc.cpp -lmysqlclient -o miner_v5_ultra

#include <bits/stdc++.h>
#include <omp.h>
#include <mysql/mysql.h>
#include <csignal>

// ==================== CONFIG ====================
struct Config {
    std::string db_host = "localhost";
    std::string db_user = "prime_miner";
    std::string db_pass = ""; // via env PRIME_DB_PASS
    std::string db_name = "twin_primes_db";

    uint64_t start_range = 1000000000000000ULL; // 1e15
    uint64_t end_range   = 1010000000000000ULL; // 1e15 + 1e13

    int threads = 56;
    int writer_threads = 4; // Múltiplas threads de escrita
    uint64_t chunk_per_task = 100000000ULL; // 1e8 por tarefa dinâmica
    int batch_size = 50000;
    int checkpoint_every = 1800; // 30 min (reduzir I/O)
    int stats_every = 7200;      // 2 horas (reduzir I/O)
    bool live_monitor = true;
} CFG;

static std::atomic<bool> RUNNING(true);
static void sig_handler(int){ RUNNING.store(false); }

// ==================== PRIMALIDADE 64-BIT DETERMINÍSTICA ====================
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

// ==================== WHEEL30 ====================
static const int8_t W30_OFFS[8] = {1,7,11,13,17,19,23,29};
static inline uint64_t next_w30(uint64_t n){
    uint64_t base = (n/30ULL)*30ULL, off = n%30ULL;
    for (int i=0;i<8;i++) if ((uint64_t)W30_OFFS[i] > off) return base + (uint64_t)W30_OFFS[i];
    return base + 30ULL + (uint64_t)W30_OFFS[0];
}

// ==================== K REAL ====================
static inline int calc_k(uint64_t p){
    if ((p & 1ULL) == 0ULL) return -1;
    uint64_t x = p ^ (p+2ULL);
    if (x > UINT64_MAX - 2ULL) return -1;
    uint64_t v = x + 2ULL;
    if ((v & (v-1ULL)) != 0ULL) return -1;
    int k = __builtin_ctzll(v) - 1;
    return (k >= 0 && k < 25) ? k : -1;
}

// ==================== STATE ====================
struct State {
    std::atomic<uint64_t> tests{0}, found{0};
    std::array<std::atomic<uint64_t>,25> k_counts;
    uint64_t current_start, target_end;
    std::chrono::steady_clock::time_point t0, last_ckp, last_stats;
    State(){ for(auto& c:k_counts) c.store(0); t0 = last_ckp = last_stats = std::chrono::steady_clock::now(); }
} G;

// ==================== MPMC QUEUE (mutex + cv, bounded) ====================
struct Row {
    uint64_t p, p2, range_start;
    int k, tid;
};

class MPMCQueue {
    std::deque<Row> q;
    size_t cap;
    std::mutex m;
    std::condition_variable cv_not_full, cv_not_empty;
public:
    explicit MPMCQueue(size_t capacity): cap(capacity) {}
    void push_blocking(const Row& r){
        std::unique_lock<std::mutex> lk(m);
        cv_not_full.wait(lk, [&]{ return q.size() < cap || !RUNNING.load(); });
        if (!RUNNING.load() && q.size() >= cap) return;
        q.emplace_back(r);
        cv_not_empty.notify_one();
    }
    bool pop_wait(Row& out){
        std::unique_lock<std::mutex> lk(m);
        cv_not_empty.wait(lk, [&]{ return !q.empty() || !RUNNING.load(); });
        if (q.empty()) return false;
        out = q.front(); q.pop_front();
        cv_not_full.notify_one();
        return true;
    }
    bool empty(){
        std::lock_guard<std::mutex> lk(m);
        return q.empty();
    }
};

static MPMCQueue WRITE_Q(1<<22); // ~4M rows (~160-200MB) - fila maior para menos bloqueio

// ==================== MYSQL HELPERS ====================
static MYSQL* mysql_connect_or_die(const std::string& host,const std::string& user,
                                   const std::string& pass,const std::string& db){
    MYSQL* c = mysql_init(nullptr);
    if (!c) { std::cerr<<"MySQL init fail\n"; std::exit(1); }
    mysql_options(c, MYSQL_OPT_RECONNECT, "1");
    if (!mysql_real_connect(c, host.c_str(), user.c_str(), pass.c_str(), db.c_str(), 0, nullptr, 0)){
        std::cerr<<"MySQL connect fail: "<<mysql_error(c)<<"\n"; std::exit(1);
    }
    return c;
}

static bool mysql_reconnect(MYSQL*& c, const std::string& host,const std::string& user,
                            const std::string& pass,const std::string& db){
    if (c) mysql_close(c);
    c = mysql_init(nullptr);
    if (!c) return false;
    mysql_options(c, MYSQL_OPT_RECONNECT, "1");
    return mysql_real_connect(c, host.c_str(), user.c_str(), pass.c_str(), db.c_str(), 0, nullptr, 0) != nullptr;
}

static bool is_conn_lost(unsigned err){
    return err==CR_SERVER_GONE_ERROR || err==CR_SERVER_LOST || err==CR_CONN_HOST_ERROR;
}

// ==================== WRITER THREAD ====================
static void writer_thread(){
    MYSQL* conn = mysql_connect_or_die(CFG.db_host, CFG.db_user, CFG.db_pass, CFG.db_name);
    const char* SQL = "INSERT IGNORE INTO twin_primes (p,p_plus_2,k_real,thread_id,range_start) VALUES (?,?,?,?,?)";
    MYSQL_STMT* st = mysql_stmt_init(conn);
    if (!st || mysql_stmt_prepare(st, SQL, (unsigned long)strlen(SQL))){
        std::cerr<<"stmt prepare fail: "<<mysql_error(conn)<<"\n"; std::exit(1);
    }

    std::vector<Row> batch; batch.reserve(CFG.batch_size);

    auto flush_batch = [&](std::vector<Row>& b){
        if (b.empty()) return;
        if (mysql_query(conn, "START TRANSACTION")){
            unsigned e=mysql_errno(conn);
            if (is_conn_lost(e) && mysql_reconnect(conn,CFG.db_host,CFG.db_user,CFG.db_pass,CFG.db_name)){
                mysql_stmt_close(st);
                st=mysql_stmt_init(conn);
                mysql_stmt_prepare(st, SQL, (unsigned long)strlen(SQL));
            }
        }
        for (auto &r : b){
            MYSQL_BIND bind[5]; memset(bind,0,sizeof(bind));
            bind[0].buffer_type = MYSQL_TYPE_LONGLONG; bind[0].buffer=(void*)&r.p;     bind[0].is_unsigned=1;
            bind[1].buffer_type = MYSQL_TYPE_LONGLONG; bind[1].buffer=(void*)&r.p2;    bind[1].is_unsigned=1;
            bind[2].buffer_type = MYSQL_TYPE_TINY;     bind[2].buffer=(void*)&r.k;
            bind[3].buffer_type = MYSQL_TYPE_SHORT;    bind[3].buffer=(void*)&r.tid;
            bind[4].buffer_type = MYSQL_TYPE_LONGLONG; bind[4].buffer=(void*)&r.range_start; bind[4].is_unsigned=1;

            if (mysql_stmt_bind_param(st, bind) || mysql_stmt_execute(st)){
                unsigned e = mysql_errno(conn);
                if (is_conn_lost(e)){
                    // reconectar e tentar novamente esta linha
                    if (mysql_reconnect(conn,CFG.db_host,CFG.db_user,CFG.db_pass,CFG.db_name)){
                        mysql_stmt_close(st);
                        st=mysql_stmt_init(conn);
                        mysql_stmt_prepare(st, SQL, (unsigned long)strlen(SQL));
                        if (mysql_stmt_bind_param(st, bind) || mysql_stmt_execute(st)){
                            std::cerr<<"writer retry failed: "<<mysql_stmt_error(st)<<"\n";
                        }
                    } else {
                        std::cerr<<"writer reconnect failed permanently\n";
                    }
                } else {
                    std::cerr<<"writer exec err: "<<mysql_stmt_error(st)<<"\n";
                }
            }
        }
        mysql_query(conn, "COMMIT");
        b.clear();
    };

    while (RUNNING.load() || !WRITE_Q.empty()){
        Row r;
        while ((int)batch.size() < CFG.batch_size && WRITE_Q.pop_wait(r)) batch.push_back(r);
        flush_batch(batch);
        if (batch.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // flush final
    flush_batch(batch);
    mysql_stmt_close(st);
    mysql_close(conn);
}

// ==================== CHECKPOINT & STATS ====================
static void load_checkpoint(MYSQL* c){
    if (mysql_query(c, "SELECT current_start,target_end,total_tests,total_found FROM mining_checkpoint WHERE id=1")){
        std::cerr<<"checkpoint load err: "<<mysql_error(c)<<"\n";
        G.current_start = CFG.start_range; G.target_end = CFG.end_range; return;
    }
    MYSQL_RES* r = mysql_store_result(c);
    MYSQL_ROW row = mysql_fetch_row(r);
    if (row){
        G.current_start = std::strtoull(row[0],nullptr,10);
        G.target_end   = std::strtoull(row[1],nullptr,10);
        G.tests.store(std::strtoull(row[2],nullptr,10));
        G.found.store(std::strtoull(row[3],nullptr,10));
        std::cout<<"✅ Checkpoint: "<<G.current_start<<" → "<<G.target_end
                 <<" | tests="<<G.tests.load()<<" | found="<<G.found.load()<<"\n";
    } else {
        G.current_start = CFG.start_range; G.target_end = CFG.end_range;
    }
    mysql_free_result(r);
}

static void save_checkpoint(MYSQL* c){
    static uint64_t last_tests = 0, last_found = 0;
    uint64_t t = G.tests.load(), f = G.found.load();
    uint64_t dt = (t>=last_tests)?(t-last_tests):t, df = (f>=last_found)?(f-last_found):f;

    std::ostringstream q;
    q<<"CALL update_checkpoint_atomic("<<G.current_start<<","<<dt<<","<<df<<")";
    if (mysql_query(c, q.str().c_str())){
        std::cerr<<"checkpoint save err: "<<mysql_error(c)<<"\n";
    } else {
        last_tests = t; last_found = f;
    }
}

static void save_hourly_stats(MYSQL* c, uint64_t range_start, uint64_t range_end){
    auto now = std::chrono::steady_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::seconds>(now-G.t0).count();
    uint64_t f = G.found.load(), t = G.tests.load();
    double avg = secs>0 ? double(f)/secs : 0.0;
    double eff = t>0 ? (100.0*double(f)/double(t)) : 0.0;

    std::ostringstream json; json<<"{";
    for (int k=1;k<20;k++){ if (k>1) json<<","; json<<"\""<<k<<"\":"<<G.k_counts[k].load(); }
    json<<"}";

    std::ostringstream q;
    q<<"INSERT INTO hourly_stats (hour_timestamp,range_start,range_end,tests_performed,twins_found,avg_twins_per_second,efficiency_percent,k_distribution) "
     <<"VALUES (NOW(),"<<range_start<<","<<range_end<<","<<t<<","<<f<<","<<avg<<","<<eff<<",'"<<json.str()<<"')";
    if (mysql_query(c, q.str().c_str())){
        std::cerr<<"hourly stats err: "<<mysql_error(c)<<"\n";
    }
}

// ==================== MONITOR ====================
static void monitor_thread(){
    using namespace std::chrono_literals;
    uint64_t last_f=0,last_t=0;
    auto last = std::chrono::steady_clock::now();
    while (RUNNING.load()){
        std::this_thread::sleep_for(2s);
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::milliseconds>(now-last).count()/1000.0;
        uint64_t f=G.found.load(), t=G.tests.load();
        double inst_f = (f-last_f)/dt, inst_t=(t-last_t)/dt;
        double secs = std::chrono::duration_cast<std::chrono::seconds>(now-G.t0).count();
        double avg = secs>0 ? double(f)/secs : 0.0;
        double eff = t>0 ? (100.0*double(f)/double(t)) : 0.0;

        std::cout<<"\r🚀 "<<std::setw(6)<<(int)secs<<"s"
                 <<" | 🧬 "<<std::setw(12)<<f
                 <<" | Avg "<<std::setw(7)<<std::fixed<<std::setprecision(1)<<avg<<"/s"
                 <<" | Inst "<<std::setw(7)<<std::setprecision(1)<<inst_f<<"/s"
                 <<" | Tests "<<std::setw(9)<<std::setprecision(0)<<inst_t<<"/s"
                 <<" | Eff "<<std::setw(6)<<std::setprecision(3)<<eff<<"%"<<std::flush;

        last=now; last_f=f; last_t=t;
    }
    std::cout<<std::endl;
}

// ==================== SCAN RANGE (OpenMP dynamic) ====================
static void scan_task(uint64_t a, uint64_t b, int tid){
    uint64_t p = next_w30(a|1ULL);
    while (p < b && RUNNING.load()){
        G.tests.fetch_add(1, std::memory_order_relaxed);
        if (is_prime64(p) && is_prime64(p+2ULL)){
            G.found.fetch_add(1, std::memory_order_relaxed);
            int k = calc_k(p);
            if (k>=0){
                G.k_counts[k].fetch_add(1, std::memory_order_relaxed);
                WRITE_Q.push_blocking(Row{p, p+2ULL, a, k, tid});
            }
        }
        p = next_w30(p+1ULL);
    }
}

int main(int argc, char** argv){
    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);

    if (const char* pass = std::getenv("PRIME_DB_PASS")) CFG.db_pass = pass;
    else { std::cerr<<"❌ Set PRIME_DB_PASS\n"; return 1; }

    // CLI overrides (simples)
    for (int i=1;i<argc;i++){
        std::string s(argv[i]);
        auto next = [&](uint64_t& dst){ if (i+1<argc){ dst = std::strtoull(argv[++i],nullptr,10); } };
        auto nexti = [&](int& dst){ if (i+1<argc){ dst = std::atoi(argv[++i]); } };
        if (s=="--threads") nexti(CFG.threads);
        else if (s=="--start") next(CFG.start_range);
        else if (s=="--end") next(CFG.end_range);
        else if (s=="--chunk") next(CFG.chunk_per_task);
        else if (s=="--batch") nexti(CFG.batch_size);
    }
    if (CFG.threads <= 0) CFG.threads = std::max(1u,std::thread::hardware_concurrency());

    MYSQL* ck = mysql_connect_or_die(CFG.db_host, CFG.db_user, CFG.db_pass, CFG.db_name);
    load_checkpoint(ck);

    // Múltiplas threads escritoras para paralelizar I/O do banco
    std::vector<std::thread> writers;
    for (int i=0; i<CFG.writer_threads; i++){
        writers.emplace_back(writer_thread);
    }
    
    std::thread monitor;
    if (CFG.live_monitor) monitor = std::thread(monitor_thread);

    std::cout<<"🚀 TWIN PRIME MINER v5 ULTRA (MPMC) INICIADO\n";
    std::cout<<"💻 Threads: "<<CFG.threads<<" | Wheel30: ON | MR64: deterministic\n";
    std::cout<<"📊 Range: "<<G.current_start<<" → "<<G.target_end<<"\n";
    std::cout<<std::string(80,'=')<<"\n";

    uint64_t global_end = G.target_end;
    // Loop externo em tarefas dinâmicas (melhor balanceamento)
    while (RUNNING.load() && G.current_start < global_end){
        uint64_t block_start = G.current_start;
        uint64_t block_end = std::min(block_start + CFG.threads*CFG.chunk_per_task, global_end);

        uint64_t tasks = (block_end - block_start + CFG.chunk_per_task - 1ULL) / CFG.chunk_per_task;

        #pragma omp parallel for schedule(dynamic,1) num_threads(CFG.threads)
        for (long long t=0; t<(long long)tasks; t++){
            int tid = omp_get_thread_num();
            uint64_t a = block_start + (uint64_t)t * CFG.chunk_per_task;
            uint64_t b = std::min(a + CFG.chunk_per_task, block_end);
            if (a < b) scan_task(a,b,tid);
        }

        G.current_start = block_end;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now-G.last_ckp).count() >= CFG.checkpoint_every){
            save_checkpoint(ck);
            G.last_ckp = now;
            std::cout<<"\n💾 Checkpoint salvo @ "<<G.current_start<<"\n";
        }
        if (std::chrono::duration_cast<std::chrono::seconds>(now-G.last_stats).count() >= CFG.stats_every){
            save_hourly_stats(ck, block_start, block_end);
            G.last_stats = now;
            std::cout<<"\n📊 Stats salvas\n";
        }
    }

    RUNNING.store(false);
    std::cout<<"\n⏳ Finalizando writers...\n";
    for (auto& w : writers) if (w.joinable()) w.join();
    if (monitor.joinable()) monitor.join();

    save_checkpoint(ck);
    mysql_close(ck);

    double secs = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-G.t0).count();
    std::cout<<"\n🏁 FIM\n";
    std::cout<<"🧮 Testes: "<<G.tests.load()<<"\n";
    std::cout<<"🧬 Gêmeos: "<<G.found.load()<<"\n";
    std::cout<<"⏱  Tempo: "<<secs<<"s\n";
    std::cout<<"\n📊 DISTRIBUIÇÃO K (1..10):\n";
    for (int k=1;k<=10;k++) std::cout<<"  k="<<k<<": "<<G.k_counts[k].load()<<"\n";
    return 0;
}
