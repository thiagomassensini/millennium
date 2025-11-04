CREATE DATABASE IF NOT EXISTS twin_primes_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'prime_miner'@'localhost'
  IDENTIFIED BY 'REPLACE_WITH_SECRET';

GRANT ALL PRIVILEGES ON twin_primes_db.* TO 'prime_miner'@'localhost';
FLUSH PRIVILEGES;

USE twin_primes_db;

-- Resultados (evitar duplicatas de p)
CREATE TABLE IF NOT EXISTS twin_primes (
    id BIGINT UNSIGNED AUTO_INCREMENT,
    p BIGINT UNSIGNED NOT NULL,
    p_plus_2 BIGINT UNSIGNED NOT NULL,
    k_real TINYINT UNSIGNED NOT NULL,
    thread_id SMALLINT UNSIGNED NOT NULL,
    range_start BIGINT UNSIGNED NOT NULL,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, k_real),
    UNIQUE KEY uniq_p_k (p, k_real),
    INDEX idx_p (p),
    INDEX idx_k (k_real),
    INDEX idx_range (range_start)
) ENGINE=InnoDB
  ROW_FORMAT=COMPRESSED
  KEY_BLOCK_SIZE=8
  PARTITION BY HASH (k_real) PARTITIONS 25;

-- Checkpoint
CREATE TABLE IF NOT EXISTS mining_checkpoint (
    id INT PRIMARY KEY DEFAULT 1,
    current_start BIGINT UNSIGNED NOT NULL,
    target_end BIGINT UNSIGNED NOT NULL,
    total_tests BIGINT UNSIGNED DEFAULT 0,
    total_found BIGINT UNSIGNED DEFAULT 0,
    last_flushed_tests BIGINT UNSIGNED DEFAULT 0,
    last_flushed_found BIGINT UNSIGNED DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT single_checkpoint CHECK (id = 1)
) ENGINE=InnoDB;

INSERT INTO mining_checkpoint (id, current_start, target_end)
VALUES (1, 1000000000000000, 1010000000000000)
ON DUPLICATE KEY UPDATE id=id;

-- Stats
CREATE TABLE IF NOT EXISTS hourly_stats (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    hour_timestamp TIMESTAMP NOT NULL,
    range_start BIGINT UNSIGNED NOT NULL,
    range_end BIGINT UNSIGNED NOT NULL,
    tests_performed BIGINT UNSIGNED NOT NULL,
    twins_found INT UNSIGNED NOT NULL,
    avg_twins_per_second DECIMAL(16,6),
    efficiency_percent DECIMAL(9,6),
    k_distribution JSON,
    INDEX idx_hour (hour_timestamp)
) ENGINE=InnoDB ROW_FORMAT=COMPRESSED;

-- (Opcional) Tuning global – requer privilégios SUPER e avaliação do ambiente
-- SET GLOBAL innodb_buffer_pool_size = 40 * 1024 * 1024 * 1024;
-- SET GLOBAL innodb_log_file_size     = 2  * 1024 * 1024 * 1024;
-- SET GLOBAL innodb_flush_log_at_trx_commit = 2;
-- SET GLOBAL max_allowed_packet = 256 * 1024 * 1024;
-- SET GLOBAL innodb_flush_method = O_DIRECT;

DELIMITER $$
CREATE PROCEDURE update_checkpoint_atomic(
    IN new_start BIGINT UNSIGNED,
    IN tests_delta BIGINT UNSIGNED,
    IN found_delta BIGINT UNSIGNED
)
BEGIN
    START TRANSACTION;
    UPDATE mining_checkpoint
       SET current_start = new_start,
           total_tests   = total_tests + tests_delta,
           total_found   = total_found + found_delta,
           last_flushed_tests  = last_flushed_tests  + tests_delta,
           last_flushed_found  = last_flushed_found  + found_delta
     WHERE id = 1;
    COMMIT;
END$$
DELIMITER ;

SELECT 'Database v5 ultra ready!' AS status;
