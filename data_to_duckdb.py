from rd import redis_host, redis_port, redis_password, collect_redis_results_to_duckdb
import time

time_start = time.time()
collect_redis_results_to_duckdb(
    redis_host=redis_host,
    redis_port=redis_port,
    redis_password=redis_password,
    queue_name="results",
    duckdb_path="results.duckdb",
    table_name="results"
)
time_end = time.time()
print(f"Data collection completed in {time_end - time_start:.2f} s.")