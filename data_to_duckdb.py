from acln.rd import redis_host, redis_port, redis_password, collect_redis_results_to_duckdb
collect_redis_results_to_duckdb(
    redis_host=redis_host,
    redis_port=redis_port,
    redis_password=redis_password,
    queue_name="results",
    duckdb_path="results1.duckdb",
    table_name="results"
)