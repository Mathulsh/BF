import redis
from rd import collect_redis_results_to_duckdb, redis_list
# 服务器Redis实例

collect_redis_results_to_duckdb(
    # redis_host=redis_host,
    # redis_port=redis_port,
    # redis_password=redis_password,
    redis_list=redis_list,
    duckdb_path="results.duckdb",
    batch_size=500000,
)
