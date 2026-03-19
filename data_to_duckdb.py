import redis
from rd3 import collect_redis_results_to_duckdb

redis_list = [
    redis.Redis(host="127.0.0.1", port=6379, decode_responses=False),
    redis.Redis(host="127.0.0.1", port=6380, decode_responses=False),
    redis.Redis(host="127.0.0.1", port=6381, decode_responses=False),
    redis.Redis(host="127.0.0.1", port=6382, decode_responses=False),
]
collect_redis_results_to_duckdb(
    # redis_host=redis_host,
    # redis_port=redis_port,
    # redis_password=redis_password,
    redis_list=redis_list,
    duckdb_path="results.duckdb",
)
