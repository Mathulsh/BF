import redis
from rd3 import collect_redis_results_to_duckdb
# 服务器Redis实例
redis_list = [
    redis.Redis(host="10.64.199.25", port=41882, decode_responses=False),
    redis.Redis(host="10.64.199.63", port=41883, decode_responses=False),
    redis.Redis(host="10.64.199.62", port=41884, decode_responses=False),
    redis.Redis(host="10.64.199.26", port=41885, decode_responses=False),
]

# 本机Redis实例
# redis_list = [
#     redis.Redis(host="127.0.0.1", port=6379, decode_responses=False),
#     redis.Redis(host="127.0.0.1", port=6380, decode_responses=False),
#     redis.Redis(host="127.0.0.1", port=6381, decode_responses=False),
#     redis.Redis(host="127.0.0.1", port=6382, decode_responses=False),
# ]

collect_redis_results_to_duckdb(
    # redis_host=redis_host,
    # redis_port=redis_port,
    # redis_password=redis_password,
    redis_list=redis_list,
    duckdb_path="results.duckdb",
)
