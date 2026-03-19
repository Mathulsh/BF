'''功能函数:redis读写模块'''
import redis, pandas as pd
import pickle, time, duckdb

# redis_host: str = "10.64.199.70"
# redis_port: int = 41528
# redis_password: str | None = None

redis_list = [
    redis.Redis(host="127.0.0.1", port=6379, decode_responses=False),
    redis.Redis(host="127.0.0.1", port=6380, decode_responses=False),
    redis.Redis(host="127.0.0.1", port=6381, decode_responses=False),
    redis.Redis(host="127.0.0.1", port=6382, decode_responses=False),
]
idx = 0

def push_to_redis(combs):
    pipes = [r.pipeline() for r in redis_list]

    for cb in combs:
        task = {"features": list(cb)}

        i = hash(cb) % len(redis_list)
        pipes[i].rpush("mylist", pickle.dumps(task))

    for pipe in pipes:
        pipe.execute()

def read_one_from_redis() -> tuple[list[list[str]] | None, bytes | None, Exception | None, redis.Redis | None]:
    """从多个 Redis 轮询读取任务（真正负载均衡版）"""
    global idx
    n = len(redis_list)

    while True:
        # 🎯 从当前 idx 开始轮询
        for i in range(n):
            r = redis_list[(idx + i) % n]

            try:
                raw_item = r.lmove(
                    "mylist",
                    "mylist:processing",
                    "LEFT",
                    "RIGHT",
                )
                if raw_item is None:
                    continue

                # ✅ 更新轮转起点（关键）
                idx = (idx + i + 1) % n

                # ---------- bytes处理 ----------
                if isinstance(raw_item, (bytes, bytearray)):
                    data_bytes = raw_item
                elif isinstance(raw_item, str):
                    data_bytes = raw_item.encode()
                else:
                    raise TypeError(f"Unexpected data type from redis: {type(raw_item)!r}")


                data = pickle.loads(data_bytes)

                # ---------- batch兼容 ----------
                if isinstance(data, list):
                    tasks = [list(map(str, d["features"])) for d in data]
                else:
                    tasks = [list(map(str, data["features"]))]

                return tasks, data_bytes, None, r

            except Exception as e:
                return None, None, e, None

        # ❗所有 Redis 都空 → sleep
        time.sleep(0.01)
        
def push_result_to_redis(
    *,
    r: redis.Redis | None = None,
    result_data: dict | None,
    raw_task: bytes | None,
) -> None:

    if result_data is None or raw_task is None:
        return

    # 🎯 结果写入：使用指定Redis或负载均衡
    if r is not None:
        target_redis = r
    else:
        target_redis = redis_list[
            hash(raw_task) % len(redis_list)
        ]
    payload = pickle.dumps(result_data, protocol=pickle.HIGHEST_PROTOCOL)

    # 1️⃣ 写结果
    pipe = target_redis.pipeline(transaction=True)
    pipe.rpush("results", payload)

    # 2️⃣ ACK
    if isinstance(raw_task, memoryview):
        value = raw_task.tobytes()
    elif isinstance(raw_task, (bytes, bytearray)):
        value = bytes(raw_task)
    elif isinstance(raw_task, str):
        value = raw_task.encode()
    else:
        raise TypeError(f"Unexpected raw_task type: {type(raw_task)!r}")
    pipe.lrem("mylist:processing", 1, value) # type: ignore
    
    # 3️⃣ 执行
    pipe.execute()
    
def collect_redis_results_to_duckdb(
    redis_list, 
    duckdb_path="results.duckdb",
    ):
    con = duckdb.connect(duckdb_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            features INTEGER[],
            mean_f1_macro DOUBLE
        )
    """)

    print("🚀 Fast collector started")
    total_count = 0
    empty_rounds = 0
    
    while True:
            any_data = False

            for r in redis_list:
                pipe = r.pipeline(transaction=False)

                for _ in range(5000):
                    pipe.rpop("results")

                items = pipe.execute()
                items = [x for x in items if x is not None]

                if not items:
                    continue

                any_data = True

                rows = []
                for item in items:
                    data = pickle.loads(item)
                    rows.append((data["features"], float(data["mean_f1_macro"])))

                total_count += len(rows)  # ✅ 统计

                df = pd.DataFrame(rows, columns=["features", "mean_f1_macro"])

                con.register("tmp_df", df)
                con.execute("INSERT INTO results SELECT * FROM tmp_df")
                con.unregister("tmp_df")

            # =========================
            # ✅ 只加这一段：停止逻辑
            # =========================
            if not any_data:
                empty_rounds += 1
                if empty_rounds >= 20:   # 连续20轮没数据 → 结束
                    print("\n✅ 收集完成")
                    break
                time.sleep(0.1)
            else:
                empty_rounds = 0

    print(f"\n📊 总收集条数: {total_count}")

    top10 = con.execute("""
        SELECT mean_f1_macro, features
        FROM results
        ORDER BY mean_f1_macro DESC
        LIMIT 10
    """).fetchall()

    print("\n🏆 Top 10:")
    for i, (mean_f1_macro, features) in enumerate(top10, 1):
        print(f"{i:01d}. mean_f1_macro={mean_f1_macro:.2f}, features={features}")
