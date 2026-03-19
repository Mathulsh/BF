'''功能函数:redis读写模块'''
import redis, random
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
    queue_name="results",
    processing_queue="results:processing",
    duckdb_path="results.duckdb",
    table_name="results",
    batch_size=5000,
    sleep_time=0.01,
    commit_every=2,
):
    score_name = "mean_f1_macro"

    # ---------- DuckDB ----------
    con = duckdb.connect(duckdb_path)
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        features INTEGER[],
        {score_name} DOUBLE
    )
    """)

    total_count = 0
    batch_counter = 0

    print("🚀 Collector started (industrial version)")

    while True:
        any_data = False

        # ⚡ 打乱 Redis 顺序（避免热点）
        random.shuffle(redis_list)

        for r in redis_list:
            buffer = []
            raw_items = []

            # ---------- 1️⃣ 原子消费 ----------
            for _ in range(batch_size):
                item = r.rpoplpush(queue_name, processing_queue)
                if item is None:
                    break
                raw_items.append(item)

            if not raw_items:
                continue

            any_data = True

            # ---------- 2️⃣ 解析 ----------
            for item in raw_items:
                try:
                    data = pickle.loads(item)

                    if isinstance(data, list):
                        for d in data:
                            buffer.append((
                                d["features"],
                                float(d[score_name])
                            ))
                    else:
                        buffer.append((
                            data["features"],
                            float(data[score_name])
                        ))
                except Exception as e:
                    # ❗ 坏数据直接丢弃，避免卡死
                    print(f"⚠️ 反序列化失败: {e}")
                    r.lrem(processing_queue, 1, item)

            # ---------- 3️⃣ 写入 DuckDB ----------
            if buffer:
                con.executemany(
                    f"INSERT INTO {table_name} VALUES (?, ?)",
                    buffer
                )

                batch_count = len(buffer)
                total_count += batch_count
                batch_counter += 1

            # ---------- 4️⃣ ACK（成功才删） ----------
            pipe = r.pipeline()
            for item in raw_items:
                pipe.lrem(processing_queue, 1, item)
            pipe.execute()

        # ---------- 空队列检测 ----------
        if not any_data:
            all_empty = True
            for r in redis_list:
                if r.llen(queue_name) > 0 or r.llen(processing_queue) > 0:
                    all_empty = False
                    break

            if all_empty:
                print("✅ 所有 Redis 队列处理完成")
                print(f"📊 总写入：{total_count:,}")
                break

            time.sleep(sleep_time)
            continue

        # ---------- commit ----------
        if batch_counter % commit_every == 0:
            con.commit()

        # ---------- 进度 ----------
        if total_count % 50000 < batch_size:
            print(f"🚀 已写入 {total_count:,}")

    # ---------- 最终提交 ----------
    con.commit()

    # ---------- Top10 ----------
    print("\n🏆 Top 10 Results:")

    try:
        top10 = con.execute(f"""
            SELECT features, {score_name}
            FROM {table_name}
            ORDER BY {score_name} DESC
            LIMIT 10
        """).fetchall()

        for i, (features, score) in enumerate(top10, 1):
            print(f"{i:02d}. score={score:.6f}, features={features}")

    except Exception as e:
        print(f"⚠️ 查询 Top10 失败: {e}")

    con.close()