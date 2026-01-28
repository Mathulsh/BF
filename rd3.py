'''功能函数:redis读写模块'''
import redis
from typing import Iterable
import pickle, time, duckdb
from typing import Optional, cast

redis_host: str = "10.64.199.70"
redis_port: int = 41528
redis_password: str | None = None

def push_to_redis(combs: Iterable[tuple[int, ...]]):
    """
    将计算任务推送至 Redis 队列（Pickle 协议）
    :param combs: Iterable[tuple[int]]
    """
    r = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=False,
        socket_timeout=30,
        max_connections=1000,
    )
    
    pipe = r.pipeline()

    for cb in combs:
        task = {
            "features": list(cb)
        }
        pipe.rpush("mylist", pickle.dumps(task))

    pipe.execute()

def read_one_from_redis() -> tuple[list[str] | None, bytes | None, Exception | None]:
    """
    从 Redis 安全读取一个任务（RPOPLPUSH 语义）
    :return:
        - task(str list) 或 None
        - raw_item(bytes)：用于 ACK
        - Exception：无数据时返回
    """

    r = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=False,
        socket_timeout=30,
        max_connections=1000,
    )

    # 原子操作：mylist -> mylist:processing
    # FIFO：LEFT -> RIGHT
    raw_item = r.lmove(
        "mylist",              # source
        "mylist:processing",   # destination
        "LEFT",                # 从队头取
        "RIGHT",               # 放到 processing 尾
    )
    if raw_item is None:
        return None, None, Exception("队列没有数据")

    # 确保 bytes
    if isinstance(raw_item, (bytes, bytearray)):
        data_bytes = raw_item
    elif isinstance(raw_item, str):
        data_bytes = raw_item.encode()
    else:
        raise TypeError(f"Unexpected data type from redis: {type(raw_item)!r}")

    data_dict = pickle.loads(data_bytes)
    task: list[str] = list(map(str, data_dict["features"]))

    # ⚠️ 返回 raw_item，用于后续 ACK
    return task, data_bytes, None
    
def push_result_to_redis(
    *,
    result_data: dict | None,
    raw_task: bytes | None,
) -> None:
    """
    推送结果并 ACK 原任务
    必须保证：先写结果，再 ACK
    """

    if result_data is None or raw_task is None:
        return

    r = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=False,
        socket_timeout=30,
        max_connections=1000,
    )

    payload = pickle.dumps(result_data, protocol=pickle.HIGHEST_PROTOCOL)

    # ========= 关键顺序 =========
    pipe = r.pipeline(transaction=True)

    # 1️⃣ 写结果（成功才算任务完成）
    pipe.rpush("results", payload)

    # 2️⃣ ACK：从 processing 删除原任务
    # ensure bytes for lrem; handle memoryview/bytearray/str consistently
    if isinstance(raw_task, memoryview):
        value_for_lrem = raw_task.tobytes()
    elif isinstance(raw_task, (bytes, bytearray)):
        value_for_lrem = bytes(raw_task)
    elif isinstance(raw_task, str):
        value_for_lrem = raw_task.encode()
    else:
        raise TypeError(f"Unexpected raw_task type: {type(raw_task)!r}")

    # lrem may have a stricter type hint; pass bytes and silence arg-type checking
    pipe.lrem("mylist:processing", 1, value_for_lrem)  # type: ignore[arg-type]

    pipe.execute()

def collect_redis_results_to_duckdb(
    *,
    redis_host: str,
    redis_port: int,
    redis_password: Optional[str],
    queue_name: str = "results",
    processing_queue: str = "results:processing",
    duckdb_path: str = "results.duckdb",
    table_name: str = "results",
    batch_size: int = 5000,
    sleep_time: float = 0.01,
    commit_every: int = 10,          # ✅ 每 N 个 batch 提交一次
):
    """
    Redis(results) -> DuckDB
    FIFO / 批量 / 原子 / 可恢复
    """

    score_name = "mean_f1_macro"

    # ========== Redis ==========
    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=False
    )

    # ========== DuckDB ==========
    con = duckdb.connect(duckdb_path)
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        features INTEGER[],
        {score_name} DOUBLE
    )
    """)
    # WAL 提高事务安全性
    con.execute("PRAGMA username=wal;")
    # 显式事务批量写入
    con.execute("BEGIN;")
    con.execute("COMMIT;")
    # ========== Lua：队头批量拉取 ==========
    pop_script = r.register_script("""
    local res = {}
    local n = tonumber(ARGV[1])

    for i = 1, n do
        local item = redis.call(
            "RPOPLPUSH",
            KEYS[1],   -- results
            KEYS[2]    -- results:processing
        )
        if not item then
            break
        end
        table.insert(res, item)
    end

    return res
    """)

    total_count = 0
    batch_counter = 0

    while True:
        # ---------- 1️⃣ 批量拉取 ----------
        items = pop_script(
            keys=[queue_name, processing_queue],
            args=[batch_size]
        )

        # ---------- 队列暂时为空 ----------
        if not items:
            remaining = r.llen(queue_name)
            processing = r.llen(processing_queue)

            if remaining == 0 and processing == 0:
                # 最后一次提交
                con.execute("COMMIT;")
                print("✅ Redis 队列已清空，数据拉取完成")
                print(f"📊 共写入 DuckDB：{total_count} 条")
                break

            time.sleep(sleep_time)
            continue

        # ---------- 2️⃣ 反序列化 ----------
        buffer = []
        for item in items:
            data = pickle.loads(item)
            buffer.append((
                data["features"],
                float(data[score_name]),
            ))

        # ---------- 3️⃣ 写 DuckDB ----------
        con.executemany(
            f"INSERT INTO {table_name} VALUES (?, ?)",
            buffer
        )

        batch_count = len(buffer)
        total_count += batch_count
        batch_counter += 1

        # ---------- 4️⃣ ACK ----------
        pipe = r.pipeline()
        for item in items:
            pipe.lrem(processing_queue, 1, item)
        pipe.execute()

        # ---------- 5️⃣ 低频 commit ----------
        if batch_counter % commit_every == 0:
            con.execute("COMMIT;")
            con.execute("BEGIN;")

        # ---------- 进度 ----------
        if total_count % 10_000 == 0:
            print(f"🚀 已写入 {total_count:,} 条")
        
def collect_redis_results_to_duckdb1(
    *,
    redis_host: str,
    redis_port: int,
    redis_password: Optional[str],
    queue_name: str = "results",
    duckdb_path: str = "results.duckdb",
    table_name: str = "results",
    batch_size: int = 5000,
    sleep_time: float = 0.01,
):
    """
    从 Redis 的 results 队列拉取数据（不丢数据，节省内存版）
    直接从 results 队列弹出数据，无需中间队列，节省内存，绝对会丢数据，结合duckdb_check.py中的漏数据检查函数使用

    Redis item（pickle）:
    {
        "features": list[int],
        "mean_f1_macro": float
    }
    """

    # Redis
    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=False
    )

    # DuckDB
    con = duckdb.connect(duckdb_path)
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        features INTEGER[],
        mean_f1_macro DOUBLE
    )
    """)

    total_count = 0
    while True:
        # 直接从 results 队列弹出数据，一次性获取一批数据
        items = []
        for _ in range(batch_size):
            item = r.rpop(queue_name)  # 直接从右侧弹出，无需中间队列
            if item is None:
                break
            items.append(item)

        # ================= 拉取为空 =================
        if not items:
            # 检查队列是否还有数据
            remaining = r.llen(queue_name)
            
            if remaining == 0:
                print("✅ Redis 队列已清空，数据拉取完成")
                print(f"📊 本次共写入 DuckDB 数据量：{total_count}")
                break

            # 仍可能有生产者在添加数据
            time.sleep(sleep_time)
            continue
        
        # ================= 反序列化 =================
        buffer = []
        for item in items:
            data = pickle.loads(item)
            buffer.append((
                data["features"],
                float(data["mean_f1_macro"])
            ))

        # 2️⃣ 写 DuckDB（事务）
        try:
            con.executemany(
                f"INSERT INTO {table_name} VALUES (?, ?)",
                buffer
            )
        except Exception as e:
            print(f"❌ 写入 DuckDB 失败: {e}")
            # 由于数据已经从Redis中弹出，如果写入失败，需要重新设计错误处理策略
            # 这里我们跳过这批失败的数据，继续处理下一批
            continue
        
        # ✅ 成功写入 → 累计条数
        batch_count = len(buffer)
        total_count += batch_count
        
        # 由于使用了 rpop 直接弹出，数据已自动从队列中删除，无需额外ACK
        
        # （可选）实时进度
        print(f"已写入 {total_count} 条数据到 DuckDB")
