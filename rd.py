'''功能函数:redis读写模块'''
import os
import redis
from typing import Iterable, Tuple, List
import pickle, time, duckdb
from typing import Optional, cast

redis_host: str = os.getenv("REDIS_HOST", "redis-17122.c8.us-east-1-4.ec2.cloud.redislabs.com")
redis_port: int = int(os.getenv("REDIS_PORT", 17122))
redis_password: str = os.getenv("REDIS_PASSWORD", "BQOmVL2fC0SfXCuO2NjJFvNriLAzbsp0")

def push_to_redis(combs: Iterable[tuple[int, ...]]):
    """
    将计算任务推送至 Redis 队列（Pickle 协议）
    :param combs: Iterable[tuple[int]]
    """
    r = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        db=0,
        password=redis_password,
        username="default",
        decode_responses=False,
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
        decode_responses=False,
        username="default",
        db=0,
        password=redis_password,
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
        db=0,
        password=redis_password,
        decode_responses=False,
        username="default",
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
    queue_name: str = "results",                 # ✅ 明确：只从 results 拉
    processing_queue: str = "results:processing",
    duckdb_path: str = "results_(43,4)_3分类.duckdb",
    table_name: str = "results",
    batch_size: int = 5000,
    sleep_time: float = 0.01,
):
    """
    从 Redis 的 results 队列拉取数据（不丢数据，高速版）

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

    # Lua：从 results → results:processing（原子、批量）
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
    while True:
        # 1️⃣ 原子批量拉取
        items = cast(list, pop_script(
            keys=[queue_name, processing_queue],
            args=[batch_size]
        ))

        if not items:
            time.sleep(sleep_time)
            continue
        
        # ================= 拉取为空 =================
        if not items:
            # 🔍 判断是否真的“拉取完”
            remaining = r.llen(queue_name)
            processing = r.llen(processing_queue)

            if remaining == 0 and processing == 0:
                print("✅ Redis 队列已清空，数据拉取完成")
                print(f"📊 本次共写入 DuckDB 数据量：{total_count}")
                break

            # 仍可能有生产者或处理中数据
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
        except Exception:
            # ❌ 写失败：不 ACK，数据仍在 processing_queue
            raise
        
        # ✅ 成功写入 → 累计条数
        batch_count = len(buffer)
        total_count += batch_count
        
        # 3️⃣ ACK：确认成功后，从 processing 删除
        pipe = r.pipeline()
        for item in items:
            pipe.lrem(processing_queue, 1, item)
        pipe.execute()
        
        # （可选）实时进度
        print(f"已写入 {total_count} 条")