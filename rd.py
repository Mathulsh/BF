'''功能函数:redis读写模块'''
import redis, pandas as pd
import pickle, time, duckdb, datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Redis配置——hz
# REDIS_CONFIGS = [
#     {"host": "10.64.199.62", "port": 41882},
#     {"host": "10.64.199.82", "port": 40090},
#     {"host": "10.64.199.63", "port": 41980},
#     {"host": "10.64.199.23", "port": 40091},
# ]

# Redis配置——bj 8个
REDIS_CONFIGS = [
    {"host": "172.19.123.201", "port": 40042},
    {"host": "172.19.123.201", "port": 40043},
    {"host": "172.19.123.201", "port": 40044},
    {"host": "172.19.123.201", "port": 40045},
    {"host": "172.19.123.201", "port": 40057},
    {"host": "172.19.123.201", "port": 40068},
    {"host": "172.19.123.201", "port": 40072},
    {"host": "172.19.123.201", "port": 40073},
]

# Redis配置——本地
# REDIS_CONFIGS = [
#     {"host": "127.0.0.1", "port": 6379},
#     {"host": "127.0.0.1", "port": 6380},
#     {"host": "127.0.0.1", "port": 6381},
#     {"host": "127.0.0.1", "port": 6382},
# ]

def create_redis_clients():
    """创建Redis客户端列表，配置健康检查和超时参数"""
    clients = []
    for config in REDIS_CONFIGS:
        try:
            client = redis.Redis(
                host=config["host"],
                port=config["port"],
                decode_responses=False,
                socket_connect_timeout=10,
                socket_timeout=30,
                health_check_interval=30,
                retry_on_timeout=True,
                max_connections=50,
            )
            # 测试连接
            client.ping()
            clients.append(client)
            logger.info(f"Redis {config['host']}:{config['port']} 连接成功")
        except Exception as e:
            logger.warning(f"Redis {config['host']}:{config['port']} 连接失败: {e}")
            # 创建占位符，后续会尝试重连
            clients.append(None)
    return clients

def get_healthy_redis_clients():
    """获取健康的Redis客户端列表，自动剔除不可用的"""
    healthy = []
    for i, client in enumerate(redis_list):
        if client is None:
            # 尝试重新连接
            try:
                config = REDIS_CONFIGS[i]
                client = redis.Redis(
                    host=config["host"],
                    port=config["port"],
                    decode_responses=False,
                    socket_connect_timeout=10,
                    socket_timeout=30,
                    health_check_interval=30,
                    retry_on_timeout=True,
                )
                client.ping()
                redis_list[i] = client
                healthy.append(client)
                logger.info(f"Redis {config['host']}:{config['port']} 重连成功")
            except Exception as e:
                pass
        else:
            try:
                client.ping()
                healthy.append(client)
            except Exception:
                logger.warning(f"Redis 实例 {i} 不健康，暂时剔除")
                redis_list[i] = None
    return healthy if healthy else []

# 初始化Redis实例
redis_list = create_redis_clients()
idx = 0

def push_to_redis(combs):
    # 过滤掉不可用的 Redis 连接
    healthy_redis_list = [r for r in redis_list if r is not None]
    
    if not healthy_redis_list:
        logger.error("没有可用的 Redis 实例，无法推送任务")
        raise Exception("没有可用的 Redis 实例")
    
    pipes = [r.pipeline() for r in healthy_redis_list]

    for cb in combs:
        task = {"features": list(cb)}

        i = hash(cb) % len(healthy_redis_list)
        pipes[i].rpush("mylist", pickle.dumps(task))

    for pipe in pipes:
        pipe.execute()

def read_one_from_redis() -> tuple[list[list[str]] | None, bytes | None, Exception | None, redis.Redis | None]:
    """从多个 Redis 轮询读取任务（真正负载均衡版，带故障转移）"""
    global idx
    
    while True:
        # 获取健康的Redis客户端
        healthy_clients = get_healthy_redis_clients()
        if not healthy_clients:
            logger.error("所有 Redis 实例均不可用，等待重试...")
            time.sleep(1)
            continue
        
        n = len(healthy_clients)
        
        # 🎯 从当前 idx 开始轮询
        for i in range(n):
            r = healthy_clients[(idx + i) % n]

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

            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"Redis 连接错误，将尝试其他实例: {e}")
                continue
            except Exception as e:
                return None, None, e, None

        # ❗所有 Redis 都空 → sleep
        time.sleep(0.01)
        
def push_result_to_redis(
    *,
    r: redis.Redis | None = None,
    result_data: dict | None,
    raw_task: bytes | None,
    max_retries: int = 3,
) -> bool:
    """
    推送结果到Redis，支持自动重试和故障转移
    
    Returns:
        bool: 是否成功推送
    """
    if result_data is None or raw_task is None:
        return False

    payload = pickle.dumps(result_data, protocol=pickle.HIGHEST_PROTOCOL)

    # 准备ACK值
    if isinstance(raw_task, memoryview):
        value = raw_task.tobytes()
    elif isinstance(raw_task, (bytes, bytearray)):
        value = bytes(raw_task)
    elif isinstance(raw_task, str):
        value = raw_task.encode()
    else:
        raise TypeError(f"Unexpected raw_task type: {type(raw_task)!r}")

    # 🎯 结果写入：使用指定Redis或负载均衡
    if r is not None:
        # 优先尝试原Redis，失败则切换到其他健康实例
        candidates = [r] + [c for c in redis_list if c is not None and c != r]
    else:
        candidates = [c for c in redis_list if c is not None]

    for attempt in range(max_retries):
        for target_redis in candidates:
            if target_redis is None:
                continue
            try:
                # 1️⃣ 写结果
                pipe = target_redis.pipeline(transaction=True)
                pipe.rpush("results", payload)

                # 2️⃣ ACK - 从processing队列移除
                pipe.lrem("mylist:processing", 1, value) # type: ignore
                
                # 3️⃣ 执行
                pipe.execute()
                return True
                
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"Redis 写入失败(尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(0.5 * (attempt + 1))  # 指数退避
                continue
            except Exception as e:
                logger.error(f"Redis 写入未知错误: {e}")
                raise
    
    logger.error(f"所有 Redis 实例写入失败，结果可能丢失: {result_data}")
    return False


def safe_ack_processing(r: redis.Redis | None, raw_task: bytes | None, max_retries: int = 3) -> bool:
    """
    安全地从processing队列移除任务，带重试机制
    用于异常处理时清理队列，避免任务卡住
    
    Returns:
        bool: 是否成功
    """
    if raw_task is None:
        return True
    
    # 准备值
    if isinstance(raw_task, memoryview):
        value = raw_task.tobytes()
    elif isinstance(raw_task, (bytes, bytearray)):
        value = bytes(raw_task)
    elif isinstance(raw_task, str):
        value = raw_task.encode()
    else:
        logger.error(f"未知的raw_task类型: {type(raw_task)}")
        return False
    
    # 候选Redis实例
    if r is not None:
        candidates = [r] + [c for c in redis_list if c is not None and c != r]
    else:
        candidates = [c for c in redis_list if c is not None]
    
    for attempt in range(max_retries):
        for target_redis in candidates:
            if target_redis is None:
                continue
            try:
                target_redis.lrem("mylist:processing", 1, value)  # type: ignore
                return True
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"ACK失败(尝试 {attempt+1}/{max_retries}): {e}")
                time.sleep(0.3 * (attempt + 1))
                continue
            except Exception as e:
                logger.error(f"ACK未知错误: {e}")
                return False
    
    logger.error("无法从processing队列移除任务，任务可能滞留")
    return False
    
def collect_redis_results_to_duckdb(
    redis_list,
    duckdb_path="results.duckdb",
    batch_size=5000,
    idle_timeout=30,

):
    import signal
    import sys
    
    # 全局标志用于控制优雅退出
    should_stop = False
    
    def signal_handler(signum, frame):
        """处理 Ctrl+C 信号"""
        nonlocal should_stop
        print("\n\n🛑 收到中断信号，正在优雅退出...")
        print("   等待当前批次处理完成...")
        should_stop = True
    
    # 注册信号处理器
    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
    
    con = duckdb.connect(duckdb_path)
    con.execute("""
                CREATE TABLE IF NOT EXISTS results(
                    features VARCHAR[],
                    cv_f1_macro DOUBLE,
                    cv_accuracy DOUBLE,
                    train_f1_macro DOUBLE,
                    train_accuracy DOUBLE,
                    test_f1_macro DOUBLE,
                    test_accuracy DOUBLE,
                )
            """)
    # 在外层循环前设置，减少 checkpoint 频率
    con.execute("SET wal_autocheckpoint='100MB'")
    print("🚀 Fast collector started")
    print("   提示: 按 Ctrl+C 可优雅退出\n")
    total_count = 0
    empty_rounds = 0
    t_start = time.time()
    FLUSH_THRESHOLD = 500000  # 攒够多少条再写入，可调整
    rows = []
    try:
        while True:
            any_data = False
            for r in redis_list:
                pipe = r.pipeline(transaction=False)
                for _ in range(batch_size):
                    pipe.rpop("results")
                items = pipe.execute()
                items = [x for x in items if x is not None]
                
                if not items:
                    continue
                
                any_data = True
                now = datetime.datetime.now()
                
                for item in items:
                    data = pickle.loads(item)
                    rows.append((data["features"], float(data["cv_f1_macro"]), float(data["cv_accuracy"]), float(data["train_f1_macro"]), float(data["train_accuracy"]), float(data["test_f1_macro"]), float(data["test_accuracy"])))

                # 每批写入一次
                if len(rows) >= FLUSH_THRESHOLD:
                    n = len(rows)
                    total_count += n
                    df = pd.DataFrame(rows, columns=["features", "cv_f1_macro", "cv_accuracy", "train_f1_macro", "train_accuracy", "test_f1_macro", "test_accuracy"])
                    con.register("tmp_df", df)
                    con.execute("INSERT INTO results SELECT * FROM tmp_df")
                    con.unregister("tmp_df")
                    # 每批打印一行，不覆盖，方便回溯
                    elapsed = time.time() - t_start
                    print(f"  +{n:<7,} → 累计 {total_count:>12,} 条 | {elapsed:6.1f}s", flush=True)
                    rows = []
                    
            # 循环结束后写入剩余数据
            if rows:
                df = pd.DataFrame(rows, columns=["features", "cv_f1_macro", "cv_accuracy", "train_f1_macro", "train_accuracy", "test_f1_macro", "test_accuracy"])
                con.execute("INSERT INTO results SELECT * FROM df")
                total_count += len(rows)
                print(f"  +{len(rows):<7,} → 累计 {total_count:>12,} 条 | 最终写入", flush=True)

            if not any_data:
                empty_rounds += 1
                if empty_rounds >= 20:
                    break
                time.sleep(0.1)
            else:
                empty_rounds = 0
                
            # 检查是否需要优雅退出
            if should_stop:
                print(f"\n⏹️  正在停止收集...")
                break
    
    finally:
        # 恢复原始信号处理器
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        
        # 确保数据写入磁盘
        if 'con' in locals():
            con.commit()
            con.close()
        
        elapsed = time.time() - t_start
        if should_stop:
            print(f"\n🛑 已优雅退出 | 已收集 {total_count:,} 条 | 耗时 {elapsed:.1f}s")
            print(f"   数据已安全保存到: {duckdb_path}")
        else:
            print(f"\n✅ 收集完成 | 总计 {total_count:,} 条 | 耗时 {elapsed:.1f}s")