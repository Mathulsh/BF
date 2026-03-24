"""结合que_push.py, train.py & data_to_duckdb.py写成一个main.py主运行脚本
潜在风险：redis中任务堆积过多，内存溢出(超1TB时)"""
import time
import pickle
import numpy as np
import os
import sys
import signal
from itertools import combinations, islice
from multiprocessing import Process
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as gbm
import catboost as cat
from math import comb as math_comb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from rd3 import (
    push_to_redis, 
    read_one_from_redis, 
    push_result_to_redis,
    # redis_host, 
    # redis_port, 
    # redis_password,
    redis_list, 
    collect_redis_results_to_duckdb
)
# ① 对应que_push.py
# 添加全局变量用于控制程序运行状态
should_stop = False
current_batch_num = 0  # 记录当前处理的批次号

def signal_handler(signum, frame):
    """处理中断信号"""
    global should_stop, current_batch_num
    print(f"\n收到中断信号，正在保存进度...")
    should_stop = True

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def batch_generator(iterable, batch_size):
    """分割可迭代对象为指定大小的批次避免内存溢出"""
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))  # 获取最多batch_size个元素迭代器
        # 批次为空则结束循环
        if not batch:
            break
        yield batch  # 惰性生成器节省内存

def push_combinations_to_redis():
    """推送数据到Redis的运行脚本"""
    # 检查是否已经完成过推送
    completed_flag_file = "push_completed.txt"
    if os.path.exists(completed_flag_file):
        print("检测到推送已完成，跳过推送步骤。")
        return
    
    print("Starting to push combinations to Redis...")
    time_start = time.time()

    # 检测是否有未完成的推送，如果有就从其+1批次开始
    progress_file = "push_progress.txt"
    start_batch_index = 0
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            try:
                last_completed_batch = int(f.read().strip())
                start_batch_index = last_completed_batch  # 从上次完成的批次号开始
                print(f"检测到上次推送进度，从第 {start_batch_index + 1} 批次开始推送...")
            except ValueError:
                start_batch_index = 0
                print("进度文件格式错误，重新开始...")
    else:
        print("开始新的推送任务...")

    # 按顺序生成特征组合
    whole_numbers: list[int] = list(range(1, 99))
    total = math_comb(len(whole_numbers), 5)
    print(f"总组合数: {total:,}")
    batch_size = 50000  # 每批处理数据量
    
    # 创建批次生成器，从指定索引开始
    def batch_generator_with_start(iterable, batch_size, start_index=0):
        """分割可迭代对象为指定大小的批次，支持从指定索引开始"""
        iterator = iter(iterable)
        
        # 跳过已处理的批次
        for _ in range(start_index):
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
        
        current_index = start_index
        while True:
            batch = list(islice(iterator, batch_size))  # 获取最多batch_size个元素迭代器
            # 批次为空则结束循环
            if not batch:
                break
            yield current_index, batch  # 同时返回索引和批次数据
            current_index += 1
    comb = combinations(whole_numbers, 5)

    for i, batch in batch_generator_with_start(comb, batch_size, start_batch_index):
        # 检查是否收到中断信号
        if should_stop:
            print(f"程序被中断，批次 {i + 1} 未完成，进度已保存")
            # 保存当前进度（但不加1，因为当前批次未完成）
            with open(progress_file, 'w') as f:
                f.write(str(i))
            sys.exit(0)
                
        current_batch_num = i + 1  # 更新当前处理的批次号
        print(f"Pushing batch {i + 1} with {len(batch)} combinations...")
        push_to_redis(batch)
        print(f"Batch {i + 1} completed")
        
        # 保存当前进度（保存已完成的批次号）
        with open(progress_file, 'w') as f:
            f.write(str(i + 1))
        
        # Uncomment the next line to limit batches during testing
        # if i == 9:
        #     break

    # 推送完成，删除进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    # 创建完成标记文件
    with open(completed_flag_file, "w") as f:
        f.write("completed")
        
    print("All combinations have been pushed to Redis!")
    print("Created completion marker file to skip this step on future runs.")
    
    time_end = time.time()
    print(f"Time cost for pushing to Redis: {time_end - time_start} seconds")    
    
# ② 对应train.py
def train_models():
    print("Starting training process...")

    data = pickle.load(open("data_98_3cls_train.pkl", "rb"))
    y = data.values[:, -1]

    while True:
        tasks, raw_task, err, source_redis = read_one_from_redis() # type: ignore

        if err is not None:
            print("Redis error:", err)
            continue   # ❗不要退出
        if source_redis is None:
            continue   # ✅ 消除 Pylance 报错
        if tasks is None or raw_task is None:
            continue   # ❗等待任务

        for task in tasks:
            try:
                X = data.loc[:, task].values

                pipe = Pipeline([
                    ('model', ExtraTreesClassifier(random_state=0))
                ])

                cv = list(StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=42
                ).split(np.zeros(len(y)), y))

                scores = cross_val_score(
                    pipe, X, y, cv=cv, scoring="f1_macro"
                )

                result_data = {
                    "features": task,
                    "mean_f1_macro": float(scores.mean().round(2)),
                }

                # ✅ 写结果 + ACK（必须同一个 Redis）
                push_result_to_redis(
                    r=source_redis, # type: ignore
                    result_data=result_data,
                    raw_task=raw_task # type: ignore
                )

            except Exception as e:
                print("Training error:", e)

                # ❗ 防止死循环（必须 ACK）
                source_redis.lrem("mylist:processing", 1, raw_task) # type: ignore
                continue
            print("task:", task, "Mean F1-macro:", scores.mean().round(2))
        
# ③ 对应data_to_duckdb.py
def collect_results_to_duckdb():
    """Collect results from Redis and save to DuckDB"""
    print("Starting to collect results to DuckDB...")
    collect_redis_results_to_duckdb(
        redis_list=redis_list,
        duckdb_path="results.duckdb",
    )

# ④ 主运行脚本
def main():
    """Main function to orchestrate all processes"""
    print("let's go BF!")
    time_start = time.time()
    # ---------- worker 数 ----------
    num_workers = 1
    if len(sys.argv) > 1:
        try:
            num_workers = int(sys.argv[1])
        except ValueError:
            pass
    print(f"启动 {num_workers} 个 worker")

    # Step 1: Push combinations to Redis
    push_combinations_to_redis()
    
    # Step 2: Train models using n workers
    workers = []

    for i in range(num_workers):
        p = Process(target=train_models, args=())
        p.start()
        workers.append(p)
        print(f"Worker {i} started, pid={p.pid}")

    time_end = time.time()
    print(f"Time cost for training: {time_end - time_start} seconds")
    # ---------- 等待所有 worker ----------
    for p in workers:
        p.join()
    
    # Step 3: Collect results to DuckDB
    collect_results_to_duckdb()
    
if __name__ == "__main__":
    main()

