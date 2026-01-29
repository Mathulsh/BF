"""结合que_push.py, train.py & data_to_duckdb.py写成一个main.py主运行脚本
潜在风险：redis中任务堆积过多，内存溢出(超1TB时)"""
import time
import pickle
import numpy as np
import os
import signal
from itertools import combinations, islice
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from pandas import DataFrame
from numpy import ndarray

from rd3 import (
    push_to_redis, 
    read_one_from_redis, 
    push_result_to_redis,
    redis_host, 
    redis_port, 
    redis_password, 
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
    print(f"批次 {current_batch_num} 未完成，进度已保存")
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
    whole_numbers: list[int] = list(range(1, 44))
    comb = combinations(whole_numbers, 4)

    batch_size = 10000  # 每批处理数据量
    
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
    """训练所有结果的运行脚本"""
    print("Starting training process...")
    time_start = time.time()

    data: DataFrame = pickle.load(open("/workspace/userdata/BF/data_43_3cls_train.pkl", "rb"))
    y = data.values[:, -1]

    # 验证阶段，9次就结束，实际跑的时候，需要while True
    i = 0
    while True:
        # i += 1
        # if i == 6:
        #     break
        task: list[str] | None
        raw_task: bytes | None
        err: Exception | None
        task, raw_task, err = read_one_from_redis()
        print(task, err)
        if err is not None:
            break
        if task is None or raw_task is None:
            # no task retrieved; stop processing
            break
        try:
            X = data.loc[:, task].values
            # 构建流水线，防止全部归一化，造成数据泄漏
            pipe = Pipeline([
                # ('scaler', MinMaxScaler()),
                ('model', RandomForestClassifier(random_state=0))
            ])
            # 分层划分交叉验证
            scoring_name = "f1_macro"
            cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42
                                      ).split(np.zeros(len(y)), y))

            # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores: ndarray = cross_val_score(pipe, X, y, cv=cv, scoring=scoring_name)
            # 训练结果推送到 Redis
            result_data: dict = {
                "features": task,
                "mean_f1_macro": float(cv_scores.mean().round(2)),  # 不mean，尝试1000条，db算均值
            }
            push_result_to_redis(result_data=result_data, raw_task=raw_task)
        except Exception as e:
            print(f"Error during training: {e}")
            continue

    time_end = time.time()
    print(f"Time cost for training: {time_end - time_start}")

# ③ 对应data_to_duckdb.py
def collect_results_to_duckdb():
    """Collect results from Redis and save to DuckDB"""
    print("Starting to collect results to DuckDB...")
    collect_redis_results_to_duckdb(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        queue_name="results",
        duckdb_path="results.duckdb",
        table_name="results"
    )

# ④ 主运行脚本
def main():
    """Main function to orchestrate all processes"""
    print("let's go BF!")
    
    # Step 1: Push combinations to Redis
    push_combinations_to_redis()
    
    # Step 2: Train models using the combinations
    train_models()
    
    # Step 3: Collect results to DuckDB
    collect_results_to_duckdb()
if __name__ == "__main__":
    main()

