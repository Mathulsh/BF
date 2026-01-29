"""结合que_push.py, train.py & data_to_duckdb.py写成一个main.py主运行脚本
潜在风险：redis中任务堆积过多，内存溢出(超1TB时)"""
import time
import pickle
import numpy as np
from itertools import combinations, islice
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from pandas import DataFrame
from numpy import ndarray
import os
from rd3 import (
    push_to_redis, 
    read_one_from_redis, 
    push_result_to_redis,
    redis_host, 
    redis_port, 
    redis_password, 
    collect_redis_results_to_duckdb
)

# ① 对应data_to_duckdb.py
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

    # 检查是否运行中断过
    flag_file = "push_progress.txt"
    if os.path.exists(flag_file):
        print("检测到推送已在进行中，退出程序。")
        return

    # 创建标记文件
    with open(flag_file, "w") as f:
        f.write("started")

    # 按顺序生成特征组合
    whole_numbers: list[int] = list(range(1, 44))
    comb = combinations(whole_numbers, 4)

    batch_size = 100000  # 每批处理数据量
    for i, batch in enumerate(batch_generator(comb, batch_size)):
        print(f"Pushing batch {i + 1} with {len(batch)} combinations...")
        push_to_redis(batch)
        print(f"Batch {i + 1} completed")
        # Uncomment the next line to limit batches during testing
        # if i == 9:
        #     break

    # 删除进度标记文件
    if os.path.exists(flag_file):
        os.remove(flag_file)
    
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

