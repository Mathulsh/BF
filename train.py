"""训练 worker：从 Redis 领取特征组合，评估后回写结果。

单个 worker 的处理流程：
    1. 加载训练集与测试集（pickle，最后一列为标签，其余列按位置作为特征）；
    2. 循环从 Redis 领取任务（一个任务可包含多条组合）；
    3. 对每个组合做 5 折分层交叉验证，再用全量训练集重训，
       计算训练集与测试集的 macro-F1 / accuracy；
    4. 通过 MULTI/EXEC 把结果写入 results 并 ACK 掉在途任务。

仅保留两种基学习器，避免依赖之外的算法：
    rf —— RandomForestClassifier（随机森林）
    gb —— GradientBoostingClassifier（梯度提升）

用法：
    python train.py                # 1 个 worker，RF
    python train.py 4              # 4 个 worker，RF
    python train.py 4 gb           # 4 个 worker，GB
    python train.py 4 gb --idle-exit 60   # 队列空闲 60 秒后自动退出
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from rd import push_result_to_redis, read_one_from_redis

# 屏蔽 numpy 相关 RuntimeWarning，避免多进程日志被刷屏
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 数据集路径：pickle 中为 DataFrame，最后一列为标签
TRAIN_PATH = "yasuo-train.pkl"
TEST_PATH = "yasuo-test.pkl"

# 支持的基学习器
MODEL_CHOICES = ("rf", "gb")
DEFAULT_MODEL = "rf"

# 交叉验证与评分配置：所有组合共用同一划分，保证分数可比
N_SPLITS = 5
CV_RANDOM_STATE = 0
SCORING = ["f1_macro", "accuracy"]


def load_datasets(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """加载训练集与测试集。

    Args:
        train_path: 训练集 pickle 路径。
        test_path: 测试集 pickle 路径。

    Returns:
        tuple: (data_train, data_test, y_train, y_test)，其中前两项为
        ``pandas.DataFrame``（特征列 + 最后一列标签），后两项为标签数组。
    """
    with open(train_path, "rb") as f:
        data_train = pickle.load(f)
    with open(test_path, "rb") as f:
        data_test = pickle.load(f)

    y_train = data_train.values[:, -1]
    y_test = data_test.values[:, -1]
    return data_train, data_test, y_train, y_test


def build_model(name=DEFAULT_MODEL):
    """按名称构建基学习器。

    Args:
        name: ``"rf"``（随机森林）或 ``"gb"``（梯度提升）。

    Returns:
        未训练的 sklearn 估计器。随机森林固定 ``n_jobs=1``，
        把并行度留给 worker 进程，避免多进程之间抢占 CPU。

    Raises:
        ValueError: name 不是受支持的模型。
    """
    key = str(name).lower()
    if key == "rf":
        return RandomForestClassifier(random_state=CV_RANDOM_STATE, n_jobs=1)
    if key == "gb":
        return GradientBoostingClassifier(random_state=CV_RANDOM_STATE)
    raise ValueError(f"不支持的模型: {name!r}，可选 {MODEL_CHOICES}")


def evaluate_combination(feature_idx, data_train, data_test, y_train, y_test, model_name=DEFAULT_MODEL):
    """评估单个特征组合，返回交叉验证与训练/测试指标。

    Args:
        feature_idx: 特征列的位置索引列表（对应 DataFrame 的列位置）。
        data_train: 训练集 DataFrame（含标签列）。
        data_test: 测试集 DataFrame（含标签列）。
        y_train: 训练集标签。
        y_test: 测试集标签。
        model_name: 基学习器名称，``"rf"`` 或 ``"gb"``。

    Returns:
        dict: 结果字典，字段包括 features、feature_idx、cv_f1_macro、
        cv_accuracy、train_f1_macro、train_accuracy、test_f1_macro、
        test_accuracy。
    """
    feature_names = data_train.columns.take(feature_idx).tolist()
    X_train = data_train.iloc[:, feature_idx].values
    X_test = data_test.iloc[:, feature_idx].values

    model = build_model(model_name)

    # 5 折分层交叉验证：固定 random_state，保证不同组合之间可比
    cv = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE).split(
            np.zeros(len(y_train)), y_train
        )
    )
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)

    # 用全部训练集重新训练，再分别评估训练集与留出测试集
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        "features": feature_names,
        "feature_idx": feature_idx,
        "cv_f1_macro": float(np.round(np.mean(scores["test_f1_macro"]), 4)),
        "cv_accuracy": float(np.round(np.mean(scores["test_accuracy"]), 4)),
        "train_f1_macro": float(np.round(f1_score(y_train, train_pred, average="macro"), 4)),
        "train_accuracy": float(np.round(accuracy_score(y_train, train_pred), 4)),
        "test_f1_macro": float(np.round(f1_score(y_test, test_pred, average="macro"), 4)),
        "test_accuracy": float(np.round(accuracy_score(y_test, test_pred), 4)),
    }


def worker(worker_id, model_name=DEFAULT_MODEL, idle_exit=0.0):
    """单个 worker 的训练循环。

    循环领取任务直到收到中断信号，或（在启用 idle_exit 时）队列持续空闲
    idle_exit 秒后自行退出。

    Args:
        worker_id: worker 编号，仅用于日志区分。
        model_name: 基学习器名称，``"rf"`` 或 ``"gb"``。
        idle_exit: 队列空闲多少秒后退出；0 表示一直等待（阻塞式领取）。
    """
    data_train, data_test, y_train, y_test = load_datasets()
    print(
        f"Worker {worker_id} 启动（模型={model_name.upper()}，"
        f"训练集 {len(y_train)} 条，测试集 {len(y_test)} 条）"
    )

    # 仅在启用空闲退出时使用带超时的领取；默认保持原来的阻塞等待
    wait_timeout = 1.0 if idle_exit and idle_exit > 0 else None
    idle_start = None

    try:
        while True:
            tasks, raw_task, err, source_redis = read_one_from_redis(timeout=wait_timeout)

            if err is not None:
                print(f"Worker {worker_id} - Redis error: {err}")
                continue

            if tasks is None or raw_task is None or source_redis is None:
                # 队列为空：只有启用 idle_exit 时才会走到这里
                if wait_timeout is not None:
                    idle_start = idle_start or time.time()
                    if time.time() - idle_start >= idle_exit:
                        print(f"Worker {worker_id} 队列已空闲 {idle_exit:.0f}s，退出")
                        break
                continue

            idle_start = None
            for task in tasks:
                try:
                    feature_idx = [int(t) for t in task]
                    result_data = evaluate_combination(
                        feature_idx, data_train, data_test, y_train, y_test, model_name
                    )

                    # 写结果 + ACK 必须在同一个 Redis 实例上完成
                    push_result_to_redis(
                        r=source_redis,
                        result_data=result_data,
                        raw_task=raw_task,
                    )

                    print(
                        f"Worker {worker_id} "
                        f"model={model_name.upper()} "
                        f"task={result_data['features']} "
                        f"CV_F1={result_data['cv_f1_macro']} "
                        f"CV_ACC={result_data['cv_accuracy']} "
                        f"TRAIN_F1={result_data['train_f1_macro']} "
                        f"TRAIN_ACC={result_data['train_accuracy']} "
                        f"TEST_F1={result_data['test_f1_macro']} "
                        f"TEST_ACC={result_data['test_accuracy']}"
                    )

                except Exception as e:
                    # 与原有实现一致：打印错误但不 ACK，任务会留在 processing 队列
                    print(f"Worker {worker_id} - Training error: {e}")
    except KeyboardInterrupt:
        print(f"Worker {worker_id} 收到中断，退出")


def run_workers(num_workers=1, model_name=DEFAULT_MODEL, idle_exit=0.0):
    """启动并等待 num_workers 个训练 worker。

    ``num_workers == 1`` 时直接在当前进程运行，便于 Ctrl+C 中断；
    大于 1 时用 ``multiprocessing.Process`` 启动，Ctrl+C 会终止全部子进程后返回。

    Args:
        num_workers: worker 进程数。
        model_name: 基学习器名称，``"rf"`` 或 ``"gb"``。
        idle_exit: 队列空闲多少秒后退出；0 表示一直等待。

    Raises:
        ValueError: num_workers 小于 1。
    """
    if num_workers < 1:
        raise ValueError(f"num_workers 必须 >= 1，当前为 {num_workers}")

    if num_workers == 1:
        worker(0, model_name, idle_exit)
        return

    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(i, model_name, idle_exit), name=f"bf-worker-{i}")
        p.start()
        processes.append(p)
        print(f"Worker {i} started, pid={p.pid}, model={model_name.upper()}")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止所有 worker ...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()


def main(argv=None):
    """解析命令行参数并启动训练 worker。

    Args:
        argv: 参数列表；None 表示使用 ``sys.argv[1:]``。

    Returns:
        int: 进程退出码，0 表示正常结束。
    """
    parser = argparse.ArgumentParser(
        description="BF 训练 worker：从 Redis 领取特征组合并评估（仅 RF / GB）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("num_workers", nargs="?", type=int, default=1, help="worker 进程数")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, choices=MODEL_CHOICES, help="基学习器")
    parser.add_argument("--idle-exit", type=float, default=0.0, help="队列空闲多少秒后退出，0 表示一直等待")
    args = parser.parse_args(argv)

    if args.num_workers < 1:
        parser.error("num_workers 必须 >= 1")
    if args.idle_exit < 0:
        parser.error("--idle-exit 不能为负数")

    run_workers(args.num_workers, args.model, args.idle_exit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
