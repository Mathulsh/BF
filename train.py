'''训练所有结果的运行脚本'''
import pickle
import numpy as np
import sys
import multiprocessing as mp
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as gbm
import catboost as cat
from sklearn.model_selection import cross_validate, StratifiedKFold
from rd import read_one_from_redis, push_result_to_redis

data = pickle.load(open("data_99_3cls_train.pkl", "rb"))
y = data.values[:, -1]


def worker(worker_id: int):
    """单个 worker 的训练循环"""
    print(f"Worker {worker_id} 启动")
    
    while True:
        tasks, raw_task, err, source_redis = read_one_from_redis()  # type: ignore

        if err is not None:
            print(f"Worker {worker_id} - Redis error:", err)
            continue
        if source_redis is None:
            continue
        if tasks is None or raw_task is None:
            continue

        for task in tasks:
            try:
                task_int = [int(t) for t in task]
                X = data.loc[:, task_int].values
                model = xgb.XGBClassifier(random_state=0, nthread=1)
                cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
                scoring = ["f1_macro", "accuracy"]
                scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)

                result_data = {
                    "features": task,
                    "mean_f1_macro": float(scores["test_f1_macro"].mean().round(2)),
                    "mean_accuracy": float(scores['test_accuracy'].mean().round(2)),
                }

                push_result_to_redis(
                    r=source_redis,  # type: ignore
                    result_data=result_data,
                    raw_task=raw_task  # type: ignore
                )
                print(f"Worker {worker_id} - task: {task}, f1: {result_data['mean_f1_macro']}, acc: {result_data['mean_accuracy']}")

            except Exception as e:
                print(f"Worker {worker_id} - Training error:", e)


if __name__ == "__main__":
    # 获取 worker 数量，默认为 1
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    
    if num_workers == 1:
        # 单进程模式（兼容原始逻辑）
        worker(0)
    else:
        # 多进程模式
        processes = []
        for i in range(num_workers):
            p = mp.Process(target=worker, args=(i,))
            p.start()
            processes.append(p)
        
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\n停止所有 workers...")
            for p in processes:
                p.terminate()
                p.join()
