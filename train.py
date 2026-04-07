'''训练所有结果的运行脚本'''
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as gbm
import catboost as cat
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from rd import read_one_from_redis, push_result_to_redis
from pandas import DataFrame
from numpy import ndarray

time_start = time.time()

data: DataFrame = pickle.load(open("./data_99_3cls_train.pkl", "rb"))
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
    task, raw_task, err, r = read_one_from_redis() # type: ignore
    print(task, err)
    if err is not None:
        break
    if task is None or raw_task is None:
        # no task retrieved; stop processing
        break
    try:
        task_int = [int(t) for t in task]
        X = data.loc[:, task_int].values
        model = GradientBoostingClassifier(random_state=0)
        cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
        scoring = ["f1_macro", "accuracy"]
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)

        result_data = {
                    "features": task,
                    "mean_f1_macro": float(scores["test_f1_macro"].mean().round(2)),
                    "mean_accuracy": float(scores['test_accuracy'].mean().round(2)),
                }
        push_result_to_redis(result_data=result_data, raw_task=raw_task)
    except Exception as e:
        continue
time_end = time.time()
print(f"Time cost: {time_end - time_start}")

# 四特征组合12w_mean_f1_macro-15节点耗时：7h