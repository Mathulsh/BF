'''训练所有结果的运行脚本'''
import time
import pickle
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from rd import read_one_from_redis, push_result_to_redis
from pandas import DataFrame
from numpy import ndarray

time_start = time.time()

data: DataFrame = pickle.load(open("./data_43_3cls_train.pkl", "rb"))
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
            "mean_f1_macro": float(cv_scores.mean().round(2)), # 不mean，尝试1000条，db算均值
        }
        push_result_to_redis(result_data=result_data, raw_task=raw_task)
    except Exception as e:
        continue
time_end = time.time()
print(f"Time cost: {time_end - time_start}")

# 四特征组合12w_mean_f1_macro-15节点耗时：7h