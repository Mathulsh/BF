'''训练所有结果的运行脚本'''
import pickle
import numpy as np
import sys
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import catboost as cat
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score
from rd import read_one_from_redis, push_result_to_redis
import warnings
from sklearn.exceptions import ConvergenceWarning
# 屏蔽 sklearn 收敛警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# 屏蔽 numpy 相关 RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


data_train = pickle.load(open("压缩_train-98tz.pkl", "rb"))
data_test = pickle.load(open("压缩_test-98tz.pkl", "rb"))
y_train = data_train.values[:, -1]
y_test = data_test.values[:, -1]

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
                feature_idx = [int(i) for i in task]
                feature_names = data_train.columns.take(feature_idx).tolist()
                X_train = data_train.iloc[:, feature_idx].values
                X_test = data_test.iloc[:, feature_idx].values
                
                model = RandomForestClassifier(random_state=0,n_jobs=1) # 修改算法
                cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y_train)), y_train))
                scoring = ["f1_macro", "accuracy"]
                scores = cross_validate(
                    model, 
                    X_train, 
                    y_train, 
                    cv=cv, 
                    scoring=scoring, 
                    n_jobs=1) # type: ignore

                # ==========================
                # 利用全部训练集重新训练
                # ==========================
                
                # 训练集
                model.fit(X_train, y_train)
                train_f1 = f1_score(
                    y_train,
                    model.predict(X_train),
                    average="macro"
                )
                train_acc = accuracy_score(
                    y_train,
                    model.predict(X_train)
                )
                
                # 测试集
                y_pred = model.predict(X_test)
                test_f1 = f1_score(
                    y_test,
                    y_pred,
                    average="macro"
                )
                test_acc = accuracy_score(
                    y_test,
                    y_pred
                )

                result_data = {
                    "features": feature_names,
                    "feature_idx": feature_idx,
                    "cv_f1_macro": float(np.round(np.mean(scores["test_f1_macro"]), 4)),
                    "cv_accuracy": float(np.round(np.mean(scores["test_accuracy"]), 4)),
                    "train_f1_macro": float(np.round(train_f1, 4)),
                    "train_accuracy": float(np.round(train_acc, 4)),
                    "test_f1_macro": float(np.round(test_f1, 4)),
                    "test_accuracy": float(np.round(test_acc, 4)),
                }

                push_result_to_redis(
                    r=source_redis,
                    result_data=result_data,
                    raw_task=raw_task
                )

                print(
                    f"Worker {worker_id} "
                    f"task={feature_names} "
                    f"CV_F1={result_data['cv_f1_macro']} "
                    f"CV_ACC={result_data['cv_accuracy']} "
                    f"TRAIN_F1={result_data['train_f1_macro']} "
                    f"TRAIN_ACC={result_data['train_accuracy']} "
                    f"TEST_F1={result_data['test_f1_macro']} "
                    f"TEST_ACC={result_data['test_accuracy']}"
                )

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
