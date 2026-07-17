'''将csv文件转换为pickle文件的脚本'''
import pickle
import pandas as pd # type: ignore

# 去冗余且标准化后的csv文件
df = pd.read_csv("压缩_train-分层划分.csv")
# 装载pickle数据

with open("./BF/压缩_train-98tz.pkl", "wb") as f:
    pickle.dump(df, f)

# 查看pickle数据
# with open("/Users/lishihong/projects/Research/HEA/BF/压缩_test-97tz.pkl", "rb") as f:
#     info = pickle.load(f)
#     print(info)