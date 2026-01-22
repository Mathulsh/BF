'''将csv文件转换为pickle文件的脚本'''
import pickle
import pandas as pd # type: ignore

# 去冗余且标准化后的csv文件
df = pd.read_csv("/Users/lishihong/projects/Research/HEA/98_train copy.csv")
# 装载pickle数据

with open("./data_98_3cls_train.pkl", "wb") as f:
    pickle.dump(df, f)

# 查看pickle数据
# with open("/Users/lishihong/projects/Research/HEA/BF/data_43_3cls_train.pkl", "rb") as f:
#     info = pickle.load(f)
#     print(info)