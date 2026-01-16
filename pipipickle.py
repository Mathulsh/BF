'''将csv文件转换为pickle文件的脚本'''
import pickle
import pandas as pd # type: ignore

# 去冗余后的csv文件
# df = pd.read_csv("/Users/lishihong/projects/Research/HEA/43_4class_cls copy.csv")
# # 装载pickle数据

# with open("./data_43_4cls.pkl", "wb") as f:
#     pickle.dump(df, f)

# 查看pickle数据
with open("/Users/lishihong/projects/Research/HEA/acln/src/acln/data_43_4cls.pkl", "rb") as f:
    info = pickle.load(f)
    print(info)