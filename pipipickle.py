'''将csv文件转换为pickle文件的脚本'''
import pickle
import pandas as pd # type: ignore

# 去冗余且标准化后的csv文件
# df = pd.read_csv("data/des_model/test-pcc后.csv", encoding='utf-8')

# # 装载pickle数据
# with open("./BF/yasuo-test.pkl", "wb") as f:
#     pickle.dump(df, f)

# 查看pickle数据
with open("BF/yasuo-test.pkl", "rb") as f:
    info = pickle.load(f)
print(info)