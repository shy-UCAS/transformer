import pandas as pd
import os
import numpy as np
import torch
# Series 一维序列
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([10, 20, 30, 40, 50],index=["a", "b", "c", "d", "e"])

'''
# 取值（索引/位置）
print(s2["b"])  # 按索引取值 → 20
print(s2[1])    # 按位置取值 → 20
print(s2[["a", "c"]])  # 多索引取值 → a:10, c:30

# 基本属性
print(s2.index)   # 索引 → Index(['a','b','c','d'], dtype='object')
print(s2.values)  # 值 → array([10,20,30,40])
print(s2.dtype)   # 数据类型 → int64
print(s2.shape)   # 形状 → (4,)

# 数值计算（向量化，无需循环）
print(s2 * 2)     # 所有值×2 → a:20, b:40...
print(s2[s2 > 25])# 条件筛选 → c:30, d:40
'''

# DataFrame 二维表格
# 从字典创建
data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
df1 = pd.DataFrame(data) 
# 自定义索引
df2 = pd.DataFrame(data, index=["membera", "memberb", "memberc"])

"""读取数据"""
# 读取csv
file_path = os.path.join(os.path.dirname(__file__), 'data', 'test_house_data.csv')
df_csv = pd.read_csv(file_path,encoding='gbk')
# 读取excel
file_path = os.path.join(os.path.dirname(__file__), 'data', 'test_house_data.xlsx')
df_excel = pd.read_excel(file_path)
# 读取txt       
file_path = os.path.join(os.path.dirname(__file__), 'data', 'test_house_data.txt')
df_txt = pd.read_csv(file_path)


"""保存数据"""
# df_csv.to_csv(file_path,encoding='gbk')
# df_excel.to_excel(file_path,sheet_name='sheet1')
# df_txt.to_csv(file_path,sep='\t')

"""数据查看"""
# print(df.head(2))  # 前2行（默认前5行）
# print(df.tail(1))  # 最后1行
# print(df.sample(2))# 随机2行（用于大数据集抽样）

"""列选择"""
# # 选单列（返回 Series）
# print(df["姓名"])
# # 选多列（返回 DataFrame）
# print(df[["姓名", "薪资"]])

"""行选择"""
# # loc：行标签 + 列标签
# print(df.loc[1, "薪资"])          # 第1行（索引1）的薪资 → 12000
# print(df.loc[0:2, ["姓名", "城市"]])  # 0-2行，姓名+城市列

# iloc：行位置 + 列位置
# print(df.iloc[1, 3])             # 第1行第3列 → 12000
# print(df.iloc[:3, [0, 2]])       # 前3行，第0、2列（姓名、城市）

# 条件筛选行
# print(df[df["薪资"] > 10000])     # 薪资>10000的行
# print(df[(df["年龄"] > 25) & (df["城市"] == "深圳")])  # 多条件（&且，|或）


"""处理缺失值"""
# df_miss = df_csv.copy()
# df_miss.iloc[0, 1] = np.nan
# df_miss.iloc[1, 2] = np.nan
# df_miss = df_miss.fillna({
#     'NumRooms': 0.0,
#     'Alley': 'NoAlley',
#     'Price': 100000.0
# })
# print(df_miss)

"""处理重复值"""
# # df_dup = pd.concat([df1, df1.iloc[0]])  # 错误：iloc[0] 返回 Series，导致错位
# df_dup = pd.concat([df1, df1.iloc[[0]]])  # 正确：iloc[[0]] 返回 DataFrame，保持结构
# print(df1)
# print(df_dup.duplicated())

# 分离变量的独热编码
inputs = pd.get_dummies(df_csv.iloc[:,0:2],dummy_na=True,dtype=np.int64)
print(inputs)
inputs = torch.tensor(inputs.to_numpy())
print(inputs)

