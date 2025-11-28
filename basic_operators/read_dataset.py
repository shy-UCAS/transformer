import os
import pandas as pd
# 创建一个示例csv数据集文件
cur_file_path = os.path.dirname(__file__)  # 获取当前文件路径
os.makedirs(os.path.join(cur_file_path, 'data'), exist_ok=True)
data_file = os.path.join(cur_file_path, 'data', 'test_house_data.csv')
print(f'Creating dataset file at: {data_file}') 
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('3,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('3,NA,140000\n')


cur_file_path = os.path.dirname(__file__)  # 获取当前文件路径
os.makedirs(os.path.join(cur_file_path, 'data'), exist_ok=True)
data_file = os.path.join(cur_file_path, 'data', 'test_house_data.xlsx')
print(f'Creating dataset file at: {data_file}')
# 创建数据
data = {
    'NumRooms': ['NA', 2, 4, 'NA'],
    'Alley': ['Pave', 'NA', 'NA', 'NA'],
    'Price': [127500, 106000, 178100, 140000]
}
# 创建 DataFrame
df = pd.DataFrame(data)
# 写入 Excel 文件
df.to_excel(data_file, index=False)