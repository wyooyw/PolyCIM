import json

# 假设你有一个名为data.json的文件
file_path = 'code.json'

# 打开文件并读取JSON数据
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
