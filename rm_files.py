import os

for filename in os.listdir('.'):
    if filename.endswith('-0.png') or filename.endswith('-1.png'):
        try:
            i = int(filename.split('-')[1])  # 提取 i
            if i % 40000 != 0 and i >= 10000:                # 判断是否不整除 4000
                os.remove(filename)          # 删除文件
                print(f"Deleted: {filename}")
        except ValueError:
            pass  # 忽略非数字前缀的文件