import pickle
file_path = '/home/junda/diffuser/logs/maze2d-large-v1/diffusion/H384_T256/diffusion_config.pkl'
# /home/junda/diffuser/logs/maze2d-large-v1/diffusion/H384_T256/dataset_config.pkl
# 读取 .pkl 文件
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 打印内容
print(data)