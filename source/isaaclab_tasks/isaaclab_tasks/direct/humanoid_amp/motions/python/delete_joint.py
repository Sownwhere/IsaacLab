import pandas as pd

# 读取CSV（无表头）
df = pd.read_csv("/home/bowen/RL/humanoid_amp/motions/g1/walk1_subject1.csv", header=None)

# 保留前19列（对应你需要的关节）
df_filtered = df.iloc[:, :19]

# 保存为新文件
df_filtered.to_csv('/home/bowen/RL/humanoid_amp/motions/g1/walk1_subject1_delete.csv', index=False, header=False)
