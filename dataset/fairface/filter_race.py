import pandas as pd

# 讀取 CSV 檔案
file_path = "processed_fairface_train.csv"  # 替換為你的 CSV 檔案路徑
data = pd.read_csv(file_path)

# 設定需要保留完整數量的 race
retain_full_race = "East Asian"

# 處理刪減資料
new_data = pd.DataFrame()  # 用來存放處理後的數據
for race, group in data.groupby("race"):
    if race == retain_full_race:
        # 保留完整數據
        new_data = pd.concat([new_data, group], ignore_index=True)
    else:
        # 隨機選擇 1/5 的數據
        reduced_group = group.sample(frac=0.1, random_state=42)  # random_state 確保隨機性可重現
        new_data = pd.concat([new_data, reduced_group], ignore_index=True)

# 儲存新資料到檔案
output_path = "reduced_data_train_10.csv"  # 替換為你想儲存的檔案路徑
new_data.to_csv(output_path, index=False)
print(f"處理後的數據已儲存到 {output_path}")
