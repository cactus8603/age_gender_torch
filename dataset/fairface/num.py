import pandas as pd

# 讀取 CSV 檔案
file_path = "reduced_data_train.csv"  # 替換為你的 CSV 檔案路徑
data = pd.read_csv(file_path)

# 統計 race 欄位的種類及數量
race_counts = data['race'].value_counts()

# 輸出結果
print("Race 種類及數量：")
print(race_counts)

# 若要存成新檔案，例如統計結果另存為 CSV：
output_path = "race_counts.csv"
race_counts.to_csv(output_path, header=["Count"])
print(f"結果已儲存到 {output_path}")
