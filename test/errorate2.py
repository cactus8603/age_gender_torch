import csv
import os
from math import fabs

# 計算 MAE 的輔助函數
def mean_absolute_error(y_true, y_pred):
    return sum(fabs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred)) / len(y_true)

# 讀取標註文件和預測文件
def read_data(file_path):
    data = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # 格式：[?案名稱, x, y, w, h, gender, age]
            filename = os.path.basename(row[0])
            x, y, w, h = map(float, row[1:5])
            gender = int(row[5])
            age = float(row[6])

            if filename not in data:
                data[filename] = []
            data[filename].append((x, y, w, h, gender, age))
    return data

# 比對標註和預測，計算正確率與 MAE loss
def evaluate_accuracy_and_mae(ground_truth_file, prediction_file):
    # 讀取標註和預測資料
    gt_data = read_data(ground_truth_file)
    pred_data = read_data(prediction_file)

    total_matches = 0
    correct_gender = 0
    age_diffs = []

    for filename, gt_rectangles in gt_data.items():
        if filename in pred_data:
            pred_rectangles = pred_data[filename]

            for gt_values in gt_rectangles:
                found = 0
                for pred_values in pred_rectangles:
                    # 計算 x, y, w, h 的差異
                    diff = sum(abs(gt - pred) for gt, pred in zip(gt_values[:4], pred_values[:4]))

                    if diff <= 8:  # 如果差異總和小於等於4，則視為同一筆資料
                        found = 1
                        total_matches += 1

                        # 計算 gender 正確數
                        if gt_values[4] == pred_values[4]:  # gender
                            correct_gender += 1

                        # 計算 age 的 MAE
                        age_diffs.append(abs(gt_values[5] - pred_values[5]))  # age
                        break  # 最初に一致した予測矩形を使用
                if found == 0:
                    print('not found', filename, gt_values)

    # 計算結果
    print('total_matches = ', total_matches)
    gender_accuracy = correct_gender / total_matches if total_matches > 0 else 0
    age_mae = mean_absolute_error([0] * len(age_diffs), age_diffs) if age_diffs else 0

    print(f"Gender Accuracy: {gender_accuracy:.2%}")
    print(f"Age MAE Loss: {age_mae:.2f}")

# 指定標註和預測文件的路徑
ground_truth_file = './result/output_csv/gtdata_combine.csv'
prediction_file = './result/output_csv/predicted_results.csv'

evaluate_accuracy_and_mae(ground_truth_file, prediction_file)
