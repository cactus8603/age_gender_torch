import csv
import os
import math

# 讀取標註文件和預測文件
def read_data(file_path):
    data = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0].startswith('#'):
                continue # ヘッダをスキップ
            # 格式：[?案名稱, x, y, w, h, gender, age]
            filename = os.path.basename(row[0])
            x, y, w, h = map(float, row[1:5])
            gender = float(row[5])
            age = float(row[6])

            if filename not in data:
                data[filename] = []
            data[filename].append([x, y, w, h, gender, age, 0]) # 最後の要素はmatch用のフラグ
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
                if gt_values[6] == 1:  # マッチ済み
                    continue
                gt_cx = gt_values[0] + gt_values[2] / 2
                gt_cy = gt_values[1] + gt_values[3] / 2
                gt_gender = 0
                if gt_values[4] > 0.5:
                    gt_gender = 1
                min_diff = 10000
                nearest_pidx = -1
                for pidx, pred_values in enumerate(pred_rectangles):
                    if pred_values[6] == 1:  # マッチ済み
                        continue
                    dt_cx = pred_values[0] + pred_values[2] / 2
                    dt_cy = pred_values[1] + pred_values[3] / 2
                    diff = math.sqrt((gt_cx - dt_cx) ** 2 + (gt_cy - dt_cy) ** 2)
                    if diff < min_diff:
                        min_diff = diff
                        nearest_pidx = pidx
                if nearest_pidx != -1:
                    pred_values = pred_rectangles[nearest_pidx]
                    threshold = gt_values[2] / 2
                    dt_gender = 0
                    if pred_values[4] > 0.5:
                        dt_gender = 1
                    if min_diff < threshold and gt_values[2] > pred_values[2] / 2 and pred_values[2] > gt_values[2] / 2:
                        found = 1
                        total_matches += 1
                        gt_values[6] = 1
                        pred_values[6] = 1

                        # 計算 gender 正確數
                        if gt_gender == dt_gender:  # gender
                            correct_gender += 1

                        # 計算 age 的 MAE
                        age_diffs.append(abs(gt_values[5] - pred_values[5]))  # age
                if found == 0:
                    print('not found', filename, gt_values)

    # 計算結果
    print('total_matches = ', total_matches)
    gender_accuracy = correct_gender / total_matches if total_matches > 0 else 0
    age_mae = sum(age_diffs) / len(age_diffs) if age_diffs else 0

    print(f"Gender Accuracy: {gender_accuracy:.2%}")
    print(f"Age MAE Loss: {age_mae:.2f}")

# 指定標註和預測文件的路徑
ground_truth_file = './output_csv/gtdata_combine.csv'
prediction_file = './output_csv/output.csv'

evaluate_accuracy_and_mae(ground_truth_file, prediction_file)
