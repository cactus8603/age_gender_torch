import sys
import os

# 添加上一层目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms

import matplotlib.pyplot as plt

from model import TimmAgeGenderModel

# 定义前处理的 transforms
transform = transforms.Compose([
    transforms.ToPILImage(),  # 将 numpy.ndarray 转为 PIL.Image
    transforms.Resize(256),  # 调整大小为 224x224
    transforms.ToTensor(),  # 转为 PyTorch Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化
])

# 加载自定义的年龄性别预测模型
def init_age_gender_model():
    age_gender_model = TimmAgeGenderModel(model_name='mobilenetv3_small_100')
    age_gender_model.load_checkpoint('../checkpoints/checkpoint_epoch_190.pth.tar')
    # checkpoint = torch.load('../checkpoints/0.8.13/checkpoint_epoch_109.pth.tar', map_location='cuda')
    # checkpoint = torch.load('../checkpoints/checkpoint_epoch_1.pth.tar', map_location='cuda')
    # age_gender_model.load_state_dict(checkpoint["state_dict"], strict=False)
    age_gender_model.eval().to('cuda')
    print('Load Pretrained Model Successful')
    return age_gender_model

# 根据 CSV 文件裁剪区域并进行预测
def process_images_from_csv(csv_path, model):
    # 读取 CSV 文件
    data = pd.read_csv(csv_path, header=None)
    results = []

    cropped_img_dir = "./cropped_images/"

    # 初始化进度条
    with tqdm(total=len(data), desc="Processing images") as pbar:
        for index, row in data.iterrows():
            # (path, x, y, w, h)
            image_path = row[0]
            x, y, w, h = int(row[1]), int(row[2]), int(row[3]), int(row[4])

            # 加载图像
            img = cv2.imread(image_path)
            if img is None:
                print(f"Image not found: {image_path}")
                pbar.update(1)
                continue
                

            # 裁剪区域
            cropped_img = img[y:y+h, x:x+w]
            if cropped_img.size == 0:
                print(f"Invalid crop region for image: {image_path}")
                pbar.update(1)
                continue

            # 提取原始图片的文件名（不包含路径和扩展名）
            basename = os.path.splitext(os.path.basename(image_path))[0]

            # 按照指定格式生成裁剪后的图片文件名
            cropped_img_filename = os.path.join(
                cropped_img_dir, f"{basename}_{x}_{y}_{w}_{h}.jpg"
            )
            cv2.imwrite(cropped_img_filename, cropped_img)

            ### debug ###
            # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

            # # 使用 Matplotlib 显示
            # plt.imshow(cropped_img)
            # plt.title("Cropped Image")
            # plt.axis("off")  # 关闭坐标轴
            # plt.show()
            ### debug ###

            # 预处理
            try:
                face_tensor = transform(cropped_img).unsqueeze(0).to('cuda')
            except Exception as e:
                print(f"Error processing cropped image: {e}")
                pbar.update(1)
                continue

            # 性别和年龄预测
            with torch.no_grad():
                # print(face_tensor.shape)
                gender_logits, age_logits = model(face_tensor)
                # gender = torch.argmax(gender_logits).item()
                # 使用 Softmax 将 logits 转换为概率分布
                gender_probs = F.softmax(gender_logits, dim=-1)  # 计算性别分类的概率
                gender = torch.argmax(gender_probs).item()  # 选择概率最高的类别（0 或 1）
                age = age_logits.item()

            # 保存结果
            results.append({
                'path': image_path,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'predicted_gender':  'M' if gender == 1 else 'F', # '1' if gender == 1 else '0',  #
                'predicted_age': age * 100
            })
            print('predicted_gender:{}, predicted_age:{}'.format(gender, age))

            # 更新进度条
            pbar.update(1)

    # 转换为 DataFrame 并保存
    results_df = pd.DataFrame(results)
    results_df.to_csv('./result/output_csv/predicted_results.csv', index=False, header=False)
    print("Predicted results saved to './result/output_csv/predicted_results.csv'")

# 主函数
if __name__ == "__main__":
    # 初始化模型
    age_gender_model = init_age_gender_model()

    # 输入 CSV 文件路径
    csv_path = './result/output_csv/gtdata_combine.csv'  # 替换为实际路径

    # 处理图片并预测
    process_images_from_csv(csv_path, age_gender_model)
