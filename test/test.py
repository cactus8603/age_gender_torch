import sys
import os

# 添加上一层目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
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
    age_gender_model.load_checkpoint('../checkpoints/checkpoint_epoch_11.pth.tar')
    # checkpoint = torch.load('../checkpoints/0.8.13/checkpoint_epoch_109.pth.tar', map_location='cuda')
    # checkpoint = torch.load('../checkpoints/checkpoint_epoch_1.pth.tar', map_location='cuda')
    # age_gender_model.load_state_dict(checkpoint["state_dict"], strict=False)

    for name, param in age_gender_model.gender_head.named_parameters():
        print(f"Gender Head - Parameter: {name}, Requires Grad: {param.requires_grad}")

    age_gender_model.eval().to('cuda')
    print('Load Pretrained Model Successful')
    return age_gender_model

# 根据 CSV 文件裁剪区域并进行预测
def process_images_from_csv(img_path, model):

    ### debug ###
    # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

    # # 使用 Matplotlib 显示
    # plt.imshow(cropped_img)
    # plt.title("Cropped Image")
    # plt.axis("off")  # 关闭坐标轴
    # plt.show()
    ### debug ###
    img = cv2.imread(img_path)

    # 预处理
    try:
        face_tensor = transform(img).unsqueeze(0).to('cuda')
    except Exception as e:
        print(f"Error processing cropped image: {e}")
        

    # 性别和年龄预测
    with torch.no_grad():
        # print(face_tensor.shape)
        gender_logits, age_logits = model(face_tensor)
        # gender = torch.argmax(gender_logits).item()
        # 使用 Softmax 将 logits 转换为概率分布
        gender_probs = F.softmax(gender_logits, dim=-1)  # 计算性别分类的概率
        gender = torch.argmax(gender_probs).item()  # 选择概率最高的类别（0 或 1）
        age = age_logits.item()


    print('predicted_gender:{}, predicted_age:{}'.format(gender, age))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 主函数
if __name__ == "__main__":

    set_seed(43)

    # 初始化模型
    age_gender_model = init_age_gender_model()

    img_path = './test_img/0009_667_397_118_118.jpg'
    # folder = './cropped_images'
    folder = './test_img'
    
    img_paths = glob(os.path.join(folder, '*.jpg'))

    for img_path in img_paths:

        # 处理图片并预测
        process_images_from_csv(img_path, age_gender_model)
