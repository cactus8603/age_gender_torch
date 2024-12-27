import cv2
import torch
from torchvision import transforms
from mmpose.apis import init_model, inference_topdown
import numpy as np
from model import TimmAgeGenderModel

# 初始化 RTMpose Wholebody 模型
def init_rtmpose_model():
    config_file = 'rtmo/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py'  # 替换为实际路径
    checkpoint_file = 'rtmo/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth'  # 替换为实际路径
    pose_model = init_model(config_file, checkpoint_file, device='cuda:0')
    return pose_model

# 加载自定义的年龄性别预测模型
def init_age_gender_model():
    age_gender_model = TimmAgeGenderModel()
    checkpoint = torch.load('./checkpoints/0.8.13/checkpoint_epoch_74.pth.tar', map_location='cuda')
    age_gender_model.load_state_dict(checkpoint["state_dict"])
    age_gender_model.eval().to('cuda')
    print('Load Pretrained Model Successful')
    return age_gender_model

# 定义前处理的 transforms
transform = transforms.Compose([
    transforms.ToPILImage(),  # 将 numpy.ndarray 转为 PIL.Image
    transforms.Resize(224),  # 调整大小为 224x224
    transforms.ToTensor(),  # 转为 PyTorch Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化
])

# 根据全身关键点计算人脸边界框
def get_face_bbox_from_keypoints(keypoints, face_indices):
    # print(f"Keypoints: {keypoints}")
    # print(f"Keypoints shape: {keypoints.shape}")
    x_coords = keypoints[face_indices, 0]
    y_coords = keypoints[face_indices, 1]
    x1, y1 = int(min(x_coords)), int(min(y_coords))
    x2, y2 = int(max(x_coords)), int(max(y_coords))
    return [x1, y1, x2, y2]

# 从全身关键点检测人脸并绘制结果
def detect_faces_and_predict(image_path, pose_model, age_gender_model, output_path="output_image.jpg"):
    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 推理 RTMpose Wholebody 检测全身关键点
    inference_results = inference_topdown(pose_model, img_rgb)

    if not inference_results:
        print("No persons detected.")
        return

    # 定义人脸关键点的索引（眼睛、鼻子、嘴巴）
    face_indices = list(range(0, 25))  # 0~24 是人脸关键点在 Wholebody 模型中的索引

    for person in inference_results:
        # 从 pred_instances 提取关键点
        keypoints = person.pred_instances.keypoints

        # 提取单个人的关键点（去掉批次维度）
        keypoints = keypoints[0]

        # 确保关键点数量足够
        if keypoints.shape[0] < max(face_indices) + 1:
            print(f"Insufficient keypoints for face detection. Keypoints shape: {keypoints.shape}")
            continue

        # 获取关键点的 x 和 y 坐标
        x_coords = keypoints[face_indices, 0]
        y_coords = keypoints[face_indices, 1]

        # 计算人脸边界框
        bbox = get_face_bbox_from_keypoints(keypoints, face_indices)

        # 裁剪人脸区域
        x1, y1, x2, y2 = bbox
        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # 使用 transforms 进行前处理
        try:
            face_tensor = transform(face).unsqueeze(0).to('cuda')
        except Exception as e:
            print(f"Error processing face: {e}")
            continue

        # 预测年龄和性别
        with torch.no_grad():
            gender_logits, age_logits = age_gender_model(face_tensor)
            gender = torch.argmax(gender_logits).item()
            age = age_logits.item()

        # 绘制边界框和标签
        label = f"{'M' if gender == 1 else 'F'} {age:.1f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色框
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # 保存结果图像
    # cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")

# 主函数
if __name__ == "__main__":
    # 初始化模型
    rtmpose_model = init_rtmpose_model()
    age_gender_model = init_age_gender_model()

    # 输入图片路径
    input_image_path = './test/imags/0000.jpg'  # 替换为实际路径
    output_image_path = './result/annotated_image.jpg'

    # 检测并标注
    detect_faces_and_predict(input_image_path, rtmpose_model, age_gender_model, output_path=output_image_path)
