import torch
import torch.nn as nn
import timm

# Model Definition
class TimmAgeGenderModel(nn.Module):
    def __init__(self, model_name='mobilenetv3_small_100.lamb_in1k', hidden_size=128, dropout_rate=0.2, num_classes_gender=2, num_classes_age=1):
        super(TimmAgeGenderModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        # in_features = self.backbone.num_features
        # 动态获取 in_features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)  # 假设输入图像大小为 224x224
            dummy_features = self.backbone(dummy_input)
            in_features = dummy_features.shape[1]
        

        # Gender classification head
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加 Dropout
            nn.Linear(hidden_size, num_classes_gender),
        )

        # Age regression head
        self.age_head = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 添加 Dropout
            nn.Linear(hidden_size, num_classes_age),
        )

    def forward(self, x):
        features = self.backbone(x)
        # print(features.shape)
        gender_logits = self.gender_head(features)
        age_logits = self.age_head(features)
        return gender_logits, age_logits

    # Save model checkpoint
    def save_checkpoint(self, optimizer, scaler, epoch, filename="checkpoint.pth.tar"):
        filename = filename.replace(".pth.tar", f"_epoch_{epoch}.pth.tar")
        torch.save({
            # 'state_dict': self.state_dict(),
            'model': self,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch
        }, filename)

    # Load model checkpoint
    @classmethod
    def load_checkpoint(cls, filename, optimizer=None, scaler=None):
        checkpoint = torch.load(filename)

        # 加载整个模型
        model = checkpoint['model']
        model.eval()  # 如果用于推理，切换到评估模式

        # 加载优化器状态（如果提供）
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # 加载混合精度状态（如果提供）
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        # 返回模型和起始 epoch
        return model, checkpoint.get('epoch', 0)