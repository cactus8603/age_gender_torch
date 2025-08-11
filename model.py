import torch
import torch.nn as nn
import timm
import os

class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks=2, init_log_vars=None):
        super().__init__()
        if init_log_vars is None:
            init_log_vars = [0.0] * num_tasks  # 0 → 初始權重相近
        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))

    def forward(self, *losses):
        # Kendall & Gal: sum( exp(-s_i)*L_i + s_i )
        total = 0.0
        ws = []
        for i, L in enumerate(losses):
            s = self.log_vars[i]
            w = torch.exp(-s)
            ws.append(w)
            total = total + w * L + s
        return total, ws  # 回傳總 loss 與目前的權重 w_i（可記錄觀察）

# Model Definition
class TimmAgeGenderModel(nn.Module):
    def __init__(self, model_name='mobilenetv3_small_100.lamb_in1k', hidden_size=256, dropout_rate=0.5, num_classes_gender=2, num_classes_age=1):
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
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),  # 增加一层隐藏层
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes_gender),
            # nn.Sigmoid()
        )

        # Age regression head
        self.age_head = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes_age),
            nn.Sigmoid()  # 输出在 [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        # print(features.shape)
        gender_logits = self.gender_head(features)
        age_logits = self.age_head(features)
        return gender_logits, age_logits

    # Save model checkpoint
    def save_checkpoint(self, optimizer, scaler, epoch, loss, acc, save_path):
        # filename = filename.replace(".pth", f"_epoch_{epoch}.pth")
        save_path = os.path.join(save_path, f'model_{epoch}_{loss:.5f}_{acc:.5f}.pth')
        torch.save({
            'state_dict': self.state_dict(),
            # 'model': self,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch
        }, save_path)

    # Load model checkpoint
    @classmethod
    def load_checkpoint(cls, filename, optimizer=None, scaler=None, device=None):
        checkpoint = torch.load(filename, map_location='cuda')

        # 加载整个模型
        model = cls()
        # model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()  # 如果用于推理，切换到评估模式

        # 加载优化器状态（如果提供）
        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # 加载混合精度状态（如果提供）
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        # 返回模型和起始 epoch
        return model , checkpoint.get('epoch', 0)