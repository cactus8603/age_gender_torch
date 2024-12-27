import torch
import torch.nn as nn
import timm

# Model Definition
class TimmAgeGenderModel(nn.Module):
    def __init__(self, model_name='mobilenetv4_conv_small.e2400_r224_in1k', hidden_size=256, dropout_rate=0.5, num_classes_gender=2, num_classes_age=1):
        super(TimmAgeGenderModel, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        in_features = 1024 # self.backbone.num_features

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
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch
        }, filename)

    # Load model checkpoint
    @classmethod
    def load_checkpoint(cls, filename, optimizer=None, scaler=None):
        checkpoint = torch.load(filename)
        model = cls()
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scaler:
            scaler.load_state_dict(checkpoint['scaler'])
        return model, checkpoint['epoch']