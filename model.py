import os, torch, timm
import torch.nn as nn
from collections import OrderedDict

import torch.nn.functional as F

class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks=2, init_log_vars=[0.0, -2.0]):
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

class TimmAgeGenderModel(nn.Module):
    def __init__(
        self,
        model_name='mobilenetv3_small_100.lamb_in1k',
        pretrained=True,
        head_hidden=128,
        out_indices=(4),   # 前中後多層
        fuse_dim=128, 
        dropout=0.2,
        num_classes_gender=2,
        age_activation=None,  # 用年齡標準化就用 linear 輸出
        phase="val",
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            features_only=True, out_indices=out_indices
        )
        chs = self.backbone.feature_info.channels()  # 各 stage 的通道數
       
        # 將各層: GAP → 1x1 conv 對齊到 fuse_dim
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in chs])
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fuse_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fuse_dim),
                nn.SiLU(inplace=True)
            ) for c in chs
        ])
        fused_dim = fuse_dim * len(chs)

        # self.backbone = timm.create_model(model_name, pretrained=pretrained,
        #                                   num_classes=0, global_pool='avg')
        # C = self.backbone.num_features

        # Gender head（建議：head LR > backbone LR）
        self.gender_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden, bias=False),
            nn.BatchNorm1d(head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 2)
        )


        # Age head
        self.age_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)   # raw linear，配 z-score 使用
        )

        self._shape_checked = False  # 只在第一次 forward 印 shape
   
        if phase == "train":
            self._init_heads()


    def _init_heads(self):
        def _init_head(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # fan_in, ReLU 專用
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.gender_head.apply(_init_head)
        self.age_head.apply(_init_head)

    def fuse_multi_level(self, x):
        feats = self.backbone(x)                 # list of [B,Ci,Hi,Wi]
        vecs = []
        for f, pool, proj in zip(feats, self.pools, self.projs):
            v = proj(pool(f)).flatten(1)         # [B, fuse_dim]
            vecs.append(v)
        fused = torch.cat(vecs, dim=1)           # [B, fuse_dim * L]
        return fused


    def forward(self, x):
        # feat_map = self.backbone.forward_features(x)     # [B, C, H, W]
        # feats = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # → [B, C]
        # # if not self._shape_checked:
        # #     print("feats:", feats.shape, "in_features:", self.in_features)
        # #     self._shape_checked = True
        # gender_logits = self.gender_head(feats)
        # age_logits = self.age_head(feats)
        # return gender_logits, age_logits

        fused = self.fuse_multi_level(x)
        gender_logits = self.gender_head(fused)
        age_logits = self.age_head(fused)           # [B,1]（標準化標籤用這個算 loss）

        return gender_logits, age_logits


    # ---- Save / Load ----
    def save_checkpoint(self, optimizer, scaler, epoch, loss, acc, dir_path,
                        loss_balancer=None, is_best=False):
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f'epoch_{epoch:03d}.pth')
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            # 'config': self.config,
            'loss_balancer': loss_balancer.state_dict() if loss_balancer else None,
            # 'metrics': {'loss': float(loss), 'acc': float(acc)}
        }, path)
        if is_best:
            best_path = os.path.join(dir_path, 'best.pth')
            try:
                import shutil; shutil.copy(path, best_path)
            except Exception:
                torch.save(torch.load(path), best_path)

    @classmethod
    def load_checkpoint(cls, filename, optimizer=None, scaler=None,
                        loss_balancer=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        ckpt = torch.load(filename, map_location=device)
        # cfg  = ckpt.get('config', {})  # 若舊檔沒 config 就用預設
        model = cls().to(device)

        sd = ckpt.get('state_dict') or ckpt.get('model') or ckpt.get('model_state')
        if sd is None:
            raise ValueError('No model weights in checkpoint')
        # 移除 DataParallel 的 "module."
        # new_sd = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v) for k,v in sd.items())
        # model.load_state_dict(new_sd, strict=False)
        model.eval()

        if optimizer is not None and ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            # 把 optimizer state tensor 搬到正確 device
            for st in optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        if loss_balancer is not None and ckpt.get('loss_balancer') is not None:
            loss_balancer.load_state_dict(ckpt['loss_balancer'])

        return model # , ckpt.get('epoch', 0)
