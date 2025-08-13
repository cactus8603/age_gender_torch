import os, torch, timm
import torch.nn as nn
from collections import OrderedDict

import torch.nn.functional as F

# class UncertaintyWeighting(nn.Module):
#     def __init__(self, num_tasks=2, init_log_vars=[0.0, -2.0]):
#         super().__init__()
#         if init_log_vars is None:
#             init_log_vars = [0.0] * num_tasks  # 0 → 初始權重相近
#         self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))

#     def forward(self, *losses):
#         # Kendall & Gal: sum( exp(-s_i)*L_i + s_i )
#         total = 0.0
#         ws = []
#         for i, L in enumerate(losses):
#             s = self.log_vars[i]
#             w = torch.exp(-s)
#             ws.append(w)
#             total = total + w * L + s
#         return total, ws  # 回傳總 loss 與目前的權重 w_i（可記錄觀察）

class UncertaintyWeighting(nn.Module):
    def __init__(self, num_tasks=2, init_log_vars=None, auto_init=True):
        super().__init__()
        self.auto_init = auto_init
        self.initialized = not auto_init  # 如果不自動初始化就直接完成
        if init_log_vars is None:
            init_log_vars = [0.0] * num_tasks
        self.log_vars = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))

    def forward(self, *losses):
        # 第一次 forward 時，自動根據 loss 大小設定 log_vars
        if self.auto_init and not self.initialized:
            with torch.no_grad():
                # 用 loss 比例決定 log_vars（Kendall & Gal 初始化技巧）
                base_loss = losses[0].detach()
                for i in range(len(losses)):
                    ratio = losses[i].detach() / (base_loss + 1e-8)
                    self.log_vars[i] = torch.log(ratio)
            self.initialized = True

        total = 0.0
        ws = []
        for i, L in enumerate(losses):
            s = self.log_vars[i]
            w = torch.exp(-s)
            ws.append(w)
            total = total + w * L + s
        return total, ws

class ResBlock(nn.Module):
    def __init__(self, dim, hidden, drop=0.2, act=nn.SiLU):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln  = nn.LayerNorm(dim)
        self.act = act()
        self.drop = nn.Dropout(drop)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.ln(x + h)

# ---- Age head：A) 殘差 MLP 版本（預設） ----
class AgeHeadResMLP(nn.Module):
    def __init__(self, in_dim, width=512, depth=3, drop=0.2, act=nn.SiLU):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, width),
            act(), nn.Dropout(drop),
        )
        nn.init.kaiming_normal_(self.in_proj[1].weight, nonlinearity='relu')
        nn.init.zeros_(self.in_proj[1].bias)

        self.blocks = nn.Sequential(*[
            ResBlock(width, width*2, drop=drop, act=act) for _ in range(depth)
        ])

        self.out = nn.Sequential(
            nn.Linear(width, width//2), act(), nn.Dropout(drop),
            nn.Linear(width//2, 1)  # 線性輸出（配 z-score 使用）
        )
        for m in self.out:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out(x)  # [B, 1]

class GenderHeadResMLP(nn.Module):
    def __init__(self, in_dim, width=512, depth=2, drop=0.2, act=nn.SiLU):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, width),
            act(), nn.Dropout(drop),
        )
        nn.init.kaiming_normal_(self.in_proj[1].weight, nonlinearity='relu'); nn.init.zeros_(self.in_proj[1].bias)

        self.blocks = nn.Sequential(*[ResBlock(width, width*2, drop, act) for _ in range(depth)])

        self.out = nn.Sequential(
            nn.Linear(width, width//2), act(), nn.Dropout(drop),
            nn.Linear(width//2, 2)   # ← 2 類 logits，搭配 CrossEntropyLoss
        )
        for m in self.out:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out(x)  # [B, 2]

def _init_all_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class TimmAgeGenderModel(nn.Module):
    def __init__(
        self,
        model_name='mobilenetv3_large_100.ra_in1k',
        pretrained=True,
        head_hidden=256,
        out_indices=(4),   # 前中後多層
        fuse_dim=256, 
        dropout=0.4,
        num_classes_gender=2,
        age_activation=None,  # 用年齡標準化就用 linear 輸出
    ):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0, global_pool='avg')

        # ---- 這裡用 dummy forward 探測實際輸出維度 C_real ----
        self.img_size = 224
        with torch.no_grad():
            # 從 default_cfg 取輸入尺寸，取不到就用 img_size
            cfg = getattr(self.backbone, 'default_cfg', {})
            H = cfg.get('input_size', (3, self.img_size, self.img_size))[1]
            dummy = torch.zeros(1, 3, H, H)
            out = self.backbone(dummy)
            C = out.shape[-1]          # ← 以實際輸出維度為準（避免 576/1024 不一致）

        # C = self.backbone.num_features

        self.pre_norm = nn.LayerNorm(C)

        self.trunk = nn.Sequential(
            nn.Linear(C, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResBlock(head_hidden, hidden=head_hidden*2, drop=dropout)  # 1 個就很有感
        )

        # Gender head（建議：head LR > backbone LR）
        self.gender_head = GenderHeadResMLP(head_hidden, width=256, depth=3, drop=dropout)


        # Age head
        self.age_head = AgeHeadResMLP(head_hidden, width=256, depth=3, drop=dropout)

        self._shape_checked = False  # 只在第一次 forward 印 shape


    def init_heads(self):
        # 只初始化 heads（常見於用 ImageNet backbone 微調）
        self.gender_head.apply(_init_all_linear)
        self.age_head.apply(_init_all_linear)



    def forward(self, x):
        f = self.backbone(x)           # [B, C]
        f = self.pre_norm(f)
        h = self.trunk(f)              # [B, H]

        g = self.gender_head(h)
        a = self.age_head(h)

        return g, a

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

        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # if sd is None:
        #     raise ValueError('No model weights in checkpoint')
        # 移除 DataParallel 的 "module."
        # new_sd = OrderedDict((k[7:], v) if k.startswith('module.') else (k, v) for k,v in sd.items())
        # print(filename)
        model.load_state_dict(sd, strict=True)
        
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
