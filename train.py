import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os 
from collections import Counter

from dataset import get_dataloader
from model import TimmAgeGenderModel, UncertaintyWeighting

import torch.optim.lr_scheduler as lr_scheduler

def regularization_loss(model, lambda_reg=1e-4):
    loss = 0.0
    for param in model.parameters():
        if param.requires_grad:  # 仅对需要训练的参数计算正则化
            loss += torch.sum(param ** 2) # l2 loss
            # loss += torch.sum(torch.abs(param)) # l1 loss
    return lambda_reg * loss

# Training Loop
def train_model(model, dataloaders, criterion_gender, criterion_age, optimizer, scaler, scheduler, writer, num_epochs=200):

    # load_state = True
    # filename = './checkpoints/checkpoint_epoch_82.pth'
    # if load_state:
    #     model.load_checkpoint(filename, optimizer, scaler)
    #     print('Load pretrained model successful')

    start_epoch = 0
    checkpoint_path = "./checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Load checkpoint if available
    # try:
    #     start_epoch = model.load_checkpoint(checkpoint_path, optimizer, scaler)[1]
    #     print(f"Resuming training from epoch {start_epoch}")
    # except FileNotFoundError:
    #     print("No checkpoint found, starting from scratch.")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        if epoch < 3:
            for param in model.backbone.parameters():
                param.requires_grad = False

            # 確保 head 可訓練
            for param in model.gender_head.parameters():
                param.requires_grad = True
            for param in model.age_head.parameters():
                param.requires_grad = True

        else:
            for param in model.parameters():
                param.requires_grad = True

        best_epoch_loss = 999.0
        best_acc_gender = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                for p in model.parameters():
                    p.requires_grad = True
            else:
                model.eval()

            running_loss = 0.0
            running_corrects_gender = 0
            age_loss_sum = 0.0
            gender_loss_sum = 0.0  

            with tqdm(total=len(dataloaders[phase]), desc=f"{phase} Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (inputs, gender_labels, age_labels) in enumerate(dataloaders[phase]):
                    inputs        = inputs.to(device, non_blocking=True)
                    gender_labels = gender_labels.to(device, non_blocking=True)         # int 類別
                    age_labels    = age_labels.float().to(device, non_blocking=True)    # 回歸

                    if phase == 'train':
                        optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(phase == 'train'):
                        with autocast():
                            gender_logits, age_logits = model(inputs)
                            age_pred = age_logits.squeeze(-1)  # ✅ 安全擠掉最後一維

                            # losses
                            loss_gender = criterion_gender(gender_logits, gender_labels)
                            loss_age    = criterion_age(age_pred, age_labels)
                            reg_loss    = regularization_loss(model)

                            loss_main, weights = loss_balancer(loss_gender, loss_age)
                            loss = loss_main + reg_loss

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            # LR & 動態權重記錄（可選）
                            if writer is not None:
                                global_step = epoch * len(dataloaders[phase]) + batch_idx
                                writer.add_scalar("Learning Rate/train",
                                                optimizer.param_groups[0]['lr'], global_step)
                                w_gender, w_age = [w.item() for w in weights]
                                writer.add_scalar("Weights/gender", w_gender, global_step)
                                writer.add_scalar("Weights/age",    w_age,    global_step)

                    # 統計
                    running_loss += loss.item() * inputs.size(0)
                    preds = gender_logits.argmax(dim=1)
                    running_corrects_gender += (preds == gender_labels).sum().item()
                    age_loss_sum    += loss_age.item()    * inputs.size(0)
                    gender_loss_sum += loss_gender.item() * inputs.size(0)

                    pbar.update(1)

                # Debug：看最後一個 batch 的分佈（可拿掉）
                gender_probs = torch.softmax(gender_logits, dim=-1).detach().cpu()
                print(gender_labels.detach().cpu())
                print(gender_probs)
                print('--------------------------')

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc_gender = running_corrects_gender / dataset_size
            epoch_loss_age = age_loss_sum / dataset_size
            epoch_loss_gender = gender_loss_sum / dataset_size


            print(f"{phase} Loss: {epoch_loss:.4f} Gender Acc: {epoch_acc_gender:.4f} Age Loss: {epoch_loss_age:.4f}")

            # Write metrics to TensorBoard
            writer.add_scalar(f"{phase} Loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase} Gender Accuracy", epoch_acc_gender, epoch)
            writer.add_scalar(f"{phase} Age Loss", epoch_loss_age, epoch)

            # Save checkpoint only if performance improves
            if phase == 'val' and (epoch_acc_gender > best_acc_gender):
    
                best_acc_gender = epoch_acc_gender  # Update best accuracy
                
                # Save the model checkpoint
                model.save_checkpoint(optimizer, scaler, epoch + 1, loss=epoch_loss, acc=epoch_acc_gender, dir_path='checkpoints', is_best=True)
                print(f"Checkpoint saved at epoch {epoch + 1} with Loss: {epoch_loss:.4f} and Gender Acc: {best_acc_gender:.4f}")
                # with torch.no_grad():
                #     print("α_gender:", torch.softmax(model.gates_gender, dim=0).cpu().numpy())
                #     print("α_age   :", torch.softmax(model.gates_age,    dim=0).cpu().numpy())

            # break
        # Step Scheduler
        scheduler.step()

    print("Training complete")
    writer.close()

if __name__ == '__main__':

    # Training Configuration
    model = TimmAgeGenderModel(model_name='efficientformerv2_s1', phase="train") # mobilenetv3_small_100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, train_list = get_dataloader()
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    print('Load dataloader successful')

    cnt = Counter([g for _, g, _, _ in train_list])    # train_list: [(path, gender, age), ...]
    w = torch.tensor([cnt.get(0,1), cnt.get(1,1)], dtype=torch.float32)
    class_weights = (w.sum() / (2*w)).to(device)

    criterion_gender = nn.CrossEntropyLoss(
        weight=class_weights,           # 不平衡就留著；平衡就設 None
        label_smoothing=0.05            # 避免過度自信
    )
    
    loss_balancer = UncertaintyWeighting(num_tasks=2).to(device)

    # 讓 Optimizer 一起更新權重參數
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_balancer.parameters()),
        lr=5e-5
    )
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)  # Cosine Annealing

    # 调用 train_model 函数
    train_model(
        model=model,
        dataloaders=dataloaders,
        criterion_gender=criterion_gender, # nn.BCEWithLogitsLoss(), # 
        criterion_age=nn.SmoothL1Loss(beta=0.1), # nn.MSELoss(),
        optimizer=optimizer,
        scaler=GradScaler(),
        scheduler=scheduler,
        writer=SummaryWriter(log_dir="logs"),
        num_epochs=200,
    )
