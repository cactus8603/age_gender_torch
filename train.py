import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from model import TimmAgeGenderModel

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
    checkpoint_path = "./checkpoints/checkpoint.pth"

    # Load checkpoint if available
    try:
        start_epoch = model.load_checkpoint(checkpoint_path, optimizer, scaler)[1]
        print(f"Resuming training from epoch {start_epoch}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        best_epoch_loss = 999.0
        best_acc_gender = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
            else:
                model.eval()

            running_loss = 0.0
            running_corrects_gender = 0
            age_loss_sum = 0.0

            # Add progress bar using tqdm
            with tqdm(total=len(dataloaders[phase]), desc=f"{phase} Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (inputs, gender_labels, age_labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    gender_labels = gender_labels.to(device)  # Gender labels
                    age_labels = age_labels.float().to(device)  # Age labels

                    optimizer.zero_grad()

                    with autocast():
                        gender_logits, age_logits = model(inputs)

                        # Compute losses
                        loss_gender = criterion_gender(gender_logits, gender_labels)
                        loss_age = criterion_age(age_logits.squeeze(), age_labels)
                        reg_loss = regularization_loss(model)
                        loss = 100 * loss_gender + 10 * loss_age + reg_loss 

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        # 记录当前的学习率
                        current_lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar("Learning Rate/train", current_lr, epoch * len(dataloaders[phase]) + batch_idx)

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(gender_logits, 1)
                    running_corrects_gender += torch.sum(preds == gender_labels.data)
                    age_loss_sum += loss_age.item() * inputs.size(0)

                    pbar.update(1)
                
                # print training result to check
                gender_probs = torch.softmax(gender_logits, dim=-1)
                print(gender_labels)
                print(gender_probs)
                print('--------------------------')

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc_gender = running_corrects_gender.double() / dataset_size
            epoch_loss_age = age_loss_sum / dataset_size

            print(f"{phase} Loss: {epoch_loss:.4f} Gender Acc: {epoch_acc_gender:.4f} Age Loss: {epoch_loss_age:.4f}")

            # Write metrics to TensorBoard
            writer.add_scalar(f"{phase} Loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase} Gender Accuracy", epoch_acc_gender, epoch)
            writer.add_scalar(f"{phase} Age Loss", epoch_loss_age, epoch)

            # Save checkpoint only if performance improves
            if phase == 'val' and (epoch_loss < best_epoch_loss or epoch_acc_gender > best_acc_gender):
                # Update best metrics
                if epoch_loss < best_epoch_loss:
                    best_epoch_loss = epoch_loss  # Update best loss
                if epoch_acc_gender > best_acc_gender:
                    best_acc_gender = epoch_acc_gender  # Update best accuracy
                
                # Save the model checkpoint
                model.save_checkpoint(optimizer, scaler, epoch + 1, filename=checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch + 1} with Loss: {best_epoch_loss:.4f} and Gender Acc: {best_acc_gender:.4f}")

        # Step Scheduler
        scheduler.step()

    print("Training complete")
    writer.close()

if __name__ == '__main__':

    # Training Configuration
    model = TimmAgeGenderModel(model_name='mobilenetv3_small_100')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader = get_dataloader()
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    print('Load dataloader successful')

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-7)  # Cosine Annealing

    # 调用 train_model 函数
    train_model(
        model=model,
        dataloaders=dataloaders,
        criterion_gender=nn.CrossEntropyLoss(), # nn.BCEWithLogitsLoss(), # 
        criterion_age=nn.MSELoss(),
        optimizer=optimizer,
        scaler=GradScaler(),
        scheduler=scheduler,
        writer=SummaryWriter(log_dir="logs"),
        num_epochs=200
    )
