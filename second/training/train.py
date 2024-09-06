import os
import torch
import argparse
import numpy as np
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from second.model.second import SECOND
from second.model.losses.total import TotalLoss
from second.utils.model_saver import SaveBestModel
from sklearn.metrics import average_precision_score
from second.dataset.paris_orleans import ORLEANS_VOXELS_Dataset
from second.model.scheduler.custom_exponential import CustomExponentialLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Seed everything

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Weights initialization

def initialize_weights(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight.data)
        module.bias.data.zero_()


# Average Precision (AP)

def compute_average_precision(scores, targets):
    
    _, num_classes, _, _ = scores.shape
    targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
    scores = scores.view(-1, num_classes)
    targets_one_hot = targets_one_hot.view(-1, num_classes)
    
    ap_values = []
    for i in range(num_classes):  # Iterate over each class
        ap = average_precision_score(targets_one_hot[:, i].cpu().numpy(), scores[:, i].cpu().detach().numpy())
        ap_values.append(ap)
    
    return sum(ap_values) / len(ap_values)


# Training function

def train_epoch(epoch, num_epochs, model, optimizer, scheduler, dataloader, criterion):
    model.train()
    model.apply(initialize_weights)
    total_loss = 0
    foc = 0
    reg = 0
    drc = 0
    progress = tqdm(dataloader, total=len(dataloader))

    for voxels, coors, class_pos, targets, directions in progress:
        voxels = voxels.to(device)
        coors = coors.to(device)
        class_pos = class_pos.to(device)
        targets = targets.to(device)
        directions = directions.to(device)

        # Zero the gradients for every batch !
        optimizer.zero_grad()

        # Make predictions for this batch
        score, regression, direction = model(voxels, coors)

        # Compute the loss and its gradients
        train_loss, focal_loss, reg_loss, dir_loss = criterion(
            score, regression, direction, class_pos, targets, directions
        )
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        total_loss += train_loss.item() / len(dataloader)
        foc += focal_loss.item() / len(dataloader)
        reg += reg_loss.item() / len(dataloader)
        drc += dir_loss.item() / len(dataloader)
        progress.set_description(
            f"[{epoch + 1:.2g}/{num_epochs:.2g}] train loss: {total_loss:.3f} || "
            f"Focal Loss: {foc:.3f} || Reg Loss: {reg:.3f} || "
            f"Dir Loss: {drc:.3f}"
        )

    # Update the scheduler
    scheduler.step()

    return total_loss


# Validation function

def validate_epoch(model, dataloader, criterion):
    total_loss = 0
    foc = 0
    reg = 0
    drc = 0
    mAP = 0
    model.eval()

    with torch.no_grad():
        progress = tqdm(dataloader, total=len(dataloader))

        for voxels, coors, class_pos, targets, directions in progress:
            voxels = voxels.to(device)
            coors = coors.to(device)
            class_pos = class_pos.to(device)
            targets = targets.to(device)
            directions = directions.to(device)

            score, regression, direction = model(voxels, coors)
            valid_loss, focal_loss, reg_loss, dir_loss = criterion(
                score, regression, direction, class_pos, targets, directions
            )

            total_loss += valid_loss.item() / len(dataloader)
            foc += focal_loss.item() / len(dataloader)
            reg += reg_loss.item() / len(dataloader)
            drc += dir_loss.item() / len(dataloader)
            
            # Compute AP for the current batch
            batch_ap = compute_average_precision(score, class_pos)
            mAP += batch_ap / len(dataloader)

            progress.set_description(
                f"[VALID] valid loss: {total_loss:.3f} || "
                f"Focal Loss: {foc:.3f} || Reg Loss: {reg:.3f} || "
                f"Dir Loss: {drc:.3f} || mAP: {mAP * 100:.2f} %"
            )

    return total_loss, foc, reg, drc, mAP


if __name__ == '__main__':
    
    # Seed everything
    
    seed_everything(1234)
    
    # Load the data
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", help="session number (you may check the number of available plots)", default=4)
    
    parser.add_argument("--train_dir", help="train set directory", default= "C:/Users/mmedeng/Documents/repositories/NUAGES_DE_POINTS/PAI_ORLEANS/train/voxels")
    parser.add_argument("--valid_dir", help="valid set directory", default="C:/Users/mmedeng/Documents/repositories/NUAGES_DE_POINTS/PAI_ORLEANS/valid/voxels")

    
    parser.add_argument("--train_annot_dir", help="train annotations directory", default="C:/Users/mmedeng/Documents/repositories/NUAGES_DE_POINTS/ANNOTATIONS/train/numpy_arrays")
    parser.add_argument("--valid_annot_dir", help="valid annotations directory", default="C:/Users/mmedeng/Documents/repositories/NUAGES_DE_POINTS/ANNOTATIONS/valid/numpy_arrays")
    args = parser.parse_args()
    
    max_voxels = 20000
    
    data_train = ORLEANS_VOXELS_Dataset(
        data_dir=args.train_dir,
        annotation_dir=args.train_annot_dir,
        max_voxels=max_voxels
    )
    data_valid = ORLEANS_VOXELS_Dataset(
        data_dir=args.valid_dir,
        annotation_dir=args.valid_annot_dir,
        max_voxels=max_voxels
    )
    
    # Hyperparameters definition
    
    batch_size = 1
    num_epochs = 160
    learning_rate = 2e-4
    weight_decay = 1e-4
    exp_decay = 0.8
    decay_step = 15
    
    num_classes = 4
    num_regression_offsets = 7
    num_directions = 2
    in_features = 4
    
    train_loader = DataLoader(
        dataset=data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=data_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Training Loop

    best_loss_val = float("inf")
    saver = SaveBestModel(best_valid_loss=best_loss_val)
    valid, focal, reg, drc, mAPs = [], [], [], [], []

    model = SECOND(
                    in_features=in_features,
                    num_classes=num_classes,
                    num_regression_offsets=num_regression_offsets,
                    num_directions=num_directions,
    )
    model.to(device)

    criterion = TotalLoss(device=device)
    criterion.to(device)

    adam = opt.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    exponential = CustomExponentialLR(
        optimizer=adam,
        decay_factor=exp_decay,
        decay_step=decay_step,
        initial_lr=learning_rate,
    )

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            epoch,
            num_epochs=num_epochs,
            model=model,
            optimizer=adam,
            scheduler=exponential,
            dataloader=train_loader,
            criterion=criterion
        )
        valid_loss, focal_loss, reg_loss, dir_loss, mAP = validate_epoch(
            model=model,
            dataloader=valid_loader,
            criterion=criterion
        )

        if valid_loss < best_loss_val:
            best_loss_val = valid_loss
            saver.save(
                current_valid_loss=best_loss_val,
                epoch=epoch,
                model=model,
                optimizer=adam,
                scheduler=exponential,
                criterion=criterion
            )
            ## save the checkpoints

        valid.append(valid_loss)
        focal.append(focal_loss)
        reg.append(reg_loss)
        drc.append(dir_loss)
        mAPs.append(mAP)
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
    
    ax1.plot(np.arange(1, len(valid) + 1), valid, color='g', label='Combined Loss')
    ax1.plot(np.arange(1, len(focal) + 1), focal, color='y', label='Focal Loss')
    ax1.plot(np.arange(1, len(reg) + 1), reg, color='m', label='Regression Loss')
    ax1.plot(np.arange(1, len(drc) + 1), drc, color='c', label='Direction Loss')
    ax2.plot(np.arange(1, len(mAPs) + 1), mAPs, color='orange')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('SECOND Losses')
    ax1.legend()
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('mean Average Precision')
    
    plt.title('Loss and mAP behaviors')
    plt.tight_layout()
    plt.savefig('C:/Users/mmedeng/Documents/repositories/second/evaluation/training_plots/training_' + str(args.session).rjust(3, '0') + '.png')
